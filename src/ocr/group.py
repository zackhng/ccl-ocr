"""Grouping: character candidates into words and line segments (Phase 5).

Everything after this stage works on words and lines, not loose components. The
recogniser reads a word at a time (that is where spaces come from, and where a
character language model has context to use), script tagging works on line crops,
and PII masking needs word and line boxes.

**Run-length smoothing on glyph cores.** Each full-height glyph paints the middle
``core_fraction`` of its box into a mask; the mask is dilated horizontally by the line
gap and labelled. Every label is one line *segment*. A gap wider than the dilation
breaks a row, which is what a form or an ID card needs (``RACE`` and ``SEX`` share a
row but are separate fields) and what the ground truth annotates. The middle band is
what makes this safe: the cores of a capital, an x-height letter and a descender on
one line still overlap vertically, while a descender cannot reach the line below.
Mild skew still chains, because neighbouring glyphs' cores overlap. All of the
per-pixel work is OpenCV; a pure-Python neighbour search measured P95 70-100 ms on
pages with thousands of components.

**Small marks join their anchor.** A diacritic sits above the core band and a full stop
below it, so neither builds lines. They join the word of the glyph the filter anchored
them to (:attr:`Component.anchor_id`).

**Junk lines are set aside, not deleted.** About a quarter of surviving components sit
on no text (Phase 6). Left alone they form lines of their own and halve line
precision. A line failing a junk gate goes to :attr:`PageResult.rejected_lines` with the
gate's name. Geometry cannot reliably tell a row of small print ("AUTHORISED
SIGNATORY") from a texture; the median-height gate trades ~4 points of line recall on
photos for ~7 of precision. So the recogniser, which can tell them apart, gets the final
say.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import GroupConfig
from .timing import StageTimer
from .types import BBox, Component, ComponentKind, Line, Word


MASK_PX_PER_GLYPH = 5.0
"""Resolution of the line mask, in pixels per glyph height. Line structure needs no
more: on the held-out set, 4, 5, 6, 8 and 12 px give line and word F1 within half a
point of each other, while the mask (and its cumsums, dilation and CCL) shrinks with
the square. Core bands are rounded *outward*. Rounding them inward was tried, to keep
adjacent lines apart at low resolution, and cost 2-3 points of word F1: shrunken cores
of neighbouring glyphs on slightly skewed text stop overlapping and lines break
mid-word. Adjacent lines did not fuse with outward rounding in any measured setting."""


@dataclass(slots=True)
class GroupStats:
    glyph_height: float = 0.0
    lines: int = 0
    words: int = 0
    junk_lines: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _union(boxes: list[BBox]) -> BBox:
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


def reading_order(boxes: list[BBox]) -> list[int]:
    """Indices of ``boxes`` in reading order: rows top to bottom, left to right.

    A plain (y, x) sort is not enough: two halves of one line can differ in y by a
    pixel and would come out right-half-first. A box joins the current row while its
    vertical centre is within half a height of the row's first box.
    """
    rows: list[list[int]] = []
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i].cy):
        b = boxes[i]
        if rows:
            ref = boxes[rows[-1][0]]
            if abs(b.cy - ref.cy) <= 0.5 * min(b.h, ref.h):
                rows[-1].append(i)
                continue
        rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda i: boxes[i].x)]


def group_components(
    components: list[Component],
    page_w: int,
    page_h: int,
    glyph_h: float,
    cfg: GroupConfig,
    timer: StageTimer | None = None,
) -> tuple[list[Line], list[Line], GroupStats]:
    """Group filtered components (processing coordinates) into lines of words.

    Returns ``(lines, rejected_lines, stats)``. Rejected lines failed a junk gate and
    carry its name in :attr:`Line.rejected`."""
    timer = timer or StageTimer()
    stats = GroupStats(glyph_height=glyph_h)
    if not cfg.enabled:
        return [], [], stats

    with timer.stage("group"):
        builders = [c for c in components
                    if c.kind is ComponentKind.TEXT and c.anchor_id is None]
        if not builders:
            return [], [], stats
        if glyph_h <= 0:
            glyph_h = float(np.median([c.bbox.h for c in builders]))
            stats.glyph_height = glyph_h

        # 1. Line mask from glyph cores, smeared horizontally by the line gap. Built at
        #    reduced resolution (MASK_PX_PER_GLYPH rows per glyph height): line
        #    structure needs no more, and every whole-page operation below shrinks by
        #    the square of the factor. Cores of adjacent lines stay >= 2 mask rows apart.
        s = min(1.0, MASK_PX_PER_GLYPH / glyph_h)
        mh, mw = max(1, int(np.ceil(page_h * s))), max(1, int(np.ceil(page_w * s)))
        box = np.array([c.bbox.as_list() for c in builders], dtype=np.float64)
        x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        core = np.minimum(cfg.core_fraction * h, cfg.core_fraction * cfg.core_cap_ratio * glyph_h)
        mid = y + h / 2.0
        y0 = np.floor((mid - core / 2.0) * s).astype(np.int64)
        y1 = np.maximum(y0 + 1, np.ceil((mid + core / 2.0) * s).astype(np.int64))
        x0 = np.floor(x * s).astype(np.int64)
        x1 = np.maximum(x0 + 1, np.ceil((x + w) * s).astype(np.int64))
        np.clip(y0, 0, mh - 1, out=y0)
        np.clip(x0, 0, mw - 1, out=x0)
        np.clip(y1, 0, mh, out=y1)
        np.clip(x1, 0, mw, out=x1)
        # Paint every rectangle at once: +1/-1 at the corners of a difference image,
        # then two cumulative sums. No Python loop over glyphs.
        diff = np.zeros((mh + 1, mw + 1), dtype=np.int32)
        np.add.at(diff, (y0, x0), 1)
        np.add.at(diff, (y0, x1), -1)
        np.add.at(diff, (y1, x0), -1)
        np.add.at(diff, (y1, x1), 1)
        mask = (diff.cumsum(0).cumsum(1)[:mh, :mw] > 0).astype(np.uint8) * 255
        k = max(1, int(round(cfg.line_gap_ratio * glyph_h * s)))
        smeared = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        _, labels = cv2.connectedComponents(smeared, connectivity=8, ltype=cv2.CV_32S)
        cy = np.minimum((y0 + y1 - 1) // 2, mh - 1)
        cx = np.minimum(np.floor((x + w / 2) * s).astype(np.int64), mw - 1)
        line_labels = labels[cy, cx]

        # 2-4. Lines, junk gates and words, as array operations over all glyphs at once.
        #      Python only loops over the *words* it emits. (A per-line loop with per-
        #      line NumPy calls and BBox unions measured P95 10-15 ms; its fixed
        #      per-call overhead, not arithmetic, was the cost.)
        candidates = _assemble(builders, components, x, y, w, h, line_labels, glyph_h, cfg, stats)
        lines = [ln for ln in candidates if not ln.rejected]
        rejected = [ln for ln in candidates if ln.rejected]

        # 5. Reading order.
        lines = [lines[i] for i in reading_order([ln.bbox for ln in lines])]
        stats.lines = len(lines)
        stats.words = sum(len(ln.words) for ln in lines)
    return lines, rejected, stats


def _group_median(values: np.ndarray, gidx: np.ndarray, starts: np.ndarray,
                  counts: np.ndarray) -> np.ndarray:
    """Median of ``values`` within each group, vectorised: sort by (group, value) and
    read the middle of each group's run. NaNs sort last and are excluded via
    ``counts`` (the number of valid values per group)."""
    order = np.lexsort((values, gidx))
    v = values[order]
    lo = starts + np.maximum(counts - 1, 0) // 2
    hi = starts + np.maximum(counts, 1) // 2 - (counts == 0)
    hi = np.maximum(hi, lo)
    return (v[lo] + v[hi]) / 2.0


def _assemble(builders, components, x, y, w, h, line_labels, glyph_h, cfg, stats) -> list[Line]:
    n = len(builders)
    x2, y2 = x + w, y + h

    # Sort by (line, x); every group of equal label is one candidate line.
    order = np.lexsort((x, line_labels))
    lab = line_labels[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(lab)) + 1))
    counts = np.diff(np.append(starts, n))
    gidx = np.repeat(np.arange(len(starts)), counts)  # group index per sorted glyph

    xs, ys, x2s, y2s = x[order], y[order], x2[order], y2[order]
    hs, ws = h[order], w[order]

    # 4. Junk gates, per group (see GroupConfig).
    max_h = np.maximum.reduceat(hs, starts)
    med_h = _group_median(hs, gidx, starts, counts)
    med_aspect = _group_median(ws / np.maximum(hs, 1.0), gidx, starts, counts)
    gates = (
        ("barcode", (counts >= 5) & (med_aspect < cfg.junk_barcode_aspect)),
        ("specks", (counts <= cfg.junk_small_max_count) & (max_h < cfg.junk_max_height_ratio * glyph_h)),
        ("single_small", (counts == 1) & (max_h < cfg.junk_single_height_ratio * glyph_h)),
        ("small_median", (counts >= 3) & (med_h < cfg.junk_median_height_ratio * glyph_h)),
    )
    reason = np.full(len(starts), "", dtype=object)
    for name, fired in gates:
        reason[fired & (reason == "")] = name
    junk = reason != ""
    stats.junk_lines = int(junk.sum())

    # 3. Word breaks. Reach = running max of right edges *within* each line: offset
    #    every group by a constant larger than any coordinate so the running max
    #    resets at each group start.
    big = float(x2s.max() + 1)
    reach = np.maximum.accumulate(x2s + gidx * big) - gidx * big
    gaps = np.full(n, np.nan)
    gaps[1:] = xs[1:] - reach[:-1]
    first = np.zeros(n, dtype=bool)
    first[starts] = True
    gaps[first] = np.nan
    threshold = np.full(len(starts), cfg.word_gap_ratio * glyph_h)
    if cfg.word_gap_median_factor > 0:
        # Wide letter spacing (monospace) raises a line's threshold: its own median
        # gap, for lines with at least three gaps.
        n_gaps = counts - 1
        med_gap = _group_median(np.where(np.isnan(gaps), np.inf, gaps), gidx, starts, n_gaps)
        wide = n_gaps >= 3
        threshold[wide] = np.maximum(threshold[wide], cfg.word_gap_median_factor * med_gap[wide])
    word_start = first | (gaps > threshold[gidx])

    word_id = np.cumsum(word_start) - 1
    w_starts = np.flatnonzero(word_start)
    wx0 = np.minimum.reduceat(xs, w_starts)
    wy0 = np.minimum.reduceat(ys, w_starts)
    wx1 = np.maximum.reduceat(x2s, w_starts)
    wy1 = np.maximum.reduceat(y2s, w_starts)
    word_line = gidx[w_starts]

    # Members: split the sorted glyph order at word starts (junk words are skipped
    # when emitting, so no per-glyph test is needed here).
    members: list[list[Component]] = [
        [builders[i] for i in chunk] for chunk in np.split(order, w_starts[1:])
    ]
    # Small marks join their anchor's word, growing its box. Only the anchors that
    # marks actually reference are looked up.
    marks = [c for c in components if c.anchor_id is not None]
    if marks:
        ids = np.fromiter((c.id for c in builders), dtype=np.int64, count=n)
        id_order = np.argsort(ids)
        anchors = np.fromiter((c.anchor_id for c in marks), dtype=np.int64, count=len(marks))
        pos = np.searchsorted(ids[id_order], anchors)
        pos = np.minimum(pos, n - 1)
        found = ids[id_order][pos] == anchors
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)  # builder index -> position in sorted order
        mark_wid = word_id[rank[id_order[pos]]]
        for c, ok, wid in zip(marks, found.tolist(), mark_wid.tolist()):
            if ok:
                members[wid].append(c)
                b = c.bbox
                wx0[wid], wy0[wid] = min(wx0[wid], b.x), min(wy0[wid], b.y)
                wx1[wid], wy1[wid] = max(wx1[wid], b.x2), max(wy1[wid], b.y2)

    # Emit every line; junk lines carry the gate that rejected them.
    words_by_line: dict[int, list[Word]] = {}
    for wid in range(len(w_starts)):
        g = int(word_line[wid])
        comps = members[wid]
        comps.sort(key=lambda c: c.bbox.x)
        bbox = BBox(int(wx0[wid]), int(wy0[wid]), int(wx1[wid] - wx0[wid]), int(wy1[wid] - wy0[wid]))
        words_by_line.setdefault(g, []).append(Word(bbox=bbox, components=comps))
    return [Line(bbox=_union([wd.bbox for wd in ws_]), words=ws_, rejected=str(reason[g]))
            for g, ws_ in words_by_line.items()]


def rescale_lines(lines: list[Line], factor: float) -> None:
    """Map line and word boxes to another frame. Components are rescaled by the
    engine already; words hold references to the same objects."""
    for ln in lines:
        ln.bbox = ln.bbox.scaled(factor)
        for w in ln.words:
            w.bbox = w.bbox.scaled(factor)
