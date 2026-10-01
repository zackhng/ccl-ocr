"""Isolation metrics: did CCL turn characters into usable component candidates?

These are deliberately *not* OCR accuracy metrics — nothing is recognised yet. They
measure the ceiling that Phase 3 inherits. A character that CCL split into three
fragments, or merged into its neighbour, or filtered away as noise, cannot be recovered
by any classifier downstream, however good.

Four outcomes per ground-truth character, in priority order:

``isolated``
    One surviving component matches it at IoU >= threshold, and that component matches
    nothing else. This is the only outcome Phase 3 can work with as designed.
``merged``
    One component swallows this character and at least one neighbour. Common on bold
    text, low DPI, and glyphs printed on top of a rule.
``over_segmented``
    Two or more components each sit mostly inside this character and together cover it.
    Broken strokes — thin fonts, aggressive thresholds, JPEG artefacts.
``missed``
    Nothing survived here at all: filtered away, or never labelled.

Plus, over components rather than characters, ``junk`` — surviving components that sit
on no ground-truth character. Junk costs latency (every one is a wasted inference) but
not accuracy, so it is reported separately rather than folded into a single score.

Where a dataset has no character annotations — which is most real ones — the same
questions are asked at word level instead. The fallback is not optional: a metric that
quietly skipped those samples would compute its headline number over synthetic data
only.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field, fields
from statistics import median

import numpy as np

from ocr.types import BBox, ComponentKind, PageResult

from .gt import Sample

SURVIVING_KINDS = frozenset({ComponentKind.TEXT, ComponentKind.DIACRITIC})
"""What still reaches the classifier. DIACRITIC counts as retained because Phase 5
merges it back into its parent glyph; it is routed, not discarded."""

IOU_MATCH = 0.5
COVER_MERGE = 0.5
"""Fraction of a character a component must cover to be said to contain it."""

FRAGMENT_INSIDE = 0.5
"""Fraction of a component that must lie inside a character to count as its fragment."""

JUNK_MAX_OVERLAP = 0.30
"""A component sitting less than this much on any character is junk."""

SMALL_CHAR_RATIO = 0.55
"""Height ratio below the page median that makes a ground-truth character a 'small
mark' — the punctuation and diacritics whose retention we track separately, because
they are the ones a naive size filter throws away."""


class BoxIndex:
    """Uniform grid over boxes. Keeps matching linear rather than quadratic.

    A dense page carries a few thousand components and a similar number of ground-truth
    characters; all-pairs comparison in Python would dominate the benchmark's own
    runtime.
    """

    __slots__ = ("boxes", "cell", "grid")

    def __init__(self, boxes: list[BBox], cell: int | None = None) -> None:
        self.boxes = boxes
        if cell is None:
            cell = int(median([max(b.w, b.h) for b in boxes])) if boxes else 16
        self.cell = max(8, cell)
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, b in enumerate(boxes):
            for gx in range(b.x // self.cell, b.x2 // self.cell + 1):
                for gy in range(b.y // self.cell, b.y2 // self.cell + 1):
                    self.grid.setdefault((gx, gy), []).append(i)

    def query(self, b: BBox) -> set[int]:
        out: set[int] = set()
        for gx in range(b.x // self.cell, b.x2 // self.cell + 1):
            for gy in range(b.y // self.cell, b.y2 // self.cell + 1):
                hit = self.grid.get((gx, gy))
                if hit:
                    out.update(hit)
        return out


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _counts_from_dict(cls, d: dict | None):
    """Rebuild a metrics dataclass from its ``as_dict`` output.

    ``as_dict`` writes the raw counts *and* the derived rates; only the counts are
    fields, so the rates are dropped here and recomputed from the counts. That keeps
    a reloaded run's aggregates identical to the original's (rates aggregate from
    counts, never from stored per-sample rates)."""
    if d is None:
        return None
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(slots=True)
class CharMetrics:
    n_gt: int = 0
    n_surviving: int = 0
    isolated: int = 0
    merged: int = 0
    over_segmented: int = 0
    missed: int = 0
    junk: int = 0
    junk_in_nontext: int = 0
    gt_complete: bool = True
    small_gt: int = 0
    small_retained: int = 0

    @property
    def isolation_recall(self) -> float:
        return _safe_div(self.isolated, self.n_gt)

    @property
    def merge_rate(self) -> float:
        return _safe_div(self.merged, self.n_gt)

    @property
    def over_segmentation_rate(self) -> float:
        return _safe_div(self.over_segmented, self.n_gt)

    @property
    def miss_rate(self) -> float:
        return _safe_div(self.missed, self.n_gt)

    @property
    def junk_rate(self) -> float | None:
        """Undefined when ground truth is incomplete — a component sitting on real but
        unannotated text is not junk, and calling it junk would punish the pipeline for
        the dataset's gaps."""
        if not self.gt_complete:
            return None
        return _safe_div(self.junk, self.n_surviving)

    @property
    def small_retention(self) -> float | None:
        """None when the slice has no small marks at all — Han text, for instance.
        Reporting 0% there would read as a total failure instead of 'not applicable'."""
        if self.small_gt == 0:
            return None
        return self.small_retained / self.small_gt

    def as_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "n_surviving": self.n_surviving,
            "isolated": self.isolated,
            "merged": self.merged,
            "over_segmented": self.over_segmented,
            "missed": self.missed,
            "junk": self.junk,
            "junk_in_nontext": self.junk_in_nontext,
            "small_gt": self.small_gt,
            "small_retained": self.small_retained,
            "gt_complete": self.gt_complete,
            "isolation_recall": round(self.isolation_recall, 4),
            "merge_rate": round(self.merge_rate, 4),
            "over_segmentation_rate": round(self.over_segmentation_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "junk_rate": None if self.junk_rate is None else round(self.junk_rate, 4),
            "small_retention": None if self.small_retention is None else round(self.small_retention, 4),
        }


@dataclass(slots=True)
class WordMetrics:
    """Region-level outcomes.

    The region is a *word* where the dataset annotates words (FUNSD, our synthetic set)
    and a *text line* where it only annotates lines (SROIE). :attr:`granularity` records
    which, because the two are not comparable: line-level coverage is mechanically
    higher than word-level, so mixing them in one average would be meaningless.
    """

    n_gt: int = 0
    n_surviving: int = 0
    hit: int = 0
    crossing: int = 0
    junk: int = 0
    gt_complete: bool = True
    coverage_sum: float = 0.0
    granularity: str = "word"

    @property
    def hit_rate(self) -> float:
        return _safe_div(self.hit, self.n_gt)

    @property
    def mean_coverage(self) -> float:
        return _safe_div(self.coverage_sum, self.n_gt)

    @property
    def crossing_rate(self) -> float:
        return _safe_div(self.crossing, self.n_gt)

    @property
    def junk_rate(self) -> float | None:
        if not self.gt_complete:
            return None
        return _safe_div(self.junk, self.n_surviving)

    def as_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "n_surviving": self.n_surviving,
            "hit": self.hit,
            "crossing": self.crossing,
            "junk": self.junk,
            "gt_complete": self.gt_complete,
            "granularity": self.granularity,
            "coverage_sum": round(self.coverage_sum, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "crossing_rate": round(self.crossing_rate, 4),
            "junk_rate": None if self.junk_rate is None else round(self.junk_rate, 4),
        }


@dataclass(slots=True)
class RegionMetrics:
    """Engine-agnostic text localisation, scored per ground-truth *line*.

    Exists because the component metrics above cannot compare engines: they match
    predicted boxes against characters or words, so an engine that returns one box per
    line (PaddleOCR) would score every word it covers as "crossing" and every line box
    as junk. Here each GT line asks only "how much of my horizontal extent did the
    engine put a text box on?", which reads the same whether the engine answered with
    one line box or forty character boxes. See :func:`region_metrics`.
    """

    n_gt: int = 0
    n_pred: int = 0
    found: int = 0
    coverage_sum: float = 0.0
    spurious: int = 0
    gt_complete: bool = True

    @property
    def line_recall(self) -> float:
        return _safe_div(self.found, self.n_gt)

    @property
    def mean_coverage(self) -> float:
        return _safe_div(self.coverage_sum, self.n_gt)

    @property
    def spurious_rate(self) -> float | None:
        """Share of predicted boxes on no GT line. Undefined where GT is incomplete,
        for the same reason as :attr:`CharMetrics.junk_rate`."""
        if not self.gt_complete:
            return None
        return _safe_div(self.spurious, self.n_pred)

    def as_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "found": self.found,
            "coverage_sum": round(self.coverage_sum, 4),
            "spurious": self.spurious,
            "gt_complete": self.gt_complete,
            "line_recall": round(self.line_recall, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "spurious_rate": None if self.spurious_rate is None else round(self.spurious_rate, 4),
        }


@dataclass(slots=True)
class TextMetrics:
    """Recognition accuracy, for engines that return text.

    CER counts edits against ground-truth characters, plus every character of a
    predicted line that landed on no GT line (an insertion) — but only where GT is
    complete, since text on an unannotated line is not an error. Word F1 is a
    bag-of-words score, independent of reading order and line matching, as a cross-
    check on the line assignment.
    """

    n_gt_chars: int = 0
    edits: int = 0
    inserted_chars: int = 0
    n_gt_words: int = 0
    n_pred_words: int = 0
    matched_words: int = 0
    gt_complete: bool = True

    @property
    def cer(self) -> float:
        return _safe_div(self.edits + self.inserted_chars, self.n_gt_chars)

    @property
    def word_precision(self) -> float:
        return _safe_div(self.matched_words, self.n_pred_words)

    @property
    def word_recall(self) -> float:
        return _safe_div(self.matched_words, self.n_gt_words)

    @property
    def word_f1(self) -> float:
        p, r = self.word_precision, self.word_recall
        return _safe_div(2 * p * r, p + r)

    def as_dict(self) -> dict:
        return {
            "n_gt_chars": self.n_gt_chars,
            "edits": self.edits,
            "inserted_chars": self.inserted_chars,
            "n_gt_words": self.n_gt_words,
            "n_pred_words": self.n_pred_words,
            "matched_words": self.matched_words,
            "gt_complete": self.gt_complete,
            "cer": round(self.cer, 4),
            "word_f1": round(self.word_f1, 4),
        }


@dataclass(slots=True)
class SampleMetrics:
    sample_id: str
    source: str
    capture: str
    script: str
    dpi: int
    template: str = ""
    chars: CharMetrics | None = None
    words: WordMetrics | None = None
    """``None`` for engines that produce no components (PaddleOCR): word metrics score
    components, and zeros would read as a failure rather than 'not applicable'."""
    regions: RegionMetrics | None = None
    text: TextMetrics | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    kind_counts: dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "capture": self.capture,
            "script": self.script,
            "dpi": self.dpi,
            "template": self.template,
            "chars": None if self.chars is None else self.chars.as_dict(),
            "words": None if self.words is None else self.words.as_dict(),
            "regions": None if self.regions is None else self.regions.as_dict(),
            "text": None if self.text is None else self.text.as_dict(),
            "timings_ms": self.timings_ms,
            "kind_counts": self.kind_counts,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SampleMetrics:
        words = _counts_from_dict(WordMetrics, d.get("words"))
        if words is not None and "coverage_sum" not in (d.get("words") or {}):
            # Results written before coverage_sum was persisted.
            words.coverage_sum = d["words"].get("mean_coverage", 0.0) * words.n_gt
        return cls(
            sample_id=d["sample_id"],
            source=d["source"],
            capture=d["capture"],
            script=d["script"],
            dpi=int(d.get("dpi", 0)),
            template=d.get("template", ""),
            chars=_counts_from_dict(CharMetrics, d.get("chars")),
            words=words,
            regions=_counts_from_dict(RegionMetrics, d.get("regions")),
            text=_counts_from_dict(TextMetrics, d.get("text")),
            timings_ms=d.get("timings_ms", {}),
            kind_counts=d.get("kind_counts", {}),
            meta=d.get("meta", {}),
        )


def _surviving_boxes(result: PageResult) -> list[BBox]:
    return [c.bbox for c in result.components if c.kind in SURVIVING_KINDS]


def char_metrics(sample: Sample, result: PageResult) -> CharMetrics:
    """Per-character outcomes. Requires ``sample.chars``."""
    gt_boxes = [c.bbox for c in (sample.chars or [])]
    comp_boxes = _surviving_boxes(result)
    gt_complete = bool(sample.meta.get("gt_complete", sample.source == "synth"))

    m = CharMetrics(n_gt=len(gt_boxes), n_surviving=len(comp_boxes), gt_complete=gt_complete)
    if not gt_boxes:
        return m

    heights = [b.h for b in gt_boxes]
    median_h = median(heights)
    small_idx = {i for i, b in enumerate(gt_boxes) if b.h < SMALL_CHAR_RATIO * median_h}
    m.small_gt = len(small_idx)

    index = BoxIndex(gt_boxes)

    gt_matches: list[list[int]] = [[] for _ in gt_boxes]
    comp_matches: list[list[int]] = [[] for _ in comp_boxes]
    comp_covers: list[list[int]] = [[] for _ in comp_boxes]
    gt_fragments: list[list[int]] = [[] for _ in gt_boxes]
    gt_frag_area: list[int] = [0] * len(gt_boxes)
    gt_touched: list[bool] = [False] * len(gt_boxes)

    nontext = BoxIndex([r.bbox for r in sample.regions_nontext]) if sample.regions_nontext else None

    for j, cb in enumerate(comp_boxes):
        best_overlap = 0.0
        for i in index.query(cb):
            gb = gt_boxes[i]
            inter = cb.intersection_area(gb)
            if inter == 0:
                continue
            gt_touched[i] = True
            best_overlap = max(best_overlap, inter / cb.area)
            union = cb.area + gb.area - inter
            if union and inter / union >= IOU_MATCH:
                gt_matches[i].append(j)
                comp_matches[j].append(i)
            if inter / gb.area >= COVER_MERGE:
                comp_covers[j].append(i)
            if inter / cb.area >= FRAGMENT_INSIDE:
                gt_fragments[i].append(j)
                gt_frag_area[i] += inter

        if best_overlap < JUNK_MAX_OVERLAP:
            m.junk += 1
            if nontext is not None and any(
                sample.regions_nontext[k].bbox.intersection_area(cb) / cb.area >= 0.5
                for k in nontext.query(cb)
            ):
                m.junk_in_nontext += 1

    merged_gt: set[int] = set()
    for j, covered in enumerate(comp_covers):
        if len(covered) >= 2:
            merged_gt.update(covered)

    for i, gb in enumerate(gt_boxes):
        matches = gt_matches[i]
        if len(matches) == 1 and len(comp_matches[matches[0]]) == 1:
            m.isolated += 1
            if i in small_idx:
                m.small_retained += 1
            continue
        if i in merged_gt:
            m.merged += 1
            # A merged character still *reached* the classifier, so for the purpose of
            # "did the size filter throw this mark away" it counts as retained.
            if i in small_idx:
                m.small_retained += 1
            continue
        if len(gt_fragments[i]) >= 2 and gt_frag_area[i] >= FRAGMENT_INSIDE * gb.area:
            m.over_segmented += 1
            if i in small_idx:
                m.small_retained += 1
            continue
        if matches:
            # Matched, but ambiguously (several candidates, or the component also
            # matches another character). Not isolated, not a clean merge or split.
            m.merged += 1
            if i in small_idx:
                m.small_retained += 1
            continue
        m.missed += 1
        if i in small_idx and gt_touched[i]:
            # Something was there but too partial to count; still not "thrown away".
            m.small_retained += 1

    return m


def word_metrics(sample: Sample, result: PageResult) -> WordMetrics:
    """Word-level fallback, computed for every sample — not only those lacking chars.

    Having both lets a real-dataset number be sanity-checked against a synthetic one
    where the character truth is known.
    """
    # Fall back to lines where a dataset annotates only lines. Skipping those samples
    # instead would silently drop entire real datasets out of the headline.
    regions = sample.words or sample.lines
    granularity = "word" if sample.words else "line"

    gt_boxes = [w.bbox for w in regions]
    comp_boxes = _surviving_boxes(result)
    gt_complete = bool(sample.meta.get("gt_complete", sample.source == "synth"))

    m = WordMetrics(
        n_gt=len(gt_boxes), n_surviving=len(comp_boxes),
        gt_complete=gt_complete, granularity=granularity,
    )
    if not gt_boxes:
        return m

    index = BoxIndex(gt_boxes)
    comp_assigned = [False] * len(comp_boxes)
    per_word: list[list[int]] = [[] for _ in gt_boxes]
    crossing = [False] * len(gt_boxes)

    for j, cb in enumerate(comp_boxes):
        inside_count = 0
        for i in index.query(cb):
            gb = gt_boxes[i]
            inter = cb.intersection_area(gb)
            if inter == 0:
                continue
            if inter / cb.area >= 0.5:
                per_word[i].append(j)
                comp_assigned[j] = True
            elif inter / cb.area >= 0.15:
                # Straddles a word boundary: one component spanning two words is the
                # word-level signature of a character merge.
                inside_count += 1
                crossing[i] = True
                comp_assigned[j] = True
        if inside_count >= 2:
            pass  # already flagged on each affected word

    for i, gb in enumerate(gt_boxes):
        comps = per_word[i]
        if comps:
            m.hit += 1
            # Exact union area via a small mask; word boxes are tiny so this is cheap
            # and avoids the double-counting that summing box areas would introduce.
            mask = np.zeros((gb.h, gb.w), dtype=bool)
            for j in comps:
                cb = comp_boxes[j]
                x0, y0 = max(0, cb.x - gb.x), max(0, cb.y - gb.y)
                x1, y1 = min(gb.w, cb.x2 - gb.x), min(gb.h, cb.y2 - gb.y)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = True
            m.coverage_sum += float(mask.mean())
        if crossing[i]:
            m.crossing += 1

    m.junk = sum(1 for assigned in comp_assigned if not assigned)
    return m


# ---------------------------------------------------------------- engine-agnostic

LINE_FOUND = 0.5
"""Span coverage at which a GT line counts as found."""

SPAN_MIN_V_OVERLAP = 0.5
"""A predicted box contributes to a GT line only if it overlaps at least this fraction
of the shorter of the two heights, *and* its vertical centre lies within the line. The
centre test is what stops a box on the line above from localising this one: PaddleOCR
pads its boxes (unclip ratio 1.5), so a 12 px line gets a 26-29 px box whose edge can
reach well into the neighbouring line."""

SPAN_MAX_HEIGHT_RATIO = 4.0
"""...and is at most this many times the line's height. A blob swallowing a whole
paragraph has not localised any line within it. 4x matches the CCL filter's own
oversize gate, and clears Paddle's padding (up to ~2.7x on small text)."""

SPAN_GAP_CLOSE = 1.0
"""Gaps narrower than this many line-heights between contributing boxes are closed
before measuring coverage. Inter-character and inter-word gaps are well under one
line-height; without closing them, per-character boxes could never reach the coverage
a single line box gets for free, and the metric would reward granularity, not
localisation."""

SPURIOUS_MAX_OVERLAP = 0.15
"""A predicted box with less than this fraction of its area on GT lines is spurious."""


def text_boxes(result: PageResult) -> list[BBox]:
    """What an engine claims is text: its line boxes if it produced lines, otherwise
    its surviving components."""
    if result.lines:
        return [ln.bbox for ln in result.lines]
    return _surviving_boxes(result)


def _span_coverage(gt: BBox, boxes: list[BBox]) -> float:
    spans: list[tuple[int, int]] = []
    for b in boxes:
        if b.h > SPAN_MAX_HEIGHT_RATIO * gt.h:
            continue
        if gt.vertical_overlap(b) < SPAN_MIN_V_OVERLAP or not gt.y <= b.cy <= gt.y2:
            continue
        x0, x1 = max(gt.x, b.x), min(gt.x2, b.x2)
        if x1 > x0:
            spans.append((x0, x1))
    if not spans:
        return 0.0
    spans.sort()
    gap = SPAN_GAP_CLOSE * gt.h
    covered = 0
    cur0, cur1 = spans[0]
    for x0, x1 in spans[1:]:
        if x0 - cur1 <= gap:
            cur1 = max(cur1, x1)
        else:
            covered += cur1 - cur0
            cur0, cur1 = x0, x1
    covered += cur1 - cur0
    return min(1.0, covered / gt.w) if gt.w else 0.0


def region_metrics(sample: Sample, result: PageResult) -> RegionMetrics:
    """Line-level localisation, comparable across engines of any output granularity.

    Coverage is measured along the line's *horizontal span* rather than its area: a
    tight character box and a padded line box over the same glyphs then score alike.
    """
    gt_boxes = [ln.bbox for ln in sample.lines]
    pred = text_boxes(result)
    gt_complete = bool(sample.meta.get("gt_complete", sample.source == "synth"))
    m = RegionMetrics(n_gt=len(gt_boxes), n_pred=len(pred), gt_complete=gt_complete)
    if not gt_boxes:
        return m

    gt_index = BoxIndex(gt_boxes)
    pred_index = BoxIndex(pred) if pred else None

    for gb in gt_boxes:
        near = [pred[j] for j in pred_index.query(gb)] if pred_index else []
        cov = _span_coverage(gb, near)
        m.coverage_sum += cov
        if cov >= LINE_FOUND:
            m.found += 1

    for pb in pred:
        on_text = sum(pb.intersection_area(gt_boxes[i]) for i in gt_index.query(pb))
        if min(1.0, _safe_div(on_text, pb.area)) < SPURIOUS_MAX_OVERLAP:
            m.spurious += 1
    return m


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two-row DP. Lines are short, so pure Python is fast enough."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(text: str) -> str:
    """NFKC, then collapse whitespace runs to one space.

    NFKC because PP-OCRv5's multilingual dictionary emits full-width punctuation
    ("NAME：" with U+FF1A) for what is printed as an ASCII colon. That is the same
    character to any reader and to any downstream parser that normalises, so charging
    it as a misread would measure the dictionary, not the recogniser."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


CASE_FOLDED_SOURCES = frozenset({"sroie"})
"""Datasets whose transcripts are case-normalised. SROIE annotates every receipt in
upper case whatever the print says, so a recogniser that correctly reads "Email" would
be charged four errors against "EMAIL". Case is folded on both sides for these sources
only; synthetic GT is exact and keeps case, so l/L and o/O confusions still count."""


def _cer_key(text: str, fold: bool) -> str:
    """What CER compares: no whitespace at all, case-folded where GT demands it.

    Whitespace is excluded because CER should measure whether the *characters* were
    read. Dropped spaces are a real defect — PP-OCRv5 on Latin produces "ROCNO:538358-H"
    — and are charged where they belong, in word F1, instead of being folded into a
    character-error figure where they cannot be told apart from misreads."""
    s = "".join(_norm(text).split())
    return s.casefold() if fold else s


UNSPACED_SCRIPTS = frozenset({"han"})
"""Scripts written without spaces between words. Bag-of-words uses characters there."""


def _tokens(text: str, script: str) -> list[str]:
    if script in UNSPACED_SCRIPTS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


TEXT_LINK_MIN = 0.5
"""Overlap, as a fraction of the *smaller* of the two boxes, at which a predicted line
and a GT line are linked for scoring.

Against the smaller box because Paddle pads its boxes to more than twice the height of
the tight GT box: less than half the prediction lies on the line even when it reads it
perfectly. A rule keyed to the prediction's area charged those lines once as a deletion
and again as an insertion (CER 185% on a form whose word F1 was 92%)."""


def _reading_order(items: list[tuple[BBox, str]]) -> list[str]:
    """Texts in reading order: rows top to bottom, left to right within a row.

    A plain (y, x) sort is not enough: two halves of one split line can differ in y
    by a pixel and would come out right-half-first. Items join the current row while
    their vertical centre is within half a height of the row's first item."""
    rows: list[list[tuple[BBox, str]]] = []
    for b, t in sorted(items, key=lambda it: it[0].cy):
        if rows:
            ref = rows[-1][0][0]
            if abs(b.cy - ref.cy) <= 0.5 * min(b.h, ref.h):
                rows[-1].append((b, t))
                continue
        rows.append([(b, t)])
    return [t for row in rows for _, t in sorted(row, key=lambda it: it[0].x)]


def text_metrics(sample: Sample, result: PageResult) -> TextMetrics | None:
    """CER and word F1. ``None`` when the engine returned no text or GT has no line text.

    GT lines and predicted lines are linked wherever they overlap substantially
    (:data:`TEXT_LINK_MIN`), and each *connected cluster* is scored as one unit: its GT
    texts joined in reading order against its predicted texts joined in reading order.

    Clusters rather than one-to-one matching, because the two sides disagree on what a
    "line" is far more often than they disagree on the text. A detector that splits a
    line in two, or merges a FUNSD label with its value ("TO:" + "George Baroody" are
    two GT entities on one printed line), or reads a multi-line FUNSD entity as three
    lines, is reading the text correctly — any matching that must pick one partner
    charges it a deletion plus an insertion for each. Clusters only charge what is
    actually wrong: misread characters, GT lines nothing landed on (deletions), and
    predictions on no GT line (insertions, where GT is complete).
    """
    preds = [(ln.bbox, _norm(ln.text)) for ln in result.lines]
    preds = [(b, t) for b, t in preds if t]
    if not preds:
        return None
    gts = [(ln.bbox, ln.text) for ln in sample.lines if ln.text]
    if not gts:
        return None
    gt_complete = bool(sample.meta.get("gt_complete", sample.source == "synth"))
    fold = sample.source in CASE_FOLDED_SOURCES
    m = TextMetrics(gt_complete=gt_complete)

    def words(t: str) -> list[str]:
        t = _norm(t)
        return _tokens(t.casefold() if fold else t, sample.script)

    # Union-find over GT nodes 0..G-1 and prediction nodes G..G+P-1.
    n_gt = len(gts)
    parent = list(range(n_gt + len(preds)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    index = BoxIndex([b for b, _ in gts])
    for j, (pb, _) in enumerate(preds):
        for i in index.query(pb):
            gb = gts[i][0]
            if _safe_div(pb.intersection_area(gb), min(pb.area, gb.area)) >= TEXT_LINK_MIN:
                parent[find(n_gt + j)] = find(i)

    clusters: dict[int, tuple[list[tuple[BBox, str]], list[tuple[BBox, str]]]] = {}
    for i, g in enumerate(gts):
        clusters.setdefault(find(i), ([], []))[0].append(g)
    for j, p in enumerate(preds):
        clusters.setdefault(find(n_gt + j), ([], []))[1].append(p)

    linked_pred_words: list[str] = []
    for g_items, p_items in clusters.values():
        hyp = "".join(_cer_key(t, fold) for t in _reading_order(p_items))
        if not g_items:
            if gt_complete:
                m.inserted_chars += len(hyp)
            continue
        ref = "".join(_cer_key(t, fold) for t in _reading_order(g_items))
        m.n_gt_chars += len(ref)
        m.edits += levenshtein(ref, hyp)
        linked_pred_words += [tok for _, t in p_items for tok in words(t)]

    gt_words = [tok for _, t in gts for tok in words(t)]
    # Where GT is incomplete, words on unannotated lines are not false positives;
    # count only predictions that landed on a GT line.
    pred_words = [tok for _, t in preds for tok in words(t)] if gt_complete else linked_pred_words
    m.n_gt_words, m.n_pred_words = len(gt_words), len(pred_words)
    m.matched_words = sum((Counter(gt_words) & Counter(pred_words)).values())
    return m


def evaluate(sample: Sample, result: PageResult) -> SampleMetrics:
    # A line-level engine (PaddleOCR) has lines and no components. Keyed on that rather
    # than on emptiness, so a blank page through CCL still reports its word metrics.
    has_components = bool(result.components) or not result.lines
    return SampleMetrics(
        sample_id=sample.sample_id,
        source=sample.source,
        capture=sample.capture,
        script=sample.script,
        dpi=sample.dpi,
        template=str(sample.meta.get("template", "")),
        chars=char_metrics(sample, result) if sample.has_char_gt and has_components else None,
        words=word_metrics(sample, result) if has_components else None,
        regions=region_metrics(sample, result),
        text=text_metrics(sample, result),
        timings_ms=result.timings_ms,
        kind_counts=result.kind_counts(),
        meta={
            "polarity_inverted": result.meta.get("polarity_inverted"),
            "recovered_inverted_regions": result.meta.get("recovered_inverted_regions"),
            "median_component_height": result.meta.get("filter", {}).get("median_height"),
            "profile": sample.meta.get("profile"),
        },
    )
