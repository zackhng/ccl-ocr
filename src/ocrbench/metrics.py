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

from dataclasses import dataclass, field
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
            "hit_rate": round(self.hit_rate, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "crossing_rate": round(self.crossing_rate, 4),
            "junk_rate": None if self.junk_rate is None else round(self.junk_rate, 4),
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
    words: WordMetrics = field(default_factory=WordMetrics)
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
            "words": self.words.as_dict(),
            "timings_ms": self.timings_ms,
            "kind_counts": self.kind_counts,
            "meta": self.meta,
        }


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


def evaluate(sample: Sample, result: PageResult) -> SampleMetrics:
    return SampleMetrics(
        sample_id=sample.sample_id,
        source=sample.source,
        capture=sample.capture,
        script=sample.script,
        dpi=sample.dpi,
        template=str(sample.meta.get("template", "")),
        chars=char_metrics(sample, result) if sample.has_char_gt else None,
        words=word_metrics(sample, result),
        timings_ms=result.timings_ms,
        kind_counts=result.kind_counts(),
        meta={
            "polarity_inverted": result.meta.get("polarity_inverted"),
            "recovered_inverted_regions": result.meta.get("recovered_inverted_regions"),
            "median_component_height": result.meta.get("filter", {}).get("median_height"),
            "profile": sample.meta.get("profile"),
        },
    )
