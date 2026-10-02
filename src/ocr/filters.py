"""Component filtering — the cheap geometric pass that protects the classifier.

Two tiers, cheapest first:

* **Tier 1** uses only a component's own geometry, so it needs no page statistics and
  can reject obvious junk before we have looked at the page as a whole.
* **Tier 2** keys off the *ink-weighted median height of Tier-1 survivors* — an estimate
  of "how tall is a character on this page" that survives a page covered in speckle,
  which a plain median does not. See :func:`ink_weighted_median_height`.

The governing principle is that **filtering routes rather than deletes**. Only
:attr:`ComponentKind.NOISE` is thrown away. Everything else is labelled and kept:
diacritics because dropping them silently caps accuracy in a way no later stage can
undo, rules because Phase 5 needs them for layout, blobs because Phase 11 needs them
for photo masking. Each decision records which gate fired, so the overlay can show
*why* a component was routed where it was and thresholds can be tuned from evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FilterConfig
from .timing import StageTimer
from .types import Component, ComponentKind


@dataclass(slots=True)
class FilterStats:
    """Diagnostics about one filtering pass — surfaced in the benchmark report."""

    total: int = 0
    median_height: float = 0.0
    relative_gates_applied: bool = False
    counts: dict[str, int] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "median_height": round(self.median_height, 2),
            "relative_gates_applied": self.relative_gates_applied,
            "counts": self.counts or {},
        }


def _route(component: Component, kind: ComponentKind, reason: str) -> None:
    component.kind = kind
    component.reason = reason


def tier1_text_mask(stats: np.ndarray, page_w: int, page_h: int, cfg: FilterConfig) -> np.ndarray:
    """Which rows of a CCL ``stats`` array :func:`apply_tier1` would leave as TEXT.

    A vectorised mirror of :func:`apply_tier1`, for the merge splitter, which must
    measure the page's glyph scale before filtering and cannot afford a Python loop over
    tens of thousands of components on a noisy photo. It must make exactly the same
    decisions; ``tests/test_split.py`` checks that on randomised components. Row 0 (the
    background) is included and comes out False.
    """
    w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    box = np.maximum(w * h, 1.0)
    fill = np.where(w * h > 0, area / box, 0.0)
    aspect = np.where(h > 0, w / np.maximum(h, 1.0), 0.0)

    degenerate = (area < cfg.min_pixel_area) | ((h < cfg.min_extent) & (w < cfg.min_extent))
    oversized = (w > cfg.max_width_frac * page_w) | (h > cfg.max_height_frac * page_h)
    vertical_rule = aspect <= cfg.min_aspect
    wide = aspect >= cfg.max_aspect
    # A wide *sparse* component stays TEXT (it is likely merged glyphs); a wide solid
    # one is a rule. Below max_aspect, a hollow component is a rule.
    rejected_wide = wide & (fill >= cfg.rule_fill_min)
    hollow = ~wide & (fill < cfg.min_fill_ratio)

    text = ~(degenerate | oversized | vertical_rule | rejected_wide | hollow)
    text[0] = False
    return text


def apply_tier1(components: list[Component], page_w: int, page_h: int, cfg: FilterConfig) -> None:
    """Absolute gates. Mutates ``kind``/``reason`` in place."""
    max_w = cfg.max_width_frac * page_w
    max_h = cfg.max_height_frac * page_h

    for c in components:
        b = c.bbox

        if c.pixel_area < cfg.min_pixel_area or (b.h < cfg.min_extent and b.w < cfg.min_extent):
            _route(c, ComponentKind.NOISE, "t1:degenerate")
            continue

        oversized = b.w > max_w or b.h > max_h
        elongated = c.aspect_ratio >= cfg.rule_aspect or c.aspect_ratio <= cfg.min_aspect

        if oversized:
            if c.fill_ratio < cfg.min_fill_ratio:
                # Hollow and large: the frame drawn around a form, a table outline, the
                # border of an ID card. Structure, not a picture.
                _route(c, ComponentKind.RULE, "t1:frame")
            elif elongated and c.fill_ratio >= cfg.rule_fill_min:
                _route(c, ComponentKind.RULE, "t1:oversized_rule")
            else:
                # Dense and large: portrait photo, logo, barcode, signature.
                _route(c, ComponentKind.BLOB, "t1:oversized")
            continue

        if c.aspect_ratio <= cfg.min_aspect:
            _route(c, ComponentKind.RULE, "t1:vertical_rule")
            continue

        if c.aspect_ratio >= cfg.max_aspect:
            if c.fill_ratio >= cfg.rule_fill_min:
                _route(c, ComponentKind.RULE, "t1:horizontal_rule")
            # A wide *sparse* component is far more likely to be several touching
            # glyphs than a rule. Keep it as TEXT so the benchmark counts it as the
            # merge it is, instead of hiding the failure as a rejected rule.
            continue

        if c.fill_ratio < cfg.min_fill_ratio:
            # Hollow: a table cell border or a box outline drawn around a field.
            _route(c, ComponentKind.RULE, "t1:hollow")
            continue


def ink_weighted_median_height(components: list[Component]) -> float:
    """The height at which half the page's *ink* lies in shorter components.

    A plain median of component heights is not robust here, and the failure is severe
    rather than marginal. A noisy photograph produces hundreds of 2-4 px specks that
    pass Tier 1 (they must — a decimal point is 4 px), and they outnumber the glyphs.
    The median then collapses onto the speck population: measured at 4 px on ID cards
    whose real glyphs are 17 px. Every relative gate is keyed to that number, so the
    "4x median" ceiling lands at 16 px and routes the largest and most important text on
    the card — the ID number, the name — to BLOB. Isolation recall on ID cards was 25%
    for this reason alone.

    Weighting by ink area fixes it because that is exactly what distinguishes the two
    populations: a speck carries ~4 px of ink, a glyph ~100. Specks can outnumber glyphs
    ten to one and still not move an ink-weighted statistic.

    Weights are **winsorised at the 95th percentile**, because raw ink weighting has the
    mirror-image failure: a single heavy component (a photo, a logo, a heading) can carry
    more ink than every glyph on the page combined and drag the estimate up by itself.
    Clipping bounds any one component's influence while leaving the speck-versus-glyph
    separation — two orders of magnitude — completely intact.

    The percentile has to be high. A lower one (75th was tried) fails in precisely the
    case this function exists for: when specks outnumber glyphs, the 75th percentile of
    *areas* is itself a speck, every weight clips to 4, and the estimator degenerates
    back into the plain median it was meant to replace.
    """
    if not components:
        return 0.0
    return ink_weighted_median_height_arrays(
        np.fromiter((c.bbox.h for c in components), dtype=np.int64, count=len(components)),
        np.fromiter((c.pixel_area for c in components), dtype=np.int64, count=len(components)),
    )


def ink_weighted_median_height_arrays(heights: np.ndarray, areas: np.ndarray) -> float:
    """:func:`ink_weighted_median_height` over parallel arrays — the one implementation.

    The merge splitter needs the same estimate straight from CCL's ``stats`` array,
    before any :class:`Component` exists for the pieces; both callers go through here so
    the splitter and the filter can never disagree about what one glyph is.
    """
    n = len(heights)
    if n == 0:
        return 0.0
    cap = max(1, int(np.sort(areas)[int(0.95 * (n - 1))]))
    weights = np.minimum(areas, cap)
    order = np.argsort(heights, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = int(np.searchsorted(cumulative, total / 2.0, side="left"))
    return float(heights[order][min(idx, n - 1)])


def _build_x_index(components: list[Component], bucket_width: float) -> dict[int, list[Component]]:
    """Bucket components by x so the diacritic search is local, not quadratic.

    A page can carry several thousand components; an all-pairs parent search would cost
    more than the CCL it is filtering.
    """
    index: dict[int, list[Component]] = {}
    width = max(1.0, bucket_width)
    for c in components:
        lo = int(c.bbox.x // width)
        hi = int((c.bbox.x2) // width)
        for b in range(lo, hi + 1):
            index.setdefault(b, []).append(c)
    return index


def _find_parent(
    small: Component,
    index: dict[int, list[Component]],
    bucket_width: float,
    median_h: float,
    cfg: FilterConfig,
) -> Component | None:
    """Is there a normal-height glyph this mark plausibly belongs to?

    Plausible means: horizontally overlapping (the dot sits over the stem) and
    vertically close (within about one glyph height above or below).
    """
    b = small.bbox
    width = max(1.0, bucket_width)
    lo = int(b.x // width) - 1
    hi = int(b.x2 // width) + 1
    max_gap = cfg.diacritic_y_gap_ratio * median_h
    min_overlap = cfg.diacritic_x_overlap * b.w

    seen: set[int] = set()
    for bucket in range(lo, hi + 1):
        for cand in index.get(bucket, ()):
            if cand.id in seen or cand is small:
                continue
            seen.add(cand.id)

            cb = cand.bbox
            overlap = min(b.x2, cb.x2) - max(b.x, cb.x)
            if overlap < min_overlap:
                continue

            if b.y2 <= cb.y:
                gap = cb.y - b.y2  # mark sits above (dot of i/j, acute, macron)
            elif cb.y2 <= b.y:
                gap = b.y - cb.y2  # mark sits below (cedilla, Thai vowel below)
            else:
                gap = 0  # vertically overlapping — a broken stroke of the same glyph

            if gap <= max_gap:
                return cand
    return None


def _find_line_neighbour(
    small: Component,
    index: dict[int, list[Component]],
    bucket_width: float,
    median_h: float,
    cfg: FilterConfig,
) -> Component | None:
    """Is there a normal-height glyph *beside* this mark, on the same line?

    The complement of :func:`_find_parent`, which only looks above and below. Baseline
    punctuation — full stop, comma, decimal point, hyphen, colon — sits next to its
    neighbour, never over it, so the parent search cannot find it, and before Phase 2b
    such a mark survived only when it happened to be *merged* into the glyph before it.
    Once the merge splitter separated those, they were routed to NOISE as orphans, and
    on a cheque that is "1,234.56" read as "123456".

    Beside means: a small horizontal gap, and the mark sitting in the lower part of the
    neighbour's vertical extent (from mid-height for a hyphen down to just below the
    baseline for a comma's tail). A mark floating above a glyph is a diacritic, and is
    the parent search's business.
    """
    b = small.bbox
    width = max(1.0, bucket_width)
    lo = int(b.x // width) - 1
    hi = int(b.x2 // width) + 1
    max_gap = cfg.punctuation_max_gap_ratio * median_h
    below = cfg.punctuation_below_baseline_ratio * median_h

    seen: set[int] = set()
    for bucket in range(lo, hi + 1):
        for cand in index.get(bucket, ()):
            if cand.id in seen or cand is small:
                continue
            seen.add(cand.id)

            cb = cand.bbox
            gap = max(cb.x - b.x2, b.x - cb.x2)
            if gap < 0 or gap > max_gap:
                continue
            if b.cy < cb.y + 0.25 * cb.h or b.y > cb.y2 + below:
                continue
            return cand
    return None


def apply_tier2(components: list[Component], page_w: int, page_h: int, cfg: FilterConfig) -> FilterStats:
    """Relative gates keyed to the median survivor height, plus diacritic recovery."""
    stats = FilterStats(total=len(components))

    survivors = [c for c in components if c.kind is ComponentKind.TEXT]
    if not cfg.use_relative_gates or len(survivors) < cfg.min_components_for_stats:
        # Too few components for the median to mean anything — a near-empty crop, a
        # single stamp. Absolute gates stand on their own; guessing a scale here would
        # discard the little text there is.
        stats.counts = _count_kinds(components)
        return stats

    page_area = page_w * page_h
    blob_area = cfg.blob_area_frac * page_area

    # Dense regions are routed *before* the scale is estimated. This test is absolute —
    # it needs no median — and leaving a portrait photo in the pool would bias an
    # ink-weighted statistic upward by its whole ink mass.
    for c in survivors:
        if c.pixel_area > blob_area:
            _route(c, ComponentKind.BLOB, "t2:dense_region")

    survivors = [c for c in survivors if c.kind is ComponentKind.TEXT]
    if len(survivors) < cfg.min_components_for_stats:
        stats.counts = _count_kinds(components)
        return stats

    median_h = ink_weighted_median_height(survivors)
    stats.median_height = median_h
    stats.relative_gates_applied = True

    too_tall = cfg.max_height_ratio * median_h
    too_short = cfg.min_height_ratio * median_h

    # Parents for the diacritic search: components still plausibly full-height glyphs.
    normals = [c for c in survivors if c.bbox.h >= 0.6 * median_h]
    index = _build_x_index(normals, median_h)

    for c in survivors:
        if c.bbox.h > too_tall:
            # Headings and drop caps land here too. BLOB rather than NOISE on purpose:
            # this is real text we are declining to classify *as a single character*,
            # and a later pass can re-segment it at a different scale.
            _route(c, ComponentKind.BLOB, "t2:oversized_vs_median")
            continue

        if c.bbox.h < too_short:
            parent = _find_parent(c, index, median_h, median_h, cfg)
            if parent is not None and c.bbox.h <= cfg.diacritic_max_height_ratio * median_h:
                _route(c, ComponentKind.DIACRITIC, "t2:diacritic")
            elif _find_line_neighbour(c, index, median_h, median_h, cfg) is not None:
                # Punctuation is a character in its own right for the classifier, not
                # part of its neighbour — so TEXT, not DIACRITIC.
                c.reason = "t2:punctuation"
            else:
                _route(c, ComponentKind.NOISE, "t2:small_orphan")

    stats.counts = _count_kinds(components)
    return stats


def _count_kinds(components: list[Component]) -> dict[str, int]:
    counts = {k.value: 0 for k in ComponentKind}
    for c in components:
        counts[c.kind.value] += 1
    return counts


def filter_components(
    components: list[Component],
    page_w: int,
    page_h: int,
    cfg: FilterConfig,
    timer: StageTimer | None = None,
) -> FilterStats:
    """Run both tiers. Mutates ``components`` in place and returns diagnostics."""
    timer = timer or StageTimer()
    with timer.stage("filter"):
        apply_tier1(components, page_w, page_h, cfg)
        stats = apply_tier2(components, page_w, page_h, cfg)
    return stats
