"""Component filtering — the cheap geometric pass that protects the classifier.

Two tiers, cheapest first:

* **Tier 1** uses only a component's own geometry, so it needs no page statistics and
  can reject obvious junk before we have looked at the page as a whole.
* **Tier 2** keys off the *median height of Tier-1 survivors*, which is a robust
  estimate of "how tall is a character on this page" precisely because Tier 1 has
  already removed the rules and blobs that would skew it.

The governing principle is that **filtering routes rather than deletes**. Only
:attr:`ComponentKind.NOISE` is thrown away. Everything else is labelled and kept:
diacritics because dropping them silently caps accuracy in a way no later stage can
undo, rules because Phase 5 needs them for layout, blobs because Phase 11 needs them
for photo masking. Each decision records which gate fired, so the overlay can show
*why* a component was routed where it was and thresholds can be tuned from evidence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

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

    median_h = float(statistics.median(c.bbox.h for c in survivors))
    stats.median_height = median_h
    stats.relative_gates_applied = True

    page_area = page_w * page_h
    blob_area = cfg.blob_area_frac * page_area
    too_tall = cfg.max_height_ratio * median_h
    too_short = cfg.min_height_ratio * median_h

    # Parents for the diacritic search: components still plausibly full-height glyphs.
    normals = [c for c in survivors if c.bbox.h >= 0.6 * median_h]
    index = _build_x_index(normals, median_h)

    for c in survivors:
        if c.pixel_area > blob_area:
            _route(c, ComponentKind.BLOB, "t2:dense_region")
            continue

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
