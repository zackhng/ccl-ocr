"""Core data types shared by the CCL pipeline and the benchmark harness.

Everything downstream speaks in :class:`BBox` and :class:`Component`. Boxes are in
*original image* pixel coordinates — the pipeline works on a downscaled copy for speed
but maps every box back before returning, so a caller never has to know the scale
factor was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box, (x, y) top-left, width/height in pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        """Area of the *box*. Distinct from a component's ink-pixel count."""
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def intersection_area(self, other: BBox) -> int:
        dx = min(self.x2, other.x2) - max(self.x, other.x)
        dy = min(self.y2, other.y2) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0
        return dx * dy

    def iou(self, other: BBox) -> float:
        inter = self.intersection_area(other)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def coverage_of(self, other: BBox) -> float:
        """Fraction of ``other`` that this box covers. Asymmetric, unlike IoU."""
        return self.intersection_area(other) / other.area if other.area else 0.0

    def vertical_overlap(self, other: BBox) -> float:
        """Fraction of the *shorter* box's height that overlaps vertically."""
        dy = min(self.y2, other.y2) - max(self.y, other.y)
        if dy <= 0:
            return 0.0
        return dy / min(self.h, other.h)

    def scaled(self, factor: float) -> BBox:
        return BBox(
            round(self.x * factor),
            round(self.y * factor),
            max(1, round(self.w * factor)),
            max(1, round(self.h * factor)),
        )

    def union(self, other: BBox) -> BBox:
        x, y = min(self.x, other.x), min(self.y, other.y)
        return BBox(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, v: list[int] | tuple[int, int, int, int]) -> BBox:
        return cls(int(v[0]), int(v[1]), int(v[2]), int(v[3]))

    @classmethod
    def from_points(cls, pts) -> BBox:
        """Build the enclosing box of an (N, 2) sequence of points.

        Used by adapters whose source annotations are quadrangles (MIDV document and
        face corners, SROIE's 8-coordinate boxes) rather than axis-aligned rectangles.
        """
        xs = [int(round(p[0])) for p in pts]
        ys = [int(round(p[1])) for p in pts]
        x, y = min(xs), min(ys)
        return cls(x, y, max(1, max(xs) - x), max(1, max(ys) - y))


class ComponentKind(str, Enum):
    """Where a component is routed after filtering.

    Filtering *routes*, it does not delete: only :attr:`NOISE` is genuinely discarded.
    A diacritic dropped here is an accuracy loss no later stage can recover, and a rule
    or photo blob is information the layout and PII stages will want.
    """

    TEXT = "text"
    """Character candidate — the input to the Phase 3 classifier."""

    DIACRITIC = "diacritic"
    """Small mark with a plausible parent glyph nearby (dot of an i, accents, Thai
    vowel signs). Kept for the Phase 5 merge step."""

    RULE = "rule"
    """Long thin run — table border, underline, the line under a cheque payee field.
    Kept for layout, never classified."""

    BLOB = "blob"
    """Oversized or dense region — portrait photo, logo, signature, barcode. Routed to
    the Phase 11 photo path."""

    NOISE = "noise"
    """Speckle, scanner dust, compression artefact. Discarded."""


@dataclass(slots=True)
class Component:
    """One connected component, with the filter's verdict attached.

    ``pixel_area`` is the ink-pixel count from CCL, *not* ``bbox.area``. The ratio
    between them (:attr:`fill_ratio`) is the cheapest signal separating a glyph from a
    hollow table cell or a diagonal rule, so the two must not be conflated.

    Coordinate note: :attr:`bbox` and :attr:`centroid` are mapped back to original-image
    coordinates before the engine returns, but :attr:`pixel_area` stays in the
    pipeline's internal (downscaled) frame — an ink count has no meaningful
    interpretation after resampling. :attr:`fill_ratio` is therefore computed once at
    labelling time and stored, not derived from ``pixel_area / bbox.area`` afterwards:
    deriving it would divide two differently-rounded quantities from two different
    coordinate systems and can exceed 1.0.
    """

    id: int
    bbox: BBox
    pixel_area: int
    centroid: tuple[float, float]
    fill_ratio: float = 0.0
    kind: ComponentKind = ComponentKind.TEXT
    reason: str = ""
    """Which gate decided :attr:`kind`. Diagnostic only — drives the overlay legend."""

    @property
    def aspect_ratio(self) -> float:
        """Width over height. Tall glyphs < 1, wide rules >> 1."""
        return self.bbox.w / self.bbox.h if self.bbox.h else 0.0

    @property
    def is_text_candidate(self) -> bool:
        return self.kind is ComponentKind.TEXT


@dataclass(slots=True)
class Word:
    """A horizontal run of components believed to form one word.

    Produced only by the benchmark's word-level fallback metric for now; Phase 5 will
    produce these for real, with transcriptions attached.
    """

    bbox: BBox
    components: list[Component] = field(default_factory=list)
    text: str = ""


@dataclass(slots=True)
class Line:
    """A text line: words sorted left-to-right. Populated in Phase 5."""

    bbox: BBox
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if w.text)


@dataclass(slots=True)
class PageResult:
    """What one pass of the engine returns for one image."""

    width: int
    height: int
    components: list[Component] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    scale: float = 1.0
    """Downscale factor applied internally. Boxes are already mapped back; this is
    recorded so the benchmark can report how much resolution the cap cost."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Non-geometric diagnostics: polarity flip, recovered inverted regions, filter
    counts. Reported by the benchmark, never consumed by the pipeline."""

    @property
    def text_candidates(self) -> list[Component]:
        return [c for c in self.components if c.kind is ComponentKind.TEXT]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def by_kind(self, kind: ComponentKind) -> list[Component]:
        return [c for c in self.components if c.kind is kind]

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {k.value: 0 for k in ComponentKind}
        for c in self.components:
            counts[c.kind.value] += 1
        return counts
