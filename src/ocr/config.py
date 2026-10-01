"""Every tunable in one place.

Phase 2's thresholds are guesses until the benchmark tunes them, so they live in a
dataclass the runner can sweep rather than as literals scattered through the filter
code. Defaults below are starting points chosen to be *permissive*: at this stage a
false accept costs one wasted CNN inference, while a false reject costs a character we
can never recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    long_side_cap: int = 1600
    """Downscale so the longer side is at most this. The dominant latency knob: CCL and
    thresholding are both linear in pixel count. 1600 keeps ~10 px glyph height on a
    photographed A4, which is about the floor for recognition."""

    upscale_small: bool = True
    """Scale *up* images whose long side is below ``min_long_side``. A 600 px phone crop
    of an NRIC has 8 px glyphs that CCL fragments; upscaling first costs little."""

    min_long_side: int = 1000

    normalize_illumination: bool = True
    """Morphological background subtraction. Matters for photos (shadow gradients,
    flash hotspots) and is near-free on flatbed scans."""

    background_kernel: int = 31
    """Must exceed the thickest stroke or the glyphs get absorbed into the background
    estimate. Scaled with image size at runtime."""

    denoise: bool = False
    """Median blur before thresholding. Off by default: it closes the gap in thin
    glyphs at low DPI, trading junk components for broken characters."""


@dataclass(frozen=True, slots=True)
class BinarizeConfig:
    method: Literal["adaptive_gaussian", "sauvola", "otsu"] = "adaptive_gaussian"

    block_size: int = 35
    """Adaptive window, forced odd at runtime. Should be a few times the glyph height;
    too small and thick strokes hollow out, too large and it degenerates to Otsu."""

    c: float = 10.0
    """Constant subtracted from the local mean. Higher = more conservative (less ink)."""

    sauvola_k: float = 0.2
    sauvola_window: int = 25

    recover_inverted_regions: bool = True
    """Re-threshold light-on-dark regions with the opposite polarity.

    Without this, a dark ID-card banner or a cheque's MICR strip binarises into one
    solid blob and every character inside it is lost before filtering even runs."""

    inverted_min_area_frac: float = 0.002
    """A blob must cover at least this fraction of the page to be worth re-examining."""

    inverted_min_fill: float = 0.55
    """...and be this solid. A sparse large component is a signature, not an inverted
    region."""


@dataclass(frozen=True, slots=True)
class FilterConfig:
    # --- Tier 1: absolute gates, no page statistics required ---
    min_pixel_area: int = 4
    """Degenerate-component floor only. Kept low on purpose: a decimal point at 150 DPI
    is about 4 px, and losing it turns "1,234.56" into "123456" on a cheque. Separating
    real punctuation from speckle is Tier 2's job, where a neighbour search can tell
    the difference; Tier 1 has no business guessing."""

    min_extent: int = 2
    """A component smaller than this in *both* dimensions is a speck.

    Both, not either. A 800x3 table rule is 3 px tall and is emphatically not noise;
    gating on height alone routes every thin rule to NOISE before the rule check runs,
    and the layout information is gone. Extent in one dimension says nothing about
    whether a component is meaningful."""

    max_width_frac: float = 0.60
    """Fraction of page width above which a component is a rule or a blob, never a
    character."""

    max_height_frac: float = 0.60

    max_aspect: float = 12.0
    """Width/height above which it is a horizontal rule."""

    min_aspect: float = 0.03
    """Below which it is a vertical rule or table border."""

    min_fill_ratio: float = 0.15
    """Ink as a fraction of the bounding box. Hollow boxes, table cells and the frames
    drawn around form fields fall out here.

    Set from the benchmark, not guessed: over the synthetic set, components sitting on a
    ground-truth character have a 1st-percentile fill ratio of 0.34, while a 2 px frame
    around a 190x50 field box sits near 0.10. 0.15 clears the frames with a wide margin
    above the glyphs. (The measurement is censored — it only sees components that
    already survived the current gate — so re-measure after any large change here.)"""

    rule_aspect: float = 8.0
    rule_fill_min: float = 0.40
    """A rule is both elongated *and* solid along its length; an elongated sparse
    component is more likely a sequence of merged glyphs."""

    # --- Tier 2: relative gates, keyed to the median height of Tier-1 survivors ---
    use_relative_gates: bool = True

    min_height_ratio: float = 0.30
    """Below this times the median height, a component is a diacritic or noise."""

    max_height_ratio: float = 4.0
    """Above this times the median height, a component is a heading, a logo, or a
    merged block. Note headings are legitimate text — hence BLOB, not NOISE, so a later
    pass can re-segment rather than lose them."""

    blob_area_frac: float = 0.02
    """Pixel area above this fraction of the page means photo/logo/signature."""

    # --- Diacritic recovery ---
    diacritic_max_height_ratio: float = 0.55
    """A small component is a diacritic rather than noise if it is below this height
    ratio *and* has a parent glyph within the search radius."""

    diacritic_x_overlap: float = 0.30
    """Minimum horizontal overlap with the candidate parent, as a fraction of the small
    component's width."""

    diacritic_y_gap_ratio: float = 1.20
    """Maximum vertical gap to the parent, in units of median height. Covers a dotted
    i/j above and a cedilla below."""

    min_components_for_stats: int = 12
    """Below this count the median height is not trustworthy, so Tier 2 is skipped
    entirely and only the absolute gates apply. Relevant for near-empty crops."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    binarize: BinarizeConfig = field(default_factory=BinarizeConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)

    connectivity: Literal[4, 8] = 8
    """8-connectivity keeps diagonally-touching strokes together; 4 fragments italic
    and script-like glyphs badly."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            preprocess=PreprocessConfig(**d.get("preprocess", {})),
            binarize=BinarizeConfig(**d.get("binarize", {})),
            filters=FilterConfig(**d.get("filters", {})),
            connectivity=d.get("connectivity", 8),
        )


DEFAULT_CONFIG = PipelineConfig()
