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

    # --- Punctuation recovery ---
    punctuation_max_gap_ratio: float = 0.6
    """A small mark is punctuation rather than noise if a normal glyph sits beside it
    within this horizontal gap (in units of median height). Covers the space before a
    full stop and the tight spacing of "1,234.56"; a word gap is wider."""

    punctuation_below_baseline_ratio: float = 0.35
    """How far below the neighbour's bottom a punctuation mark may reach — a comma's
    tail descends below the baseline."""

    min_components_for_stats: int = 12
    """Below this count the median height is not trustworthy, so Tier 2 is skipped
    entirely and only the absolute gates apply. Relevant for near-empty crops."""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Splitting of merged glyphs, between CCL and filtering (Phase 2b).

    Merging, not loss, is the dominant isolation failure: on photographs 37.8% of
    glyphs reached the classifier fused with a neighbour. The page-wide adaptive window
    (``BinarizeConfig.block_size``, 35 px) is about twice the glyph height, so it
    averages across neighbours and fills the grey seam between touching glyphs.
    Shrinking it everywhere opens the seams but breaks strokes inside glyphs, and
    plateaus around 68-70% isolation on photos. So the large window stays, and only
    components too wide to be one glyph are re-examined. All ratios are relative to the
    page's glyph height / width, measured with the filter's own robust estimator.

    Defaults were chosen by ``scripts/tune_split.py`` on a *held-out* synthetic set
    (seed 11), never on the published benchmark (seed 7) the results are reported on.
    The plateau is flat: every re-threshold setting in the grid landed within ~2 points
    of the best, which is what a real effect looks like rather than a fitted one.
    """

    enabled: bool = True

    suspect_width_ratio: float = 1.2
    """A component at least this many median glyph widths wide is a merge suspect."""

    suspect_min_height_ratio: float = 0.6
    suspect_max_height_ratio: float = 2.5
    """Only glyph-height components are suspects. Shorter ones are punctuation, rules
    and underlines; taller ones are headings or blobs that Tier 2 routes anyway."""

    max_suspects: int = 1500
    """Latency guard, like ``MAX_RECOVERED_REGIONS`` in binarisation. Suspects are
    processed widest first, so a cap drops the narrowest — the least likely to hold
    several glyphs. Hit only by pathological pages: the noisiest receipt in the
    benchmark measures 5 px "glyphs" and yields ~4,500 suspects."""

    small_block_ratio: float = 0.4
    """Window of the local re-threshold, as a fraction of glyph height. Small enough
    to resolve a one-pixel grey seam between glyphs."""

    small_c: float = 10.0

    min_part_height: float = 0.4
    """A re-threshold is accepted only if it yields >= 2 parts each at least this
    fraction of the suspect's height. Below that, the small window has broken strokes
    rather than separated glyphs, and the suspect is left as it was."""

    split_touching_columns: bool = False
    """Cut what is still touching at low-ink columns.

    **Off by default, deliberately.** Column cutting assumes glyphs are separate
    shapes that touch at thin points, which is a Latin assumption. Arabic letters
    within a word are joined, and Devanagari words hang from a continuous headline
    (shirorekha); cutting at low-ink columns destroys both. Script detection arrives in
    Phase 7, which should enable this per region once a region is tagged Latin."""

    cut_max_ink: float = 0.3
    """A column is a cut candidate if its ink is at most this fraction of the
    component's densest column."""

    min_piece_width: float = 0.8
    """Pieces narrower than this many median glyph widths are not produced; an ``m``
    is not three ``i``s."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    binarize: BinarizeConfig = field(default_factory=BinarizeConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
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
            split=SplitConfig(**d.get("split", {})),
            filters=FilterConfig(**d.get("filters", {})),
            connectivity=d.get("connectivity", 8),
        )


DEFAULT_CONFIG = PipelineConfig()
