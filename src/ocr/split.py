"""Merge splitting: separate glyphs that CCL labelled as one component (Phase 2b).

Merging is the dominant isolation failure, and the one a classifier cannot repair: by
the time it sees the crop, two glyphs are one blob. On photographs 37.8% of glyphs
were merged, against 8.2% missed. They are not merged by rotation (deskew would not
help) nor by too few pixels (photos are already upscaled to ~16 px glyphs before
thresholding). The neighbours' ground-truth boxes have a median gap of **0 px**: after
capture blur the seam between them is a grey ridge one pixel wide, and the page-wide
adaptive window — about twice the glyph height — averages it into the ink.

Shrinking the window everywhere opens those seams but breaks strokes *inside* glyphs:
isolation on photos plateaus at 68-70% whatever the window, trading merges for misses.
So the page is thresholded as before, and only components too wide to be one glyph
are re-examined:

1. **Re-threshold** the suspect with a window sized to the page's glyph height, inside
   the suspect's own mask, and keep the result only if it yields at least two
   *whole-height* parts. A finer threshold that merely fragments strokes is rejected,
   so this step cannot make a glyph worse than it was.
2. **Cut at low-ink columns** what still touches — optional, off by default, because
   it is a Latin assumption (see :class:`~ocr.config.SplitConfig`).

**Latency.** Splitting only ever *removes* ink from inside one component, so the pieces
it produces cannot touch anything outside that component. They are therefore labelled
locally, inside the suspect's box, and swapped in for the suspect — the page is never
relabelled. (A full relabel rebuilt every component object in Python and was the
largest cost of this stage: +40-60 ms on every cheque.) Everything that runs over *all*
components — the glyph-scale estimate, suspect selection — is array arithmetic on
CCL's ``stats``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import FilterConfig, SplitConfig
from .filters import glyph_height_estimate, tier1_text_mask
from .timing import StageTimer
from .types import BBox, Component


@dataclass(slots=True)
class SplitStats:
    glyph_height: float = 0.0
    glyph_width: float = 0.0
    suspects: int = 0
    suspects_capped: bool = False
    rethresholded: int = 0
    """Suspects replaced by their local re-threshold."""
    column_cuts: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SplitResult:
    binary: np.ndarray
    """The patched binary — the same object as the input when nothing was split."""

    removed: set[int] = field(default_factory=set)
    """Labels (== ``Component.id``) of the suspects that were split."""

    added: list[Component] = field(default_factory=list)
    """The pieces that replace them, in processing coordinates, flagged ``split``."""

    stats: SplitStats = field(default_factory=SplitStats)

    def apply(self, components: list[Component]) -> list[Component]:
        """``components`` with the split suspects swapped for their pieces."""
        if not self.removed:
            return components
        return [c for c in components if c.id not in self.removed] + self.added


def page_glyph_scale(
    stats: np.ndarray, page_w: int, page_h: int, cfg: FilterConfig
) -> tuple[float, float] | None:
    """(median glyph height, median glyph width) in processing pixels, or ``None``
    when the page has too few glyph-like components to say.

    The same estimate the filter makes — Tier-1 survivors, dense regions excluded,
    ink-weighted median height — computed on CCL's ``stats`` array, so the splitter and
    the filter agree on what one glyph is.
    """
    text = tier1_text_mask(stats, page_w, page_h, cfg)
    areas = stats[:, cv2.CC_STAT_AREA]
    text &= areas <= cfg.blob_area_frac * page_w * page_h
    if int(text.sum()) < cfg.min_components_for_stats:
        return None

    heights = stats[text, cv2.CC_STAT_HEIGHT]
    h = glyph_height_estimate(stats[text, cv2.CC_STAT_LEFT], stats[text, cv2.CC_STAT_TOP],
                              stats[text, cv2.CC_STAT_WIDTH], heights, areas[text])
    if h <= 0:
        return None
    widths = stats[text, cv2.CC_STAT_WIDTH][(heights >= 0.7 * h) & (heights <= 1.3 * h)]
    w = float(np.median(widths)) if len(widths) else 0.6 * h
    return h, w


def select_suspects(stats: np.ndarray, glyph_h: float, glyph_w: float, cfg: SplitConfig) -> np.ndarray:
    """Labels of merge suspects, widest first, selected without a Python loop."""
    st = stats[1:]
    h = st[:, cv2.CC_STAT_HEIGHT]
    w = st[:, cv2.CC_STAT_WIDTH]
    keep = (
        (h >= cfg.suspect_min_height_ratio * glyph_h)
        & (h <= cfg.suspect_max_height_ratio * glyph_h)
        & (w >= cfg.suspect_width_ratio * glyph_w)
    )
    labels = np.flatnonzero(keep) + 1
    return labels[np.argsort(-stats[labels, cv2.CC_STAT_WIDTH], kind="stable")]


def _whole_height_parts(
    sub: np.ndarray, min_h: float, connectivity: int
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Labels of the parts of ``sub`` at least ``min_h`` tall, plus the label image,
    stats and centroids they index into — reused as the pieces themselves when no
    column cut follows, so an accepted suspect is labelled once, not twice."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sub, connectivity=connectivity, ltype=cv2.CV_32S
    )
    parts = [j for j in range(1, count) if stats[j, cv2.CC_STAT_HEIGHT] >= min_h]
    return parts, labels, stats, centroids


def cut_columns(region: np.ndarray, mask: np.ndarray, glyph_w: float, cfg: SplitConfig) -> int:
    """Zero low-ink columns of ``mask`` inside ``region`` (in place). Returns cuts made.

    Scans for the deepest valley of the column ink profile within each glyph-width
    window, and cuts there if it is shallow enough relative to the densest column.
    Pieces narrower than ``min_piece_width`` glyph widths are never produced, which is
    what keeps an ``m`` from becoming three ``i``s.
    """
    ink = (region > 0) & mask
    profile = ink.sum(axis=0)
    width = profile.shape[0]
    peak = int(profile.max()) if width else 0
    min_piece = max(2, int(cfg.min_piece_width * glyph_w))
    if peak == 0 or width < 2 * min_piece:
        return 0

    cuts: list[int] = []
    last = 0
    x = min_piece
    step = max(1, int(0.6 * glyph_w))
    while x <= width - min_piece:
        hi = min(width - min_piece, x + step)
        j = x + int(np.argmin(profile[x : hi + 1]))
        if profile[j] <= cfg.cut_max_ink * peak and j - last >= min_piece:
            cuts.append(j)
            last = j
            x = j + min_piece
        else:
            x = hi + 1
    for j in cuts:
        region[mask[:, j], j] = 0
    return len(cuts)


def _pieces(region: np.ndarray, mask: np.ndarray, x0: int, y0: int,
            connectivity: int, next_id: int) -> list[Component]:
    """Label what remains of one suspect, as components in page coordinates."""
    local = np.where(mask, region, 0).astype(np.uint8)
    _, _, st, centroids = cv2.connectedComponentsWithStats(
        local, connectivity=connectivity, ltype=cv2.CV_32S
    )
    return _components_at(st, centroids, x0, y0, next_id)


def _components_at(st: np.ndarray, centroids: np.ndarray, x0: int, y0: int,
                   next_id: int) -> list[Component]:
    """Components from a local ``stats``/``centroids`` pair, offset to page coordinates."""
    out: list[Component] = []
    for j in range(1, len(st)):
        x, y, w, h, ink = (int(v) for v in st[j])
        box = BBox(x0 + x, y0 + y, w, h)
        out.append(Component(
            id=next_id + len(out),
            bbox=box,
            pixel_area=ink,
            centroid=(x0 + float(centroids[j, 0]), y0 + float(centroids[j, 1])),
            fill_ratio=ink / box.area if box.area else 0.0,
            split=True,
        ))
    return out


def split_merged(
    gray: np.ndarray,
    binary: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    cfg: SplitConfig,
    filter_cfg: FilterConfig,
    connectivity: int = 8,
    timer: StageTimer | None = None,
) -> SplitResult:
    """Split merge suspects. See the module docstring for the method."""
    timer = timer or StageTimer()
    result = SplitResult(binary=binary)
    if not cfg.enabled:
        return result

    with timer.stage("split"):
        page_h, page_w = binary.shape[:2]
        scale = page_glyph_scale(stats, page_w, page_h, filter_cfg)
        if scale is None:
            return result
        glyph_h, glyph_w = scale
        result.stats.glyph_height, result.stats.glyph_width = glyph_h, round(glyph_w, 2)

        suspects = select_suspects(stats, glyph_h, glyph_w, cfg)
        result.stats.suspects = int(len(suspects))
        if len(suspects) > cfg.max_suspects:
            suspects = suspects[: cfg.max_suspects]
            result.stats.suspects_capped = True
        if not len(suspects):
            return result

        block = max(7, int(round(cfg.small_block_ratio * glyph_h)) | 1)
        # One page-wide pass rather than one per suspect: an adaptive threshold
        # computed on a crop sees different borders, and suspects are often many.
        fine = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.small_c
        )
        out = binary.copy()
        wide = cfg.suspect_width_ratio * glyph_w
        next_id = len(stats)  # above every label CCL assigned, so ids stay unique

        for label in suspects:
            x, y, w, h = (int(v) for v in stats[label, :4])
            mask = labels[y : y + h, x : x + w] == label
            region = out[y : y + h, x : x + w]
            sub = np.where(mask, fine[y : y + h, x : x + w], 0).astype(np.uint8)

            parts, sub_labels, part_stats, part_centroids = _whole_height_parts(
                sub, cfg.min_part_height * h, connectivity
            )
            pieces: list[Component] = []
            if len(parts) >= 2:
                region[mask] = sub[mask]
                result.stats.rethresholded += 1
                cuts = 0
                if cfg.split_touching_columns:
                    # Parts that are still too wide get the column cut too.
                    for j in parts:
                        if part_stats[j, cv2.CC_STAT_WIDTH] >= wide:
                            cuts += cut_columns(region, sub_labels == j, glyph_w, cfg)
                    result.stats.column_cuts += cuts
                pieces = (
                    _pieces(region, mask, x, y, connectivity, next_id) if cuts
                    # Uncut, the suspect's new content is exactly ``sub``: its parts,
                    # already labelled above, are the pieces.
                    else _components_at(part_stats, part_centroids, x, y, next_id)
                )
            elif cfg.split_touching_columns:
                cuts = cut_columns(region, mask, glyph_w, cfg)
                result.stats.column_cuts += cuts
                if cuts:
                    pieces = _pieces(region, mask, x, y, connectivity, next_id)

            if pieces:
                next_id += len(pieces)
                result.removed.add(int(label))
                result.added.extend(pieces)

    if result.removed:
        result.binary = out
    return result
