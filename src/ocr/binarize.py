"""Binarisation, plus recovery of locally-inverted regions.

Output convention everywhere below: **ink is 255, background is 0**, so CCL labels
characters rather than the page.

The inverted-region recovery exists because of one specific failure that this
architecture cannot survive. ID cards and cheques carry dark banners with light text
(an NRIC header strip, a MICR band, a reversed table header). Global polarity
normalisation keeps the *page* right-way-up, which means those regions threshold into a
single solid rectangle — and every character inside is gone before filtering even gets
a chance to look at it. No downstream stage can recover a character that was never a
component.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import BinarizeConfig
from .timing import StageTimer

MAX_RECOVERED_REGIONS = 8
"""Latency guard. Re-thresholding is cheap per region but unbounded otherwise, and a
page with dozens of large solid blobs is a photograph, not a document."""


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def adaptive_gaussian(gray: np.ndarray, cfg: BinarizeConfig) -> np.ndarray:
    block = max(3, _odd(cfg.block_size))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.c
    )


def otsu(gray: np.ndarray, cfg: BinarizeConfig) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def sauvola(gray: np.ndarray, cfg: BinarizeConfig) -> np.ndarray:
    """Sauvola's local threshold, via box filters rather than a Python loop.

    t(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1)), R = 128 for 8-bit input.

    Lowers the threshold in low-variance areas, which suppresses the speckle that
    ``adaptive_gaussian`` produces across blank page regions — at roughly 2-3x the
    cost, which is why it is not the default.
    """
    win = max(3, _odd(cfg.sauvola_window))
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = cv2.sqrt(cv2.max(mean_sq - mean * mean, 0.0))
    threshold = mean * (1.0 + cfg.sauvola_k * (std / 128.0 - 1.0))
    return ((f < threshold) * 255).astype(np.uint8)


_METHODS = {
    "adaptive_gaussian": adaptive_gaussian,
    "sauvola": sauvola,
    "otsu": otsu,
}


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """A blob's mask including its interior holes.

    This matters more than it looks. In a dark banner with light text, the banner's
    *ink* pixels are the background around the letters, and the letters themselves are
    holes. Masking the re-thresholded region to the blob's ink would therefore erase
    exactly the characters recovery just found. Filling the holes first keeps the patch
    confined to the banner — so neighbouring text is still safe — while letting the
    recovered glyphs through.
    """
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8) * 255
    flood = np.zeros((h + 4, w + 4), dtype=np.uint8)
    # Flood the background from outside the blob; whatever stays 0 is enclosed.
    cv2.floodFill(padded, flood, (0, 0), 255)
    holes = padded[1:-1, 1:-1] == 0
    return (mask > 0) | holes


def _looks_like_text(roi_binary: np.ndarray, connectivity: int = 8) -> bool:
    """Does a candidate re-thresholded region contain plausible glyphs?

    The bar is deliberately crude: several components whose height is a small fraction
    of the region. Anything that passes is better than the solid blob it replaces, and
    anything that fails leaves the blob untouched — so a false negative here costs
    nothing we did not already lose.
    """
    h, w = roi_binary.shape[:2]
    if h < 6 or w < 6:
        return False
    if np.count_nonzero(roi_binary) / roi_binary.size > 0.5:
        return False  # still mostly ink — we just flipped the blob over

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        roi_binary, connectivity=connectivity, ltype=cv2.CV_32S
    )
    glyph_like = 0
    for i in range(1, count):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 4 and 3 <= ch <= h * 0.8:
            glyph_like += 1
            if glyph_like >= 3:
                return True
    return False


def recover_inverted_regions(
    gray: np.ndarray,
    binary: np.ndarray,
    cfg: BinarizeConfig,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """Re-threshold large solid components with the opposite polarity.

    Returns the patched binary and how many regions were recovered. Patching is masked
    to the blob's own pixels, so text sitting next to a dark banner is never clobbered
    by the banner's recovery.
    """
    page_area = binary.shape[0] * binary.shape[1]
    min_area = max(64, int(cfg.inverted_min_area_frac * page_area))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S
    )
    candidates = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area / (bw * bh) < cfg.inverted_min_fill:
            continue  # sparse: a signature or a scribble, not a filled banner
        candidates.append((area, i))

    if not candidates:
        return binary, 0

    candidates.sort(reverse=True)
    out = binary.copy()
    recovered = 0

    for _, label in candidates[:MAX_RECOVERED_REGIONS]:
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        roi_gray = gray[y : y + bh, x : x + bw]
        flipped = cv2.bitwise_not(roi_gray)
        roi_binary = _METHODS[cfg.method](flipped, cfg)

        mask = _fill_holes(labels[y : y + bh, x : x + bw] == label)
        candidate = np.where(mask, roi_binary, 0).astype(np.uint8)

        if _looks_like_text(candidate, connectivity):
            region = out[y : y + bh, x : x + bw]
            region[mask] = candidate[mask]
            recovered += 1

    return out, recovered


def binarize(
    gray: np.ndarray,
    cfg: BinarizeConfig,
    connectivity: int = 8,
    timer: StageTimer | None = None,
) -> tuple[np.ndarray, int]:
    """Threshold to ink=255, then recover locally-inverted regions.

    Returns ``(binary, n_recovered_regions)``.
    """
    timer = timer or StageTimer()
    recovered = 0
    with timer.stage("binarize"):
        method = _METHODS.get(cfg.method)
        if method is None:
            raise ValueError(f"unknown binarize method: {cfg.method!r}")
        binary = method(gray, cfg)

        if cfg.recover_inverted_regions:
            binary, recovered = recover_inverted_regions(gray, binary, cfg, connectivity)

    return binary, recovered
