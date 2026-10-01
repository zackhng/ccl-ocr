"""Grayscale, rescale, polarity normalisation, illumination normalisation.

Order is load-bearing. Polarity is normalised *before* illumination correction because
the background estimator assumes dark ink on a light ground; running it on an inverted
image estimates the background from the text and erases it. Illumination correction
runs *before* thresholding because an adaptive threshold can absorb a gentle shadow
gradient but not a hard flash hotspot, and the hotspot is what phone captures produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PreprocessConfig
from .timing import StageTimer


@dataclass(slots=True)
class Preprocessed:
    gray: np.ndarray
    """Single-channel uint8, rescaled, polarity-normalised to dark-ink-on-light."""

    scale: float
    """Factor applied to the original. Multiply pipeline boxes by ``1 / scale`` to map
    them back to original-image coordinates."""

    original_size: tuple[int, int]
    """(height, width) of the input."""

    polarity_inverted: bool
    """True when the source was light ink on a dark ground and we flipped it."""


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image if image.dtype == np.uint8 else cv2.convertScaleAbs(image)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_scale(height: int, width: int, cfg: PreprocessConfig) -> float:
    """Pick the resize factor: cap the long side, optionally floor it."""
    long_side = max(height, width)
    if long_side > cfg.long_side_cap:
        return cfg.long_side_cap / long_side
    if cfg.upscale_small and long_side < cfg.min_long_side:
        return cfg.min_long_side / long_side
    return 1.0


def resize(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    h, w = gray.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # AREA for downscale (anti-aliases, keeps strokes connected); CUBIC for upscale
    # (keeps edges crisp enough to threshold, where LINEAR smears thin strokes away).
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, size, interpolation=interp)


def detect_polarity(gray: np.ndarray) -> bool:
    """True when the image is light ink on a dark ground.

    Decided by Otsu: if the majority of pixels fall on the dark side of the split, the
    dark class is the *background*, not the ink. Documents are overwhelmingly mostly
    background, which is what makes this cheap test reliable.

    The comparison must be ``<=``, not ``<``. OpenCV's Otsu threshold is inclusive —
    it splits into ``<= t`` and ``> t`` — and it lands *on* the dark peak for a
    strongly bimodal image. With ``<``, a page that is 96% dark ink-coloured pixels
    measures as 0% dark and the polarity check inverts its answer exactly when the
    signal is clearest.
    """
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fraction = float(np.count_nonzero(gray <= thresh)) / gray.size
    return dark_fraction > 0.5


def normalize_illumination(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Divide out a morphological background estimate.

    Closing with a kernel wider than any stroke removes the ink and leaves the page
    illumination; dividing by it flattens shadows and hotspots while preserving local
    contrast. The kernel must exceed the thickest stroke or glyphs get absorbed into
    the estimate and vanish.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Floor at 1 to avoid dividing by zero on a fully black background estimate.
    # np.maximum, not `background + 1`: uint8 addition wraps 255 to 0, which would
    # zero out every pixel of a white page.
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def preprocess(
    image: np.ndarray,
    cfg: PreprocessConfig,
    timer: StageTimer | None = None,
) -> Preprocessed:
    timer = timer or StageTimer()
    with timer.stage("preprocess"):
        gray = to_grayscale(image)
        h, w = gray.shape[:2]

        scale = compute_scale(h, w, cfg)
        gray = resize(gray, scale)

        inverted = detect_polarity(gray)
        if inverted:
            gray = cv2.bitwise_not(gray)

        if cfg.denoise:
            gray = cv2.medianBlur(gray, 3)

        if cfg.normalize_illumination:
            long_side = max(gray.shape[:2])
            # Scale the kernel with the image so the same config works on a 400 px
            # cheque crop and a 1600 px page.
            kernel_size = max(cfg.background_kernel, long_side // 50)
            gray = normalize_illumination(gray, kernel_size)

    return Preprocessed(
        gray=gray,
        scale=scale,
        original_size=(h, w),
        polarity_inverted=inverted,
    )
