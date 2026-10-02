"""The Phase 0-2 engine: preprocess → binarize → CCL → filter.

:class:`Engine` is deliberately defined now, before there is a second implementation.
Phase 6 compares this pipeline against PaddleOCR end to end, and that comparison is
only honest if both sides are driven through one interface by one runner — otherwise
the measured difference includes whatever each harness happens to do around the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .binarize import binarize
from .ccl import label_with_maps
from .config import DEFAULT_CONFIG, PipelineConfig
from .filters import filter_components
from .group import group_components, rescale_lines
from .preprocess import Preprocessed, preprocess
from .split import split_merged
from .timing import StageTimer
from .types import PageResult


@runtime_checkable
class Engine(Protocol):
    """What the benchmark runner needs from any OCR implementation."""

    name: str

    def run(self, image: np.ndarray) -> PageResult:
        """Process one BGR or grayscale image and return boxes (+ text, once Phase 5
        exists) in *original image* coordinates."""
        ...


@dataclass(slots=True)
class DebugArtifacts:
    """Intermediate images, for the overlay tool. Not produced on the timed path."""

    preprocessed: Preprocessed
    binary: np.ndarray


class CCLEngine:
    """Connected-component engine: boxes and layout, no recognition yet.

    ``run`` returns components with :class:`~ocr.types.ComponentKind` already assigned,
    grouped into :attr:`PageResult.lines` of words (Phase 5), all in original-image
    coordinates. Character classification (Phase 3) is not implemented, so word text,
    and :attr:`PageResult.text`, are empty.
    """

    name = "ccl"

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def run(self, image: np.ndarray) -> PageResult:
        result, _ = self._run(image, debug=False)
        return result

    def run_with_debug(self, image: np.ndarray) -> tuple[PageResult, DebugArtifacts]:
        result, artifacts = self._run(image, debug=True)
        assert artifacts is not None
        return result, artifacts

    def _run(self, image: np.ndarray, debug: bool) -> tuple[PageResult, DebugArtifacts | None]:
        cfg = self.config
        timer = StageTimer()

        pre = preprocess(image, cfg.preprocess, timer)
        binary, recovered = binarize(pre.gray, cfg.binarize, cfg.connectivity, timer)
        components, labels, stats = label_with_maps(binary, cfg.connectivity, timer)

        split = split_merged(
            pre.gray, binary, labels, stats, cfg.split, cfg.filters, cfg.connectivity, timer
        )
        binary = split.binary
        components = split.apply(components)
        split_stats = split.stats

        proc_h, proc_w = binary.shape[:2]
        filter_stats = filter_components(components, proc_w, proc_h, cfg.filters, timer)
        lines, rejected_lines, group_stats = group_components(
            components, proc_w, proc_h, filter_stats.median_height, cfg.group, timer
        )

        # Map geometry back to the caller's coordinate frame. pixel_area is deliberately
        # left alone: an ink count does not survive resampling, and fill_ratio was
        # already computed from it at labelling time while both were exact.
        if pre.scale != 1.0:
            inv = 1.0 / pre.scale
            with timer.stage("rescale_boxes"):
                for c in components:
                    c.bbox = c.bbox.scaled(inv)
                    c.centroid = (c.centroid[0] * inv, c.centroid[1] * inv)
                rescale_lines(lines, inv)
                rescale_lines(rejected_lines, inv)

        orig_h, orig_w = pre.original_size
        result = PageResult(
            width=orig_w,
            height=orig_h,
            components=components,
            lines=lines,
            rejected_lines=rejected_lines,
            timings_ms=timer.as_dict(),
            scale=pre.scale,
            meta={
                "polarity_inverted": pre.polarity_inverted,
                "recovered_inverted_regions": recovered,
                "processed_size": [proc_w, proc_h],
                "filter": filter_stats.as_dict(),
                "split": split_stats.as_dict(),
                "group": group_stats.as_dict(),
            },
        )

        artifacts = DebugArtifacts(preprocessed=pre, binary=binary) if debug else None
        return result, artifacts


def load_image(path: str) -> np.ndarray:
    """Read an image, raising a useful error instead of returning None.

    ``cv2.imread`` fails silently on a missing file and on non-ASCII paths on Windows,
    so go through numpy/imdecode.
    """
    with open(path, "rb") as fh:
        buf = np.frombuffer(fh.read(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image
