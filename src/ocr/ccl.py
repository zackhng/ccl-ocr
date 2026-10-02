"""Connected component labelling.

Thin wrapper over OpenCV's SAUF/BBDT implementation — the point is not to reimplement
CCL but to give every component the two measurements Phase 2 reasons about: the box,
and the *ink pixel count* inside it. OpenCV returns both; code that forgets the
distinction and uses box area as a proxy for ink cannot tell a letter from an empty
table cell.
"""

from __future__ import annotations

import cv2
import numpy as np

from .timing import StageTimer
from .types import BBox, Component


def _components_from_stats(stats: np.ndarray, centroids: np.ndarray) -> list[Component]:
    components: list[Component] = []
    # Label 0 is the background by construction — skip it.
    for i in range(1, len(stats)):
        box = BBox(
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        ink = int(stats[i, cv2.CC_STAT_AREA])
        components.append(
            Component(
                id=i,
                bbox=box,
                pixel_area=ink,
                centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                # Computed here, in processing coordinates, where both quantities
                # are exact. See Component's coordinate note.
                fill_ratio=ink / box.area if box.area else 0.0,
            )
        )
    return components


def label_with_maps(
    binary: np.ndarray,
    connectivity: int = 8,
    timer: StageTimer | None = None,
) -> tuple[list[Component], np.ndarray, np.ndarray]:
    """Label, and also return OpenCV's ``labels`` image and ``stats`` array.

    The merge splitter needs per-pixel membership (to re-threshold one component
    without touching its neighbours) and wants to select suspects with array
    operations on ``stats`` rather than a Python loop over every component.
    ``components[i]`` corresponds to label ``i + 1``.
    """
    timer = timer or StageTimer()
    with timer.stage("ccl"):
        # Keyword arguments are mandatory here. The positional signature is
        # (image, labels, stats, centroids, connectivity, ltype), so passing
        # connectivity third silently lands it in an output-array slot and leaves
        # connectivity at its default of 8 -- the setting appears to work and does
        # nothing.
        _count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity, ltype=cv2.CV_32S
        )
        components = _components_from_stats(stats, centroids)
    return components, labels, stats


def label_components(
    binary: np.ndarray,
    connectivity: int = 8,
    timer: StageTimer | None = None,
) -> list[Component]:
    """Label an ink=255 binary image.

    Boxes are in the coordinate frame of ``binary`` (i.e. post-resize); the engine maps
    them back to original-image coordinates.
    """
    components, _labels, _stats = label_with_maps(binary, connectivity, timer)
    return components


def label_map(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """The raw label image, for callers that need per-pixel membership.

    Not used on the latency path — Phase 3 will want it to crop glyph masks rather than
    rectangles, so that a tall neighbour's stroke does not bleed into a crop.
    """
    _, labels = cv2.connectedComponents(binary, connectivity=connectivity, ltype=cv2.CV_32S)
    return labels
