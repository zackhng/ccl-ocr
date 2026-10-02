"""Glyph clusters and their CNN inputs — one implementation for training and inference.

The CNN is trained on exactly what it will see at inference, so the crop, the cluster
definition and the geometry features live here and nowhere else.

**Cluster.** A full-height component plus the small marks the filter anchored to it
(``Component.anchor_id``): the dot of an ``i``, an accent, and Vietnamese's stacked
marks (``ế`` is a base and two marks). The CNN classifies the cluster as one composed
letter. Baseline punctuation is anchored too but is a character in its own right, so it
forms its own cluster (it is TEXT, not DIACRITIC).

**Crop.** Each glyph is cropped at its *line's* scale — a square window spanning the
line's band from above the cap line to below the baseline, centred on the glyph — then
resized to 32x32. A glyph-scaled crop (CNN v1) made ``c``/``C``, ``o``/``O``, ``s``/``S``
identical images, and real-document case errors were rife ("AkademiSCheS").

**Geometry.** Five features add the remaining position information, measured against
the cluster's line:
``h/g``, ``w/g``, ``(top - line_top)/g``, ``(bottom - line_base)/g``, ``w/h``
where ``g`` is the line's typical glyph height.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import BBox, Component, ComponentKind, Line, PageResult

CROP = 32
N_GEOMETRY = 5
ABOVE = 0.35
"""Window above the line's cap line, in line heights: room for accents (Vietnamese
stacks two marks above a capital)."""
BELOW = 0.35
"""Window below the baseline, in line heights: room for descenders and commas."""


@dataclass(slots=True)
class Cluster:
    bbox: BBox
    members: list[Component]
    line_index: int
    """Index into ``PageResult.lines + PageResult.rejected_lines``."""
    word_index: int
    """Index of the word within its line."""
    rejected: bool
    """From a rejected (junk-gated) line."""


def clusters_of(result: PageResult) -> list[Cluster]:
    """Every cluster in reading order: emitted lines, then rejected lines."""
    out: list[Cluster] = []
    lines: list[Line] = result.lines + result.rejected_lines
    for li, line in enumerate(lines):
        for wi, word in enumerate(line.words):
            marks: dict[int, list[Component]] = {}
            for c in word.components:
                if c.kind is ComponentKind.DIACRITIC and c.anchor_id is not None:
                    marks.setdefault(c.anchor_id, []).append(c)
            for c in word.components:
                if c.kind is ComponentKind.DIACRITIC:
                    continue
                members = [c] + marks.get(c.id, [])
                box = members[0].bbox
                for m in members[1:]:
                    box = box.union(m.bbox)
                out.append(Cluster(box, members, li, wi, bool(line.rejected)))
    return out


def _line_bands(result: PageResult) -> list[tuple[float, float, float]]:
    """(top, base, glyph height) per line, from its full-height components."""
    bands = []
    for line in result.lines + result.rejected_lines:
        comps = [c for w in line.words for c in w.components
                 if c.kind is ComponentKind.TEXT and c.anchor_id is None]
        if not comps:
            bands.append((float(line.bbox.y), float(line.bbox.y2), float(max(1, line.bbox.h))))
            continue
        h = np.array([c.bbox.h for c in comps], dtype=np.float64)
        g = float(np.median(h))
        tall = [c for c in comps if c.bbox.h >= 0.6 * g] or comps
        # Cap/ascender line: a low percentile of tops, so a mostly lower-case line still
        # finds its capitals' height. Baseline: median bottom of full-height glyphs
        # (descenders are the minority and do not move a median).
        top = float(np.percentile([c.bbox.y for c in tall], 15))
        base = float(np.median([c.bbox.y2 for c in tall]))
        bands.append((top, base, max(1.0, base - top)))
    return bands


def extract(gray: np.ndarray, result: PageResult, clusters: list[Cluster] | None = None
            ) -> tuple[np.ndarray, np.ndarray, list[Cluster]]:
    """``(crops uint8 [N,32,32], geometry float32 [N,5], clusters)`` for a page.

    ``gray`` is the original image in greyscale (the frame ``result`` boxes are in).
    """
    clusters = clusters_of(result) if clusters is None else clusters
    n = len(clusters)
    crops = np.full((n, CROP, CROP), 255, dtype=np.uint8)
    geom = np.zeros((n, N_GEOMETRY), dtype=np.float32)
    if not n:
        return crops, geom, clusters
    bands = _line_bands(result)
    H, W = gray.shape[:2]
    for i, cl in enumerate(clusters):
        b = cl.bbox
        top, base, g = bands[cl.line_index]
        # Line-height crop: the window spans the line's band (with room above for
        # stacked accents and below for descenders), not the glyph's own box, so the
        # *scale* is the line's — a lower-case 'c' fills less of it than a 'C'. A glyph
        # sticking out of the band (a misgrouped tall component) extends the window.
        y0f = min(top - ABOVE * g, b.y - 1)
        y1f = max(base + BELOW * g, b.y2 + 1)
        side = max(int(round(y1f - y0f)), b.w + 4)
        cx = b.x + b.w / 2.0
        x0, y0 = int(round(cx - side / 2)), int(round((y0f + y1f) / 2 - side / 2))
        x1, y1 = x0 + side, y0 + side
        patch = np.full((side, side), 255, dtype=np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = gray[sy0:sy1, sx0:sx1]
        interp = cv2.INTER_AREA if side > CROP else cv2.INTER_CUBIC
        crops[i] = cv2.resize(patch, (CROP, CROP), interpolation=interp)

        geom[i] = (b.h / g, b.w / g, (b.y - top) / g, (b.y2 - base) / g, b.w / max(1, b.h))
    return crops, geom, clusters


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
