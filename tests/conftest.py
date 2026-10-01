"""Shared fixtures: hand-built binary images with known component counts.

Fixtures are constructed arithmetically rather than rendered, so the expected answer is
a fact about the fixture rather than a value someone observed once and froze.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def blank() -> np.ndarray:
    """A 200x300 ink=0 canvas. Ink is 255 by the pipeline's convention."""
    return np.zeros((200, 300), dtype=np.uint8)


def put_rect(image: np.ndarray, x: int, y: int, w: int, h: int, filled: bool = True) -> None:
    if filled:
        image[y : y + h, x : x + w] = 255
    else:
        image[y : y + h, x : x + w] = 255
        image[y + 1 : y + h - 1, x + 1 : x + w - 1] = 0


@pytest.fixture
def three_blobs(blank: np.ndarray) -> np.ndarray:
    """Three separated 10x10 squares — the simplest possible CCL assertion."""
    for i in range(3):
        put_rect(blank, 20 + i * 40, 50, 10, 10)
    return blank


@pytest.fixture
def document_like(blank: np.ndarray) -> np.ndarray:
    """A page carrying one of each thing the filter must tell apart.

    Twelve glyph-sized marks (enough to pass ``min_components_for_stats``), a long thin
    rule, a hollow frame, a dense blob, a speck, and a dot positioned above a glyph.
    """
    for i in range(12):
        put_rect(blank, 10 + i * 12, 100, 6, 12)      # glyphs: 6x12
    put_rect(blank, 10, 150, 260, 2)                   # rule: aspect 130
    put_rect(blank, 10, 20, 90, 50, filled=False)      # frame: hollow
    put_rect(blank, 150, 20, 50, 50)                   # blob: dense 50x50
    put_rect(blank, 280, 180, 1, 1)                    # speck
    put_rect(blank, 10, 94, 3, 3)                      # dot above the first glyph
    return blank
