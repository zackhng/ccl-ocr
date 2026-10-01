"""PaddleOCR baseline wrapper.

The raw-result conversion is tested against hand-built Paddle results, so it runs
without Paddle installed. The end-to-end test needs Paddle and its models and is
skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr.engine import Engine
from ocr.paddle_engine import page_from_paddle


def quad(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int16)


class TestConversion:
    def test_full_result_becomes_lines_with_text(self):
        raw = {"rec_polys": [quad(10, 50, 100, 20), quad(10, 10, 80, 20)],
               "rec_texts": ["SECOND", "FIRST"]}
        page = page_from_paddle(raw, 200, 100, {"paddle_ocr": 12.0})
        assert [ln.text for ln in page.lines] == ["FIRST", "SECOND"]  # reading order
        assert page.lines[0].bbox.as_list() == [10, 10, 80, 20]
        assert page.components == []
        assert page.timings_ms["total"] == pytest.approx(12.0)

    def test_detection_only_result_has_boxes_without_text(self):
        raw = {"dt_polys": np.stack([quad(5, 5, 50, 10)]), "dt_scores": [0.9]}
        page = page_from_paddle(raw, 100, 100)
        assert len(page.lines) == 1
        assert page.lines[0].text == ""

    def test_empty_ndarray_result(self):
        """dt_polys arrives as an ndarray; `polys or []` would raise on it."""
        page = page_from_paddle({"dt_polys": np.zeros((0, 4, 2))}, 100, 100)
        assert page.lines == []

    def test_polygons_are_clipped_to_the_image(self):
        raw = {"rec_polys": [quad(-4, -3, 60, 20)], "rec_texts": ["EDGE"]}
        box = page_from_paddle(raw, 50, 50).lines[0].bbox
        assert box.x >= 0 and box.y >= 0 and box.x2 <= 50

    def test_text_count_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            page_from_paddle({"rec_polys": [quad(0, 0, 5, 5)], "rec_texts": []}, 10, 10)


@pytest.fixture(scope="module")
def paddle_engine():
    pytest.importorskip("paddleocr")
    from ocr.paddle_engine import PaddleConfig, PaddleEngine

    try:
        return PaddleEngine(PaddleConfig(cpu_threads=2))
    except Exception as exc:  # model download blocked, etc.
        pytest.skip(f"PaddleOCR unavailable: {exc}")


def test_paddle_end_to_end(paddle_engine):
    import cv2

    img = np.full((120, 600, 3), 255, np.uint8)
    cv2.putText(img, "ACCOUNT 12345", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    page = paddle_engine.run(img)
    assert isinstance(paddle_engine, Engine)
    assert len(page.lines) == 1
    assert "12345" in page.lines[0].text
    box = page.lines[0].bbox
    assert 0 <= box.x < 60 and 20 <= box.y < 60  # original-image coordinates
    assert page.timings_ms["total"] > 0
