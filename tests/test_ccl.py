from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr.binarize import binarize, recover_inverted_regions
from ocr.ccl import label_components
from ocr.config import DEFAULT_CONFIG, BinarizeConfig, PreprocessConfig
from ocr.engine import CCLEngine
from ocr.preprocess import compute_scale, detect_polarity, normalize_illumination, preprocess
from ocr.types import BBox


def test_labels_each_separated_blob(three_blobs):
    components = label_components(three_blobs, connectivity=8)
    assert len(components) == 3
    for c in components:
        assert c.bbox.w == 10 and c.bbox.h == 10
        assert c.pixel_area == 100
        assert c.fill_ratio == pytest.approx(1.0)


def test_background_label_is_excluded(blank):
    assert label_components(blank) == []


def test_fill_ratio_distinguishes_hollow_from_solid(blank):
    blank[10:60, 10:60] = 255
    blank[11:59, 11:59] = 0  # 50x50 frame, 1 px wide
    blank[100:150, 100:150] = 255  # 50x50 solid

    components = sorted(label_components(blank), key=lambda c: c.bbox.x)
    frame, solid = components[0], components[1]
    assert frame.bbox.area == solid.bbox.area  # identical boxes...
    assert frame.fill_ratio < 0.1               # ...wholly different content
    assert solid.fill_ratio == pytest.approx(1.0)


def test_connectivity_affects_diagonal_touching(blank):
    blank[10, 10] = 255
    blank[11, 11] = 255
    assert len(label_components(blank, connectivity=8)) == 1
    assert len(label_components(blank, connectivity=4)) == 2


class TestPreprocess:
    def test_scale_caps_long_side(self):
        cfg = PreprocessConfig(long_side_cap=1000, upscale_small=False)
        assert compute_scale(500, 2000, cfg) == pytest.approx(0.5)

    def test_scale_floors_small_images(self):
        cfg = PreprocessConfig(long_side_cap=1600, upscale_small=True, min_long_side=1000)
        assert compute_scale(200, 500, cfg) == pytest.approx(2.0)

    def test_scale_is_identity_in_range(self):
        cfg = PreprocessConfig(long_side_cap=1600, upscale_small=True, min_long_side=1000)
        assert compute_scale(900, 1200, cfg) == 1.0

    def test_detects_light_on_dark(self):
        dark = np.full((100, 100), 20, dtype=np.uint8)
        dark[40:60, 40:60] = 230
        assert detect_polarity(dark) is True

        light = np.full((100, 100), 230, dtype=np.uint8)
        light[40:60, 40:60] = 20
        assert detect_polarity(light) is False

    def test_illumination_normalisation_does_not_blank_a_white_page(self):
        """Regression: `background + 1` wraps 255 to 0 in uint8 and zeroed everything."""
        page = np.full((120, 120), 255, dtype=np.uint8)
        page[50:70, 50:70] = 0
        out = normalize_illumination(page, 31)
        assert out.max() > 200
        assert out.min() < 60

    def test_illumination_normalisation_flattens_a_gradient(self):
        yy = np.linspace(90, 255, 160, dtype=np.uint8)
        page = np.tile(yy.reshape(-1, 1), (1, 160))
        page[20:30, 20:30] = 0
        page[130:140, 130:140] = 0
        out = normalize_illumination(page, 31)
        # Both marks should end up similarly dark despite sitting in different lighting.
        assert abs(int(out[20:30, 20:30].mean()) - int(out[130:140, 130:140].mean())) < 40

    def test_boxes_map_back_to_original_coordinates(self):
        image = np.full((400, 2400, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (1000, 150), (1060, 230), (0, 0, 0), -1)
        result = CCLEngine().run(image)

        assert result.width == 2400 and result.height == 400
        assert result.scale < 1.0  # it was downscaled internally
        found = [c for c in result.components if c.bbox.w > 30]
        assert found, "the rectangle should survive as a component"
        box = found[0].bbox
        assert box.x == pytest.approx(1000, abs=12)
        assert box.y == pytest.approx(150, abs=12)

    def test_fill_ratio_never_exceeds_one_after_rescaling(self):
        """Regression: rescaling pixel_area and bbox independently could push the
        derived ratio above 1.0, which is geometrically impossible."""
        rng = np.random.default_rng(0)
        image = np.full((300, 2000, 3), 240, dtype=np.uint8)
        for _ in range(400):
            x, y = int(rng.integers(0, 1990)), int(rng.integers(0, 290))
            image[y : y + int(rng.integers(1, 6)), x : x + int(rng.integers(1, 6))] = 10
        result = CCLEngine().run(image)
        assert result.components
        assert all(0.0 <= c.fill_ratio <= 1.0 for c in result.components)


class TestBinarize:
    def test_ink_is_255(self):
        page = np.full((120, 120), 240, dtype=np.uint8)
        page[50:70, 50:70] = 10
        binary, _ = binarize(page, BinarizeConfig(recover_inverted_regions=False))
        assert binary[60, 60] == 255
        assert binary[5, 5] == 0

    def test_sauvola_and_adaptive_both_find_the_mark(self):
        page = np.full((120, 120), 240, dtype=np.uint8)
        page[50:70, 50:70] = 10
        for method in ("adaptive_gaussian", "sauvola", "otsu"):
            cfg = BinarizeConfig(method=method, recover_inverted_regions=False)
            binary, _ = binarize(page, cfg)
            assert binary[60, 60] == 255, method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown binarize method"):
            binarize(np.zeros((10, 10), np.uint8), BinarizeConfig(method="nope"))

    def test_recovers_text_inside_an_inverted_banner(self):
        """A dark banner with light text must not binarise into one solid blob.

        This is the ID-card header case; a character lost here is lost for good.
        """
        page = np.full((200, 400), 245, dtype=np.uint8)
        page[20:80, 20:380] = 30                      # dark banner
        for i in range(8):                            # light glyphs inside it
            page[35:65, 40 + i * 40 : 40 + i * 40 + 18] = 240

        cfg = BinarizeConfig(recover_inverted_regions=False)
        without, _ = binarize(page, cfg)
        with_recovery, n = binarize(page, BinarizeConfig(recover_inverted_regions=True))

        banner_before = label_components(without[20:80, 20:380])
        banner_after = label_components(with_recovery[20:80, 20:380])
        assert n >= 1
        assert len(banner_after) > len(banner_before)

    def test_recovery_leaves_a_clean_page_alone(self):
        page = np.full((200, 300), 245, dtype=np.uint8)
        for i in range(10):
            page[100:118, 20 + i * 25 : 20 + i * 25 + 10] = 20
        plain, _ = binarize(page, BinarizeConfig(recover_inverted_regions=False))
        recovered, n = recover_inverted_regions(page, plain, BinarizeConfig())
        assert n == 0
        assert np.array_equal(plain, recovered)


def test_bbox_geometry():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 0, 10, 10)
    assert a.intersection_area(b) == 50
    assert a.iou(b) == pytest.approx(50 / 150)
    assert a.coverage_of(b) == pytest.approx(0.5)
    assert a.union(b) == BBox(0, 0, 15, 10)
    assert a.intersection_area(BBox(100, 100, 5, 5)) == 0
    assert BBox.from_points([(3, 7), (11, 2), (9, 20)]) == BBox(3, 2, 8, 18)


def test_bbox_scaled_keeps_at_least_one_pixel():
    assert BBox(10, 10, 1, 1).scaled(0.01) == BBox(0, 0, 1, 1)


def test_engine_reports_every_stage():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.putText(image, "AB", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    result = CCLEngine(DEFAULT_CONFIG).run(image)
    for stage in ("preprocess", "binarize", "ccl", "filter", "total"):
        assert stage in result.timings_ms
    assert result.timings_ms["total"] > 0
