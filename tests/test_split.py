"""Merge splitter (Phase 2b), on hand-built pages whose right answer is geometry.

Each page carries a row of ordinary glyphs (enough for the scale estimator to measure
the page) plus the case under test. Images are grayscale: dark ink on a light page,
the form the splitter receives after preprocessing.
"""

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from ocr.ccl import _components_from_stats, label_components, label_with_maps
from ocr.config import DEFAULT_CONFIG, SplitConfig
from ocr.engine import CCLEngine
from ocr.filters import (
    apply_tier1,
    ink_weighted_median_height,
    ink_weighted_median_height_arrays,
    tier1_text_mask,
)
from ocr.split import cut_columns, page_glyph_scale, select_suspects, split_merged
from ocr.types import BBox, ComponentKind

GLYPH_H, GLYPH_W = 20, 10
INK, SEAM, PAPER = 20, 150, 235

FILTERS = DEFAULT_CONFIG.filters


def page(w: int = 600, h: int = 200) -> np.ndarray:
    gray = np.full((h, w), PAPER, np.uint8)
    # Fifteen ordinary, well-separated glyphs: the page's scale reference.
    for i in range(15):
        gray[20 : 20 + GLYPH_H, 20 + i * 25 : 20 + i * 25 + GLYPH_W] = INK
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    # The production threshold: the large window that merges touching glyphs.
    b = DEFAULT_CONFIG.binarize
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, b.block_size | 1, b.c)


def run_split(gray: np.ndarray, cfg: SplitConfig):
    """Split, and return (components after the swap, SplitResult)."""
    binary = binarize(gray)
    comps, labels, stats = label_with_maps(binary)
    result = split_merged(gray, binary, labels, stats, cfg, FILTERS)
    return result.apply(comps), result


def count_in(components, x0, y0, x1, y1, min_h=10) -> int:
    return sum(1 for c in components
               if x0 <= c.bbox.cx <= x1 and y0 <= c.bbox.cy <= y1 and c.bbox.h >= min_h)


COLUMNS = SplitConfig(split_touching_columns=True)


class TestScale:
    def test_measures_the_page_glyphs(self):
        gray = page()
        _, _, stats = label_with_maps(binarize(gray))
        h, w = page_glyph_scale(stats, gray.shape[1], gray.shape[0], FILTERS)
        assert h == pytest.approx(GLYPH_H, abs=2)
        assert w == pytest.approx(GLYPH_W, abs=2)

    def test_too_few_glyphs_is_no_scale(self):
        gray = np.full((100, 100), PAPER, np.uint8)
        gray[10:30, 10:20] = INK
        _, _, stats = label_with_maps(binarize(gray))
        assert page_glyph_scale(stats, 100, 100, FILTERS) is None

    def test_vectorised_tier1_matches_apply_tier1(self):
        """The splitter's array version must make exactly the filter's decisions, or
        the two would disagree about what one glyph is."""
        rng = np.random.default_rng(3)
        n = 3000
        w = rng.integers(1, 700, n)
        h = rng.integers(1, 700, n)
        # A mix of shapes: specks, thin rules, and everything in between.
        w[:500], h[:500] = rng.integers(1, 4, 500), rng.integers(1, 4, 500)
        w[500:1000], h[500:1000] = rng.integers(200, 900, 500), rng.integers(1, 6, 500)
        area = np.maximum(1, (w * h * rng.uniform(0.05, 1.0, n)).astype(int))
        stats = np.zeros((n + 1, 5), np.int32)
        stats[1:, 2], stats[1:, 3], stats[1:, 4] = w, h, area
        comps = _components_from_stats(stats, np.zeros((n + 1, 2)))
        apply_tier1(comps, 1000, 1000, FILTERS)
        expected = np.array([False] + [c.kind is ComponentKind.TEXT for c in comps])
        np.testing.assert_array_equal(tier1_text_mask(stats, 1000, 1000, FILTERS), expected)

    def test_array_estimator_matches_component_estimator(self):
        rng = np.random.default_rng(5)
        gray = page()
        gray[rng.integers(0, 200, 400), rng.integers(0, 600, 400)] = INK  # specks
        comps, _, stats = label_with_maps(binarize(gray))
        from ocr.filters import glyph_height_estimate

        st = stats[1:]
        assert ink_weighted_median_height(comps) == glyph_height_estimate(
            st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
        )


class TestSuspects:
    def test_only_wide_glyph_height_components(self):
        stats = np.array([
            [0, 0, 600, 200, 0],   # background
            [0, 0, 10, 20, 150],   # one glyph
            [0, 0, 30, 20, 400],   # three glyphs wide: suspect
            [0, 0, 300, 3, 900],   # underline: too short
            [0, 0, 30, 90, 2000],  # heading/blob: too tall
        ], dtype=np.int32)
        got = set(select_suspects(stats, GLYPH_H, GLYPH_W, SplitConfig()).tolist())
        assert 2 in got and 1 not in got and 3 not in got and 4 not in got

    def test_square_glyph_on_a_square_glyph_page_is_not_a_suspect(self):
        """Han: glyph width ~ glyph height, so the page's own median width is ~20 and a
        single square glyph is nowhere near 1.3x it."""
        stats = np.array([[0, 0, 600, 200, 0], [0, 0, 20, 20, 300]], dtype=np.int32)
        assert len(select_suspects(stats, 20.0, 20.0, SplitConfig())) == 0


class TestRethreshold:
    def test_grey_seam_between_glyphs_is_resolved(self):
        """Two glyphs whose gap is a one-pixel grey ridge — what blur leaves behind.
        The large window fills it; the local re-threshold opens it."""
        gray = page()
        y, x = 100, 100
        gray[y : y + GLYPH_H, x : x + GLYPH_W] = INK
        gray[y : y + GLYPH_H, x + GLYPH_W] = SEAM
        gray[y : y + GLYPH_H, x + GLYPH_W + 1 : x + 2 * GLYPH_W + 1] = INK
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)

        before, _, _ = label_with_maps(binarize(gray))
        assert count_in(before, x, y, x + 25, y + GLYPH_H) == 1, "fixture must start merged"

        after, result = run_split(gray, SplitConfig())
        assert result.stats.rethresholded >= 1
        assert count_in(after, x, y, x + 25, y + GLYPH_H) == 2
        assert len(result.added) == 2 and all(c.split for c in result.added)

    def test_rethreshold_never_accepts_fragments(self):
        """A wide solid glyph-height block: a fine threshold only hollows it out, so
        no whole-height parts appear and the suspect must be left exactly as it was."""
        gray = page()
        gray[100:120, 100:140] = INK
        _, result = run_split(gray, SplitConfig())
        assert result.stats.rethresholded == 0
        assert not result.removed


class TestColumnCut:
    def bridged_pair(self):
        """Two glyphs joined by a thin bridge of real ink — no seam to re-threshold."""
        gray = page()
        y, x = 100, 100
        gray[y : y + GLYPH_H, x : x + GLYPH_W] = INK
        gray[y + 9 : y + 11, x + GLYPH_W : x + GLYPH_W + 3] = INK
        gray[y : y + GLYPH_H, x + GLYPH_W + 3 : x + 2 * GLYPH_W + 3] = INK
        return gray, x, y

    def test_bridge_is_cut_only_when_enabled(self):
        gray, x, y = self.bridged_pair()
        off, _ = run_split(gray, SplitConfig())
        assert count_in(off, x, y, x + 25, y + GLYPH_H) == 1
        on, result = run_split(gray, COLUMNS)
        assert result.stats.column_cuts >= 1
        assert count_in(on, x, y, x + 25, y + GLYPH_H) == 2

    def test_wide_glyph_without_a_valley_is_not_cut(self):
        """An 'M'-like glyph: wide, but every column carries ink well above the cut
        threshold, so there is nowhere legitimate to cut."""
        region = np.zeros((20, 16), np.uint8)
        region[:, :3] = 255
        region[:, 13:] = 255
        region[:12, :] = 255  # heavy top bar: every column at >= 60% of the peak
        mask = region > 0
        assert cut_columns(region, mask, GLYPH_W, COLUMNS) == 0

    def test_pieces_never_narrower_than_min_width(self):
        region = np.full((20, 30), 255, np.uint8)
        region[:, 1] = 0  # a valley one pixel from the edge
        mask = np.ones_like(region, dtype=bool)
        cut_columns(region, mask, GLYPH_W, COLUMNS)
        assert region[:, :5].any()  # no sliver was cut off at column 1


class TestEngine:
    def test_disabled_is_identical_to_phase_0_2(self):
        gray = page()
        gray[100:120, 100:130] = INK
        off = replace(DEFAULT_CONFIG, split=replace(DEFAULT_CONFIG.split, enabled=False))
        a = CCLEngine(off).run(gray)
        b = CCLEngine(DEFAULT_CONFIG).run(gray)  # a solid block: suspect, not split
        assert "split" not in a.timings_ms
        assert not any(c.split for c in a.components)
        assert [c.bbox for c in a.components] == [c.bbox for c in b.components]

    def test_engine_swaps_in_the_splitters_pieces(self, monkeypatch):
        """Wiring, isolated from the splitting heuristics: the suspect is replaced by
        the pieces the splitter returns, and those go on through the filter."""
        import ocr.engine as engine_mod
        from ocr.split import SplitResult, SplitStats

        def fake_split(gray, binary, labels, stats, cfg, filter_cfg, connectivity=8, timer=None):
            widest = int(np.argmax(stats[1:, 2])) + 1
            x, y, w, h = (int(v) for v in stats[widest, :4])
            out = binary.copy()
            out[y : y + h, x + w // 2] = 0
            pieces, _, _ = label_with_maps(out[y : y + h, x : x + w])
            for i, c in enumerate(pieces):
                c.id, c.split = 10_000 + i, True
                c.bbox = BBox(c.bbox.x + x, c.bbox.y + y, c.bbox.w, c.bbox.h)
            return SplitResult(binary=out, removed={widest}, added=pieces,
                               stats=SplitStats(rethresholded=1))

        monkeypatch.setattr(engine_mod, "split_merged", fake_split)
        gray = page(600, 400)
        gray[250:290, 200:280] = INK  # one wide block in the lower half
        result = CCLEngine(DEFAULT_CONFIG).run(gray)
        assert sum(c.split for c in result.components) == 2
        assert result.meta["split"]["rethresholded"] == 1

    def test_local_relabel_equals_full_relabel(self):
        """Pieces are labelled inside the suspect's box only. That is exact because
        splitting only removes ink from inside one component; check it on a page with
        many merged pairs."""
        gray = page(800, 400)
        for k in range(12):
            y, x = 100 + (k // 6) * 60, 40 + (k % 6) * 120
            gray[y : y + GLYPH_H, x : x + GLYPH_W] = INK
            gray[y : y + GLYPH_H, x + GLYPH_W] = SEAM
            gray[y : y + GLYPH_H, x + GLYPH_W + 1 : x + 2 * GLYPH_W + 1] = INK
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
        after, result = run_split(gray, COLUMNS)
        assert result.removed, "fixture must exercise the splitter"
        full = label_components(result.binary)
        assert sorted(c.bbox.as_list() for c in after) == sorted(c.bbox.as_list() for c in full)
        assert sorted(c.pixel_area for c in after) == sorted(c.pixel_area for c in full)

    def test_cost_does_not_scale_with_speck_count(self):
        """Suspect selection is vectorised: 40k specks must not mean 40k Python
        iterations."""
        rng = np.random.default_rng(0)
        gray = page(1600, 1000)
        ys, xs = rng.integers(0, 1000, 40000), rng.integers(0, 1600, 40000)
        gray[ys, xs] = INK
        binary = binarize(gray)
        comps, labels, stats = label_with_maps(binary)
        assert len(comps) > 20000
        start = time.perf_counter()
        split_merged(gray, binary, labels, stats, SplitConfig(), FILTERS)
        assert time.perf_counter() - start < 0.25

    def test_suspects_are_capped_widest_first(self):
        stats = np.array([[0, 0, 900, 900, 0]] + [[0, 0, 15 + i, 20, 200] for i in range(30)],
                         dtype=np.int32)
        order = select_suspects(stats, GLYPH_H, GLYPH_W, SplitConfig())
        widths = list(stats[order, 2])
        assert widths == sorted(widths, reverse=True)
