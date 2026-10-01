from __future__ import annotations

import pytest

from ocr.ccl import label_components
from ocr.config import FilterConfig
from ocr.filters import apply_tier1, apply_tier2, filter_components
from ocr.types import BBox, Component, ComponentKind

CFG = FilterConfig()


def make(x=0, y=0, w=10, h=20, fill=0.5, cid=1) -> Component:
    return Component(
        id=cid,
        bbox=BBox(x, y, w, h),
        pixel_area=int(w * h * fill),
        centroid=(x + w / 2, y + h / 2),
        fill_ratio=fill,
    )


def route(components, page_w=1000, page_h=1000, cfg=CFG):
    filter_components(components, page_w, page_h, cfg)
    return components


class TestTier1:
    def test_keeps_a_normal_glyph(self):
        c = make(w=10, h=20, fill=0.5)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.TEXT

    def test_rejects_degenerate(self):
        c = make(w=1, h=1, fill=1.0)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.NOISE
        assert c.reason == "t1:degenerate"

    def test_long_solid_run_is_a_rule(self):
        c = make(w=800, h=3, fill=0.95)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.RULE

    def test_tall_thin_run_is_a_vertical_rule(self):
        c = make(w=3, h=400, fill=0.9)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.RULE

    def test_large_hollow_box_is_a_frame_not_a_photo(self):
        """A form's field box is structure. Calling it a photo would route it to the
        Phase 11 masking path, which is the wrong destination entirely."""
        c = make(w=700, h=700, fill=0.02)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.RULE
        assert c.reason == "t1:frame"

    def test_large_dense_region_is_a_blob(self):
        c = make(w=700, h=700, fill=0.85)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.BLOB

    def test_hollow_small_box_is_a_rule(self):
        c = make(w=40, h=30, fill=0.08)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.RULE
        assert c.reason == "t1:hollow"

    def test_wide_sparse_component_stays_text(self):
        """Several touching glyphs look elongated but are not a rule. Keeping them as
        TEXT is what lets the benchmark count the merge instead of hiding it."""
        c = make(w=200, h=14, fill=0.25)
        apply_tier1([c], 1000, 1000, CFG)
        assert c.kind is ComponentKind.TEXT


class TestTier2:
    def _page(self, n=20):
        return [make(x=i * 15, y=100, w=10, h=20, fill=0.5, cid=i) for i in range(n)]

    def test_skipped_when_too_few_components(self):
        few = [make(cid=i, x=i * 15) for i in range(3)]
        stats = apply_tier2(few, 1000, 1000, CFG)
        assert stats.relative_gates_applied is False
        assert all(c.kind is ComponentKind.TEXT for c in few)

    def test_median_height_drives_the_gates(self):
        page = self._page()
        stats = apply_tier2(page, 1000, 1000, CFG)
        assert stats.relative_gates_applied is True
        assert stats.median_height == pytest.approx(20.0)

    def test_oversized_relative_to_median_becomes_blob(self):
        page = self._page()
        tall = make(x=500, y=50, w=60, h=120, fill=0.6, cid=99)
        page.append(tall)
        apply_tier2(page, 1000, 1000, CFG)
        assert tall.kind is ComponentKind.BLOB

    def test_small_mark_with_a_parent_is_a_diacritic(self):
        """The dot of an i. Dropping it is an accuracy loss nothing downstream can
        undo, so it must be routed, not discarded."""
        page = self._page()
        dot = make(x=0, y=92, w=4, h=4, fill=1.0, cid=99)  # directly above glyph 0
        page.append(dot)
        apply_tier2(page, 1000, 1000, CFG)
        assert dot.kind is ComponentKind.DIACRITIC

    def test_small_mark_below_its_parent_is_also_a_diacritic(self):
        page = self._page()
        cedilla = make(x=0, y=122, w=4, h=4, fill=1.0, cid=99)
        page.append(cedilla)
        apply_tier2(page, 1000, 1000, CFG)
        assert cedilla.kind is ComponentKind.DIACRITIC

    def test_small_orphan_is_noise(self):
        page = self._page()
        speck = make(x=900, y=900, w=4, h=4, fill=1.0, cid=99)
        page.append(speck)
        apply_tier2(page, 1000, 1000, CFG)
        assert speck.kind is ComponentKind.NOISE
        assert speck.reason == "t2:small_orphan"

    def test_far_above_is_not_a_diacritic(self):
        page = self._page()
        far = make(x=0, y=20, w=4, h=4, fill=1.0, cid=99)  # 80 px above, median is 20
        page.append(far)
        apply_tier2(page, 1000, 1000, CFG)
        assert far.kind is ComponentKind.NOISE

    def test_relative_gates_can_be_disabled(self):
        page = self._page()
        tall = make(x=500, y=50, w=60, h=120, fill=0.6, cid=99)
        page.append(tall)
        apply_tier2(page, 1000, 1000, FilterConfig(use_relative_gates=False))
        assert tall.kind is ComponentKind.TEXT


class TestOnRealPixels:
    def test_document_like_image_routes_each_element(self, document_like):
        components = label_components(document_like)
        filter_components(components, 300, 200, CFG)
        kinds = {c.kind for c in components}

        assert ComponentKind.TEXT in kinds, "the 12 glyphs should survive"
        assert ComponentKind.RULE in kinds, "the long rule and hollow frame"
        assert ComponentKind.BLOB in kinds, "the 50x50 dense square"
        assert ComponentKind.NOISE in kinds, "the 1 px speck"

        glyphs = [c for c in components if c.kind is ComponentKind.TEXT]
        assert len(glyphs) == 12

    def test_filter_stats_report_every_kind(self, document_like):
        components = label_components(document_like)
        stats = filter_components(components, 300, 200, CFG)
        counts = stats.counts
        assert sum(counts.values()) == len(components)
        assert set(counts) == {k.value for k in ComponentKind}


def test_every_routed_component_records_a_reason(document_like):
    components = label_components(document_like)
    filter_components(components, 300, 200, CFG)
    for c in components:
        if c.kind is not ComponentKind.TEXT:
            assert c.reason, f"{c.kind} with no reason is undiagnosable in the overlay"
