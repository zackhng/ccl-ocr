"""Grouping (Phase 5) on hand-built components whose right answer is geometry.

Components are constructed directly (processing coordinates), so each test states the
layout it is about rather than depending on thresholding.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import time
from dataclasses import replace

import pytest

from ocr.config import GroupConfig
from ocr.group import group_components, reading_order
from ocr.types import BBox, Component, ComponentKind, PageResult
from ocrbench.gt import GTBox, Sample
from ocrbench.metrics import grouping_metrics

H = 20  # glyph height
CFG = GroupConfig()
_ids = iter(range(1, 10**9))


def glyph(x, y, w=12, h=H, kind=ComponentKind.TEXT, anchor=None) -> Component:
    return Component(id=next(_ids), bbox=BBox(int(x), int(y), int(w), int(h)),
                     pixel_area=int(w * h * 0.5), centroid=(x + w / 2, y + h / 2),
                     fill_ratio=0.5, kind=kind, anchor_id=anchor)


def word(x, y, n, gap=2, w=12, h=H) -> list[Component]:
    return [glyph(x + i * (w + gap), y, w, h) for i in range(n)]


def run(comps, cfg=CFG, w=1200, h=600):
    """(lines, info): info has the GroupStats fields plus ``rejected`` lines."""
    lines, rejected, stats = group_components(comps, w, h, float(H), cfg)
    return lines, SimpleNamespace(**stats.as_dict(), rejected=rejected)


def texts(lines):
    """Each line as a list of word lengths, for compact assertions."""
    return [[len(w.components) for w in ln.words] for ln in lines]


class TestLines:
    def test_two_rows_are_two_lines(self):
        lines, _ = run(word(20, 20, 5) + word(20, 80, 4))
        assert texts(lines) == [[5], [4]]

    def test_column_gap_splits_a_row_into_segments(self):
        """ID card: 'RACE' and 'SEX' share a row but are separate fields."""
        lines, _ = run(word(20, 20, 4) + word(20 + 4 * 14 + 5 * H, 20, 3))
        assert texts(lines) == [[4], [3]]

    def test_rotated_row_is_still_one_line(self):
        comps = []
        x = 20.0
        for i in range(14):
            comps.append(glyph(x, 200 + math.tan(math.radians(4)) * (x - 20), 12, H))
            x += 14
        lines, _ = run(comps)
        assert len(lines) == 1

    def test_descender_does_not_bridge_to_the_line_below(self):
        top = word(20, 20, 6)
        top.append(glyph(20 + 6 * 14, 20, 12, int(1.45 * H)))  # a 'p' reaching down
        below = word(20, 20 + int(1.5 * H), 6)
        lines, _ = run(top + below)
        assert len(lines) == 2

    def test_disabled_emits_nothing(self):
        lines, _ = run(word(20, 20, 5), replace(CFG, enabled=False))
        assert lines == []


class TestWords:
    def test_space_splits_words_kerning_does_not(self):
        comps = word(20, 20, 4, gap=2) + word(20 + 4 * 14 + int(0.5 * H), 20, 3, gap=2)
        lines, _ = run(comps)
        assert texts(lines) == [[4, 3]]

    def test_monospace_spacing_does_not_split_words(self):
        """Receipts: letter gaps of 0.35 h would cross a fixed 0.25 h threshold."""
        g = int(0.35 * H)
        comps = word(20, 20, 6, gap=g) + word(20 + 6 * (12 + g) + int(1.0 * H), 20, 5, gap=g)
        lines, _ = run(comps)
        assert texts(lines) == [[6, 5]]
        fixed, _ = run(comps, replace(CFG, word_gap_median_factor=0.0))
        assert len(fixed[0].words) > 2  # what the adaptive threshold prevents

    def test_unspaced_han_row_is_one_word(self):
        lines, _ = run(word(20, 20, 8, gap=2, w=H, h=H))
        assert texts(lines) == [[8]]


class TestSmallMarks:
    def test_dot_and_full_stop_join_their_anchors_word(self):
        comps = word(20, 40, 3)
        dot = glyph(20 + 14 + 4, 32, 4, 4, ComponentKind.DIACRITIC, anchor=comps[1].id)
        stop = glyph(20 + 3 * 14, 40 + H - 4, 4, 4, ComponentKind.TEXT, anchor=comps[2].id)
        lines, _ = run(comps + [dot, stop])
        assert len(lines) == 1 and len(lines[0].words) == 1
        members = lines[0].words[0].components
        assert dot in members and stop in members
        assert lines[0].words[0].bbox.y == 32  # the dot grew the word box

    def test_marks_never_start_a_line(self):
        orphan_anchor = glyph(500, 300, 4, 4, ComponentKind.DIACRITIC, anchor=999_999)
        lines, _ = run(word(20, 20, 4) + [orphan_anchor])
        assert len(lines) == 1


class TestJunk:
    def test_speck_lines_are_gated_but_components_kept(self):
        comps = word(20, 20, 5)
        specks = [glyph(600 + i * 8, 400, 5, 5) for i in range(4)]  # a row of specks
        lone = glyph(900, 500, 8, 10)                               # one small blob
        all_comps = comps + specks + [lone]
        lines, stats = run(all_comps)
        assert texts(lines) == [[5]]
        assert stats.junk_lines == 2
        # Routed, not deleted: both junk lines are kept aside, with the gate's name.
        # The 4-speck row is longer than junk_small_max_count, so the median-height gate
        # catches it; the lone blob is caught by the first gate that applies, "specks".
        assert sorted(ln.rejected for ln in stats.rejected) == ["small_median", "specks"]
        kept = {id(c) for ln in stats.rejected for w in ln.words for c in w.components}
        assert {id(c) for c in specks + [lone]} <= kept

    def test_single_full_height_field_is_kept(self):
        """The 'M' in the ID card's SEX field is a one-glyph line and must survive."""
        lines, _ = run(word(20, 20, 4) + [glyph(400, 200, 14, H)])
        assert len(lines) == 2

    def test_barcode_is_gated(self):
        bars = [glyph(100 + i * 5, 300, 2, int(1.8 * H)) for i in range(20)]
        lines, _ = run(word(20, 20, 4) + bars)
        assert len(lines) == 1

    def test_run_of_digit_ones_is_not_a_barcode(self):
        ones = [glyph(100 + i * 9, 300, 6, H) for i in range(8)]  # w/h 0.3
        lines, _ = run(ones)
        assert len(lines) == 1


class TestReadingOrder:
    def test_rows_then_columns(self):
        boxes = [BBox(300, 100, 50, 20), BBox(10, 101, 50, 20), BBox(10, 10, 50, 20)]
        assert reading_order(boxes) == [2, 1, 0]

    def test_lines_come_out_in_reading_order(self):
        comps = word(400, 20, 3) + word(20, 21, 3) + word(20, 80, 3)
        lines, _ = run(comps)
        assert [ln.bbox.x for ln in lines] == [20, 400, 20]


def test_cost_is_bounded():
    comps = []
    for row in range(60):
        for col in range(80):
            comps.append(glyph(10 + col * 14, 10 + row * 30, 12, H))
    start = time.perf_counter()
    lines, _ = run(comps, w=1200, h=1900)
    assert time.perf_counter() - start < 0.1
    assert len(lines) == 60


class TestGroupingMetrics:
    def sample(self, lines, words, source="synth"):
        return Sample(sample_id="t", source=source, capture="scan", script="latin", dpi=300,
                      lines=[GTBox(BBox(*b)) for b in lines], words=[GTBox(BBox(*b)) for b in words])

    def page(self, comps):
        lines, _ = run(comps)
        return PageResult(width=1200, height=600, components=comps, lines=lines)

    def test_perfect_grouping(self):
        comps = word(20, 20, 4) + word(20 + 4 * 14 + int(0.5 * H), 20, 3)
        w1, w2 = (20, 20, 4 * 14 - 2, H), (20 + 4 * 14 + 10, 20, 3 * 14 - 2, H)
        m = grouping_metrics(self.sample([(20, 20, w2[0] + w2[2] - 20, H)], [w1, w2]), self.page(comps))
        assert m.line_prf == (1.0, 1.0, 1.0)
        assert m.word_prf[2] == pytest.approx(1.0)
        assert m.word_splits == m.word_merges == 0

    def test_merge_and_split_are_counted(self):
        # One predicted word spanning two equal GT words: IoU with each is < 0.5.
        comps = word(20, 20, 8)
        gt_words = [(20, 20, 4 * 14 - 2, H), (20 + 4 * 14, 20, 4 * 14 - 2, H)]
        m = grouping_metrics(self.sample([], gt_words), self.page(comps))
        assert m.word_merges == 1 and m.tp_words == 0

    def test_entity_lines_are_skipped(self):
        m = grouping_metrics(self.sample([(0, 0, 10, 10)], [], source="funsd"), self.page(word(20, 20, 3)))
        assert m.n_gt_lines == 0

    def test_no_lines_is_not_applicable(self):
        empty = PageResult(width=10, height=10)
        assert grouping_metrics(self.sample([], []), empty) is None
