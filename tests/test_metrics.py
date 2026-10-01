"""Metric tests on fixtures where the right answer is arithmetic, not observed.

Each case constructs a situation whose outcome follows from the definition, so a
regression in the metric is distinguishable from a regression in the pipeline.
"""

from __future__ import annotations

import pytest

from ocr.types import BBox, Component, ComponentKind, PageResult
from ocrbench.gt import GTBox, GTChar, Sample
from ocrbench.metrics import BoxIndex, char_metrics, evaluate, word_metrics
from ocrbench.runner import aggregate_chars, aggregate_words, percentiles


def comp(x, y, w, h, kind=ComponentKind.TEXT, cid=1) -> Component:
    return Component(
        id=cid, bbox=BBox(x, y, w, h), pixel_area=w * h // 2,
        centroid=(x + w / 2, y + h / 2), fill_ratio=0.5, kind=kind,
    )


def page(components, w=500, h=200) -> PageResult:
    return PageResult(width=w, height=h, components=components, timings_ms={"total": 1.0})


def sample(chars=None, words=None, source="synth", **kw) -> Sample:
    return Sample(
        sample_id="t", source=source, capture="scan", script="latin", dpi=300,
        chars=chars, words=words or [], **kw,
    )


class TestCharOutcomes:
    def test_perfect_isolation(self):
        gt = [GTChar(BBox(i * 20, 50, 10, 20), "A") for i in range(5)]
        comps = [comp(i * 20, 50, 10, 20, cid=i) for i in range(5)]
        m = char_metrics(sample(chars=gt), page(comps))
        assert m.n_gt == 5
        assert m.isolated == 5
        assert m.isolation_recall == pytest.approx(1.0)
        assert m.merged == m.over_segmented == m.missed == 0

    def test_one_component_spanning_two_characters_is_a_merge(self):
        gt = [GTChar(BBox(0, 50, 10, 20), "A"), GTChar(BBox(10, 50, 10, 20), "B")]
        m = char_metrics(sample(chars=gt), page([comp(0, 50, 20, 20)]))
        assert m.merged == 2
        assert m.isolated == 0

    def test_two_components_inside_one_character_is_over_segmentation(self):
        gt = [GTChar(BBox(0, 50, 10, 20), "E")]
        comps = [comp(0, 50, 10, 9, cid=1), comp(0, 61, 10, 9, cid=2)]
        m = char_metrics(sample(chars=gt), page(comps))
        assert m.over_segmented == 1
        assert m.isolated == 0

    def test_filtered_character_counts_as_missed(self):
        gt = [GTChar(BBox(0, 50, 10, 20), "A")]
        m = char_metrics(sample(chars=gt), page([comp(0, 50, 10, 20, kind=ComponentKind.NOISE)]))
        assert m.missed == 1
        assert m.n_surviving == 0

    def test_diacritic_kind_counts_as_surviving(self):
        """A DIACRITIC is routed, not discarded -- Phase 5 merges it back."""
        gt = [GTChar(BBox(0, 50, 6, 6), ".")]
        m = char_metrics(sample(chars=gt), page([comp(0, 50, 6, 6, kind=ComponentKind.DIACRITIC)]))
        assert m.isolated == 1

    def test_junk_counts_components_on_no_character(self):
        gt = [GTChar(BBox(0, 50, 10, 20), "A")]
        comps = [comp(0, 50, 10, 20, cid=1), comp(400, 10, 12, 12, cid=2)]
        m = char_metrics(sample(chars=gt), page(comps))
        assert m.junk == 1
        assert m.junk_rate == pytest.approx(0.5)

    def test_junk_rate_is_undefined_when_gt_is_incomplete(self):
        """A component on real-but-unannotated text is not junk. Reporting it as junk
        would punish the pipeline for the dataset's gaps."""
        gt = [GTChar(BBox(0, 50, 10, 20), "A")]
        s = sample(chars=gt, source="funsd")
        s.meta["gt_complete"] = False
        m = char_metrics(s, page([comp(400, 10, 12, 12)]))
        assert m.junk_rate is None

    def test_junk_inside_a_photo_region_is_tracked_separately(self):
        gt = [GTChar(BBox(0, 50, 10, 20), "A")]
        s = sample(chars=gt)
        s.regions_nontext = [GTBox(BBox(300, 0, 100, 100), type="photo")]
        comps = [comp(0, 50, 10, 20, cid=1), comp(320, 20, 12, 12, cid=2)]
        m = char_metrics(s, page(comps))
        assert m.junk == 1
        assert m.junk_in_nontext == 1

    def test_small_marks_are_tracked_separately(self):
        gt = [GTChar(BBox(i * 20, 50, 10, 20), "A") for i in range(5)]
        gt.append(GTChar(BBox(200, 64, 4, 4), "."))  # 4 px vs median 20 -> "small"
        m = char_metrics(sample(chars=gt), page([comp(i * 20, 50, 10, 20, cid=i) for i in range(5)]))
        assert m.small_gt == 1
        assert m.small_retained == 0
        assert m.small_retention == pytest.approx(0.0)

    def test_small_retention_is_none_when_there_are_no_small_marks(self):
        gt = [GTChar(BBox(i * 20, 50, 10, 20), "A") for i in range(5)]
        m = char_metrics(sample(chars=gt), page([]))
        assert m.small_retention is None

    def test_empty_ground_truth_is_safe(self):
        m = char_metrics(sample(chars=[]), page([comp(0, 0, 10, 10)]))
        assert m.n_gt == 0
        assert m.isolation_recall == 0.0


class TestRegionMetrics:
    def test_word_granularity_when_words_exist(self):
        s = sample(words=[GTBox(BBox(0, 50, 40, 20), "HI")])
        m = word_metrics(s, page([comp(0, 50, 40, 20)]))
        assert m.granularity == "word"
        assert m.hit == 1
        assert m.mean_coverage == pytest.approx(1.0)

    def test_falls_back_to_lines_when_no_words(self):
        """SROIE annotates lines only. Skipping those samples would drop an entire real
        dataset out of the headline without anyone noticing."""
        s = sample()
        s.lines = [GTBox(BBox(0, 50, 100, 20), "A LINE")]
        m = word_metrics(s, page([comp(0, 50, 50, 20)]))
        assert m.granularity == "line"
        assert m.n_gt == 1
        assert m.hit == 1
        assert m.mean_coverage == pytest.approx(0.5)

    def test_component_straddling_two_words_is_crossing(self):
        s = sample(words=[GTBox(BBox(0, 50, 40, 20), "AA"), GTBox(BBox(60, 50, 40, 20), "BB")])
        m = word_metrics(s, page([comp(30, 50, 50, 20)]))
        assert m.crossing >= 1

    def test_coverage_does_not_double_count_overlapping_components(self):
        s = sample(words=[GTBox(BBox(0, 0, 100, 10), "W")])
        comps = [comp(0, 0, 60, 10, cid=1), comp(40, 0, 60, 10, cid=2)]
        m = word_metrics(s, page(comps))
        assert m.mean_coverage == pytest.approx(1.0)  # union, not 1.2

    def test_no_regions_contributes_nothing(self):
        """Cheque and MIDV samples deliberately carry no text geometry."""
        m = word_metrics(sample(), page([comp(0, 0, 10, 10)]))
        assert m.n_gt == 0
        assert m.hit_rate == 0.0


class TestAggregation:
    def test_rates_come_from_summed_counts_not_averaged_rates(self):
        """A 12-character card must not weigh the same as a 900-character form."""
        sparse = char_metrics(
            sample(chars=[GTChar(BBox(0, 0, 10, 10), "A")]),
            page([comp(0, 0, 10, 10)]),
        )
        dense_gt = [GTChar(BBox(i * 20, 50, 10, 20), "X") for i in range(99)]
        dense = char_metrics(sample(chars=dense_gt), page([]))

        agg = aggregate_chars([sparse, dense])
        assert agg.n_gt == 100
        assert agg.isolated == 1
        assert agg.isolation_recall == pytest.approx(0.01)
        # The mean of the two per-sample rates would have been ~0.5.

    def test_mixed_granularity_is_flagged(self):
        word_only = sample(words=[GTBox(BBox(0, 0, 10, 10), "A")])
        line_only = sample()
        line_only.lines = [GTBox(BBox(0, 0, 10, 10), "A")]
        agg = aggregate_words([
            word_metrics(word_only, page([comp(0, 0, 10, 10)])),
            word_metrics(line_only, page([comp(0, 0, 10, 10)])),
        ])
        assert agg.granularity == "mixed"

    def test_percentiles(self):
        stats = percentiles([float(i) for i in range(1, 101)])
        assert stats["p50"] == pytest.approx(50.5)
        assert stats["max"] == pytest.approx(100.0)
        assert stats["n"] == 100

    def test_percentiles_of_nothing(self):
        assert percentiles([]) == {}


class TestBoxIndex:
    def test_finds_overlapping_boxes(self):
        boxes = [BBox(i * 50, 0, 20, 20) for i in range(10)]
        index = BoxIndex(boxes)
        hits = index.query(BBox(45, 0, 30, 20))
        assert 1 in hits

    def test_agrees_with_brute_force(self):
        boxes = [BBox((i * 37) % 400, (i * 53) % 200, 15, 15) for i in range(200)]
        index = BoxIndex(boxes)
        probe = BBox(100, 80, 60, 40)
        expected = {i for i, b in enumerate(boxes) if b.intersection_area(probe) > 0}
        assert expected <= index.query(probe)

    def test_empty(self):
        assert BoxIndex([]).query(BBox(0, 0, 10, 10)) == set()


def test_evaluate_carries_slice_keys_through():
    s = sample(chars=[GTChar(BBox(0, 0, 10, 10), "A")])
    s.meta["template"] = "cheque"
    m = evaluate(s, page([comp(0, 0, 10, 10)]))
    assert m.template == "cheque"
    assert m.script == "latin"
    assert m.capture == "scan"
    assert m.chars is not None
    assert "total" in m.timings_ms
