"""Phase 6 metrics: localisation and recognition, comparable across engines.

Same rule as ``test_metrics.py``: every expected value follows from the definition, so
a failure points at the metric, not at an engine.
"""

from __future__ import annotations

import pytest

from ocr.types import BBox, Component, ComponentKind, Line, PageResult, Word
from ocrbench.gt import GTBox, Sample
from ocrbench.metrics import evaluate, levenshtein, region_metrics, text_metrics
from ocrbench.runner import aggregate_regions, aggregate_text


def sample(lines, source="synth", script="latin", **meta) -> Sample:
    return Sample(
        sample_id="t", source=source, capture="scan", script=script, dpi=300,
        lines=[GTBox(BBox(*b), text=t) for b, t in lines], meta=meta,
    )


def comp(x, y, w, h) -> Component:
    return Component(id=0, bbox=BBox(x, y, w, h), pixel_area=w * h // 2,
                     centroid=(x + w / 2, y + h / 2), fill_ratio=0.5, kind=ComponentKind.TEXT)


def line_page(*lines) -> PageResult:
    """A line-level engine's output: ``(x, y, w, h, text)`` per line, no components."""
    out = []
    for x, y, w, h, t in lines:
        b = BBox(x, y, w, h)
        out.append(Line(bbox=b, words=[Word(bbox=b, text=t)]))
    return PageResult(width=1000, height=1000, lines=out, timings_ms={"total": 1.0})


def comp_page(comps) -> PageResult:
    return PageResult(width=1000, height=1000, components=comps, timings_ms={"total": 1.0})


# A 200x20 GT line at (100, 100).
GT_LINE = ((100, 100, 200, 20), "HELLO WORLD")


class TestRegionMetrics:
    def test_line_box_and_character_boxes_score_alike(self):
        """The whole point of the metric: granularity must not change the answer."""
        s = sample([GT_LINE])
        as_line = region_metrics(s, line_page((100, 100, 200, 20, "")))
        # Ten 14 px glyphs with 6 px gaps, plus one wider word gap: well under the
        # one-line-height closing distance.
        glyphs = [comp(100 + i * 20, 100, 14, 20) for i in range(10)]
        as_chars = region_metrics(s, comp_page(glyphs))
        assert as_line.found == as_chars.found == 1
        assert as_line.mean_coverage == pytest.approx(1.0)
        assert as_chars.mean_coverage == pytest.approx(194 / 200)

    def test_padded_line_box_still_localises(self):
        """Paddle pads a 12 px line to ~28 px. Centre on the line -> it counts."""
        s = sample([((100, 100, 200, 12), "X")])
        m = region_metrics(s, line_page((95, 92, 210, 28, "")))
        assert m.found == 1

    def test_box_centred_on_the_next_line_does_not_count(self):
        s = sample([((100, 100, 200, 12), "X")])
        # Overlaps the GT line's bottom 6 px, but is centred 20 px below it.
        m = region_metrics(s, line_page((100, 106, 200, 28, "")))
        assert m.found == 0

    def test_paragraph_blob_does_not_localise_its_lines(self):
        s = sample([GT_LINE])
        m = region_metrics(s, comp_page([comp(100, 30, 200, 160)]))  # 8x the line height
        assert m.found == 0

    def test_partial_coverage_below_threshold_is_not_found(self):
        s = sample([GT_LINE])
        m = region_metrics(s, line_page((100, 100, 80, 20, "")))
        assert m.found == 0
        assert m.mean_coverage == pytest.approx(0.4)

    def test_spurious_boxes(self):
        s = sample([GT_LINE])
        m = region_metrics(s, line_page((100, 100, 200, 20, ""), (500, 500, 50, 20, "")))
        assert m.spurious == 1
        assert m.spurious_rate == pytest.approx(0.5)

    def test_spurious_undefined_when_gt_incomplete(self):
        s = sample([GT_LINE], source="cheque", gt_complete=False)
        m = region_metrics(s, line_page((500, 500, 50, 20, "")))
        assert m.spurious_rate is None

    def test_aggregates_from_counts(self):
        one = sample([GT_LINE])
        many = sample([((100, 100 + i * 40, 200, 20), "X") for i in range(9)])
        agg = aggregate_regions([
            region_metrics(one, line_page()),  # 0/1 found
            region_metrics(many, line_page(*[(100, 100 + i * 40, 200, 20, "") for i in range(9)])),
        ])
        assert agg.line_recall == pytest.approx(9 / 10)  # not the 0.5 a mean of rates gives


class TestLevenshtein:
    @pytest.mark.parametrize("a,b,d", [
        ("", "", 0), ("abc", "", 3), ("", "ab", 2), ("kitten", "sitting", 3),
        ("abc", "abc", 0), ("ab", "ba", 2),
    ])
    def test_distance(self, a, b, d):
        assert levenshtein(a, b) == d
        assert levenshtein(b, a) == d


class TestTextMetrics:
    def test_no_text_is_not_applicable(self):
        s = sample([GT_LINE])
        assert text_metrics(s, comp_page([comp(100, 100, 10, 20)])) is None
        assert text_metrics(s, line_page((100, 100, 200, 20, ""))) is None

    def test_perfect_read(self):
        m = text_metrics(sample([GT_LINE]), line_page((100, 100, 200, 20, "HELLO WORLD")))
        assert m.cer == 0.0
        assert m.word_f1 == pytest.approx(1.0)

    def test_dropped_space_costs_word_f1_not_cer(self):
        m = text_metrics(sample([GT_LINE]), line_page((100, 100, 200, 20, "HELLOWORLD")))
        assert m.cer == 0.0
        assert m.matched_words == 0

    def test_misread_character(self):
        m = text_metrics(sample([GT_LINE]), line_page((100, 100, 200, 20, "HELL0 WORLD")))
        assert m.edits == 1
        assert m.cer == pytest.approx(1 / 10)  # 10 non-space GT characters

    def test_unread_gt_line_is_a_deletion(self):
        s = sample([GT_LINE, ((100, 200, 200, 20), "ABCDE")])
        m = text_metrics(s, line_page((100, 100, 200, 20, "HELLO WORLD")))
        assert m.edits == 5
        assert m.n_gt_chars == 15

    def test_text_on_no_gt_line_is_an_insertion(self):
        m = text_metrics(sample([GT_LINE]),
                         line_page((100, 100, 200, 20, "HELLO WORLD"), (100, 600, 50, 20, "XYZ")))
        assert m.inserted_chars == 3
        assert m.cer == pytest.approx(3 / 10)

    def test_insertions_not_charged_when_gt_incomplete(self):
        s = sample([GT_LINE], source="funsd", gt_complete=False)
        m = text_metrics(s, line_page((100, 100, 200, 20, "HELLO WORLD"), (100, 600, 50, 20, "XYZ")))
        assert m.inserted_chars == 0
        assert m.n_pred_words == 2  # the stray word is not a false positive either

    def test_split_line_rejoins(self):
        """A detector splitting one line in two read it correctly; charge nothing."""
        m = text_metrics(sample([GT_LINE]),
                         line_page((201, 101, 99, 20, "WORLD"), (100, 100, 90, 20, "HELLO")))
        assert m.cer == 0.0

    def test_two_gt_entities_read_as_one_line(self):
        """FUNSD: 'TO:' and its value are two GT entities on one printed line."""
        s = sample([((100, 100, 30, 20), "TO:"), ((140, 100, 160, 20), "GEORGE")])
        m = text_metrics(s, line_page((95, 95, 210, 30, "TO: GEORGE")))
        assert m.cer == 0.0

    def test_padded_prediction_is_linked(self):
        """Regression: assignment keyed to the prediction's area alone rejected padded
        boxes and charged each line twice (CER 185% on a perfectly-read form)."""
        s = sample([((100, 100, 200, 12), "PADDED")])
        m = text_metrics(s, line_page((95, 90, 210, 30, "PADDED")))
        assert m.cer == 0.0

    def test_fullwidth_punctuation_is_normalised(self):
        s = sample([((100, 100, 200, 20), "NAME: TAN")])
        m = text_metrics(s, line_page((100, 100, 200, 20, "NAME： TAN")))
        assert m.cer == 0.0

    def test_case_folded_only_for_case_normalised_sources(self):
        lines = [((100, 100, 200, 20), "EMAIL")]
        pred = line_page((100, 100, 200, 20, "Email"))
        assert text_metrics(sample(lines, source="sroie", gt_complete=True), pred).cer == 0.0
        assert text_metrics(sample(lines), pred).edits == 4

    def test_han_words_are_characters(self):
        s = sample([((100, 100, 200, 20), "存款 金额")], script="han")
        m = text_metrics(s, line_page((100, 100, 200, 20, "存款金额")))
        assert m.cer == 0.0
        assert m.word_f1 == pytest.approx(1.0)

    def test_aggregates_from_counts(self):
        short = text_metrics(sample([((0, 0, 50, 20), "AB")]), line_page((0, 0, 50, 20, "XX")))
        long_ = text_metrics(sample([((0, 50, 500, 20), "A" * 98)]),
                             line_page((0, 50, 500, 20, "A" * 98)))
        agg = aggregate_text([short, long_, None])
        assert agg.cer == pytest.approx(2 / 100)  # not the 0.5 a mean of rates gives


class TestEvaluate:
    def test_line_engine_gets_no_component_metrics(self):
        """Zeros would read as failure; None reads as 'not applicable'."""
        m = evaluate(sample([GT_LINE]), line_page((100, 100, 200, 20, "HELLO WORLD")))
        assert m.chars is None and m.words is None
        assert m.regions.found == 1
        assert m.text.cer == 0.0

    def test_blank_ccl_page_keeps_word_metrics(self):
        m = evaluate(sample([GT_LINE]), comp_page([]))
        assert m.words is not None
        assert m.text is None
