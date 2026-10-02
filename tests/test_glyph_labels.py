"""Glyph labelling for CNN training (Phase 3): synthetic GT matching and real-document
alignment, on hand-built clusters."""

from __future__ import annotations

from ocr.recog.charset import GLYPH_INDEX, MULTI, NONTEXT, PART
from ocr.recog.crops import Cluster
from ocr.types import BBox
from ocrbench.glyphs import doc_seed, is_tune_doc, label_by_alignment, label_clusters
from ocrbench.gt import GTBox, GTChar, Sample


def cl(x, y=10, w=10, h=20) -> Cluster:
    return Cluster(BBox(x, y, w, h), [], 0, 0, False)


def sample(words=(), chars=None, complete=True) -> Sample:
    return Sample(sample_id="t", source="x", capture="scan", script="latin", dpi=300,
                  words=[GTBox(BBox(*b), text=t) for b, t in words],
                  chars=chars, meta={"gt_complete": complete})


class TestAlignment:
    WORD = ((0, 5, 50, 30), "AB1")

    def test_counts_match_labels_in_x_order(self):
        clusters = [cl(30), cl(2), cl(16)]  # deliberately out of order
        labels = label_by_alignment(sample([self.WORD]), clusters)
        assert labels == [GLYPH_INDEX["1"], GLYPH_INDEX["A"], GLYPH_INDEX["B"]]

    def test_count_mismatch_skips_the_word(self):
        """Two glyphs fused into one cluster: never guess which is which."""
        assert label_by_alignment(sample([self.WORD]), [cl(2, w=24), cl(30)]) == [-1, -1]

    def test_dot_standing_in_for_a_letter_rejects_the_word(self):
        """Counts match only because a smudge replaced a fused letter."""
        clusters = [cl(2), cl(18, y=26, w=3, h=3), cl(30)]
        assert label_by_alignment(sample([self.WORD]), clusters) == [-1, -1, -1]

    def test_outside_any_word_is_nontext_only_when_gt_is_complete(self):
        stray = cl(200)
        assert label_by_alignment(sample([self.WORD]), [stray]) == [GLYPH_INDEX[NONTEXT]]
        assert label_by_alignment(sample([self.WORD], complete=False), [stray]) == [-1]

    def test_spaces_are_ignored_and_accents_kept(self):
        word = ((0, 5, 80, 30), "Ế a")
        labels = label_by_alignment(sample([word]), [cl(2), cl(40)])
        assert labels == [GLYPH_INDEX["Ế"], GLYPH_INDEX["a"]]


class TestSyntheticMatching:
    def test_match_merge_part_and_junk(self):
        chars = [GTChar(BBox(0, 10, 10, 20), "A"), GTChar(BBox(12, 10, 10, 20), "B"),
                 GTChar(BBox(40, 10, 10, 20), "C")]
        clusters = [
            cl(0),                     # A, exact
            cl(12, w=10),              # B, exact
            cl(40, w=10, h=8),         # top of C only: a fragment
            cl(100),                   # nothing there
        ]
        labels = label_clusters(sample(chars=chars), clusters)
        assert labels == [GLYPH_INDEX["A"], GLYPH_INDEX["B"], GLYPH_INDEX[PART], GLYPH_INDEX[NONTEXT]]

    def test_merge(self):
        chars = [GTChar(BBox(0, 10, 10, 20), "A"), GTChar(BBox(10, 10, 10, 20), "B")]
        assert label_clusters(sample(chars=chars), [cl(0, w=20)]) == [GLYPH_INDEX[MULTI]]


def test_tuning_split_is_stable_and_about_ten_percent():
    ids = [f"doc_{i}" for i in range(2000)]
    share = sum(map(is_tune_doc, ids)) / len(ids)
    assert 0.07 < share < 0.13
    assert [is_tune_doc(i) for i in ids[:50]] == [is_tune_doc(i) for i in ids[:50]]
    assert doc_seed("a") != doc_seed("b") and doc_seed("a") > 0


def test_thin_glyph_in_a_wide_gt_box_is_the_character_not_a_part():
    """Regression: an 'I' 3 px wide in a 10 px GT box (side bearings) has IoU 0.3. It is
    the whole character; labelling it <PART> taught CNN v1 to drop thin letters."""
    chars = [GTChar(BBox(0, 10, 10, 20), "I")]
    assert label_clusters(sample(chars=chars), [cl(4, w=3)]) == [GLYPH_INDEX["I"]]


def test_short_piece_inside_a_character_is_still_a_part():
    chars = [GTChar(BBox(0, 10, 10, 20), "E")]
    assert label_clusters(sample(chars=chars), [cl(0, w=10, h=6)]) == [GLYPH_INDEX[PART]]
