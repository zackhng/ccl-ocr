"""Latin line CRNN plumbing: scored reads, case-insensitive CTC targets, word assignment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocr.recog import seq  # noqa: E402
from ocr.recog.recognizer import _set_line_text  # noqa: E402
from ocr.types import BBox, Line, Word  # noqa: E402


def _train_seq():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_seq.py"
    spec = importlib.util.spec_from_file_location("train_seq", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_seq"] = mod
    spec.loader.exec_module(mod)
    return mod


def _line(n_words: int) -> Line:
    words = [Word(bbox=BBox(10 + 50 * k, 5, 40, 20), text="?") for k in range(n_words)]
    return Line(bbox=BBox(10, 5, 50 * n_words - 10, 20), words=words)


class TestSetLineText:
    def test_matching_tokens_keep_word_boxes(self):
        line = _line(2)
        boxes = [w.bbox for w in line.words]
        _set_line_text(line, "TOTAL 29.50")
        assert [w.text for w in line.words] == ["TOTAL", "29.50"]
        assert [w.bbox for w in line.words] == boxes

    def test_mismatch_collapses_to_one_word(self):
        line = _line(3)
        _set_line_text(line, "TOTAL 29.50")
        assert len(line.words) == 1
        assert line.text == "TOTAL 29.50"
        assert line.words[0].bbox == line.bbox


class TestFoldCase:
    def test_lowercase_mass_counts_for_uppercase_target(self):
        ts = _train_seq()
        # classes: blank, "A", "a", "1"
        p = torch.tensor([[[0.1, 0.05, 0.8, 0.05]]]).log()
        folded = ts.fold_case(p, torch.tensor([1]), torch.tensor([2]))
        assert folded[0, 0, 1].exp().item() == pytest.approx(0.85)
        assert folded[0, 0, 3].exp().item() == pytest.approx(0.05)  # untouched


class TestReadScored:
    def test_scores_in_input_order_and_bounded(self, tmp_path):
        charset = seq.build_charset(["abc"])
        model = seq.CRNN(len(charset), hidden=32)
        path = tmp_path / "m.pt"
        seq.save(model, path, "latin", charset, {})
        reader = seq.SequenceRecognizer(path, "cpu")
        crops = [np.full((20, w), 255, np.uint8) for w in (300, 40, 120)]
        crops[1][5:15, 5:30] = 0
        out = reader.read_scored(crops, max_batch=2)
        assert len(out) == 3
        assert all(0.0 <= c <= 1.0 for _, c in out)
        assert [t for t, _ in out] == reader.read(crops)
