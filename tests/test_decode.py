"""Joint decoder (Phase 3) on hand-set CNN posteriors and a fake LM with known
preferences, so each outcome follows from the scores rather than from a trained model."""

from __future__ import annotations

import math

import torch

from ocr.recog.charset import GLYPH_INDEX, LM_INDEX, MULTI, N_GLYPH_CLASSES, N_LM_TOKENS, NONTEXT
from ocr.recog.decode import DecodeConfig, decode_line
from ocr.recog.formats import sg_nric_check_letter


def cnn(*readings) -> torch.Tensor:
    """Log-prob rows: each reading is {char: prob}; the rest share a tiny floor."""
    rows = torch.full((len(readings), N_GLYPH_CLASSES), 1e-6)
    for r, probs in enumerate(readings):
        for ch, p in probs.items():
            rows[r, GLYPH_INDEX[ch]] = p
    return torch.log(rows / rows.sum(1, keepdim=True))


class FakeLM(torch.nn.Module):
    """Bigram LM: after 'C' it strongly expects 'H'; otherwise uniform. The state
    tensor carries the previous token id."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _logp(self, prev: torch.Tensor) -> torch.Tensor:
        out = torch.full((prev.shape[0], N_LM_TOKENS), -math.log(N_LM_TOKENS))
        after_c = prev == LM_INDEX["C"]
        out[after_c] = math.log(0.001)
        out[after_c, LM_INDEX["H"]] = math.log(0.9)
        return out

    def start(self, batch, device):
        prev = torch.full((batch,), -1)
        return self._logp(prev), prev[None, :, None].float()

    def step(self, tokens, state):
        return self._logp(tokens), tokens[None, :, None].float()


LM = FakeLM()
ON = DecodeConfig(lm_weight=1.0)


def test_no_lm_is_argmax():
    w = cnn({"C": 0.9}, {"X": 0.55, "H": 0.45}, {"E": 0.9}, {"Q": 0.9})
    assert decode_line([w], None, DecodeConfig(lm_weight=0.0)) == ["CXEQ"]


def test_context_resolves_an_ambiguous_character():
    """The user's example: C ? E Q, with the crop torn between X and H."""
    w = cnn({"C": 0.9}, {"X": 0.55, "H": 0.45}, {"E": 0.9}, {"Q": 0.9})
    assert decode_line([w], LM, ON) == ["CHEQ"]


def test_confident_crop_beats_context():
    """Unlikely, never impossible: a clear X stays X."""
    w = cnn({"C": 0.9}, {"X": 0.999, "H": 0.0005}, {"E": 0.9}, {"Q": 0.9})
    assert decode_line([w], LM, DecodeConfig(lm_weight=0.3)) == ["CXEQ"]


def test_structured_field_ignores_the_lm_and_prefers_a_valid_checksum():
    digits = "1234567"
    good = sg_nric_check_letter("S", digits)
    bad = "Z" if good != "Z" else "Y"
    rows = [{"S": 0.95}] + [{d: 0.95} for d in digits] + [{bad: 0.55, good: 0.45}]
    assert decode_line([cnn(*rows)], LM, ON) == ["S" + digits + good]


def test_invalid_checksum_is_not_forced_when_no_candidate_validates():
    rows = [{"S": 0.95}] + [{d: 0.95} for d in "1234567"] + [{"Q": 0.99}]
    out = decode_line([cnn(*rows)], LM, ON)[0]
    assert out.startswith("S1234567")  # read as seen; never invented


def test_nontext_emits_nothing_and_words_are_kept_apart():
    a = cnn({"N": 0.9}, {NONTEXT: 0.9}, {"O": 0.9})
    b = cnn({"1": 0.9})
    assert decode_line([a, b], None, DecodeConfig(lm_weight=0.0)) == ["NO", "1"]


def test_multi_expands_through_the_callback():
    w = cnn({MULTI: 0.9, "m": 0.05})
    pieces = [cnn({"r": 0.9})[0], cnn({"n": 0.9})[0]]
    out = decode_line([w], None, DecodeConfig(lm_weight=0.0), expand=lambda wi, ci: pieces)
    assert out == ["rn"]


def test_multi_without_callback_falls_back_to_one_character():
    w = cnn({MULTI: 0.9, "m": 0.05})
    assert decode_line([w], None, DecodeConfig(lm_weight=0.0)) == ["m"]
