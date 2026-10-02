"""Format registry (Phase 3): checksums against published examples, and the
conservative token classification the decoder relies on."""

from __future__ import annotations

import random

import pytest

from ocr.recog import formats as F
from ocrbench.synth.content import nric


def _corrupt(token: str, i: int) -> str:
    d = token[i]
    return token[:i] + str((int(d) + 1) % 10) + token[i + 1:]


class TestChecksums:
    def test_sg_nric_matches_the_generator(self):
        rng = random.Random(3)
        ids = [nric(rng) for _ in range(300)]
        assert all(F.match_format(i).name == "sg_nric" and F.satisfies(i, "sg_nric") for i in ids)

    def test_sg_nric_wrong_letter_rejected(self):
        good = nric(random.Random(1))
        bad = good[:-1] + ("A" if good[-1] != "A" else "B")
        assert not F.satisfies(bad, "sg_nric")

    def test_cn_resident_id_published_example(self):
        assert F.cn_resident_id_valid("11010519491231002X")
        assert not F.cn_resident_id_valid("11010519491231003X")

    def test_iban_published_example(self):
        assert F.iban_valid("GB82WEST12345698765432")
        assert not F.iban_valid("GB82WEST12345698765433")

    def test_luhn_and_verhoeff_catch_single_digit_errors(self):
        # Build valid numbers by appending the right check digit, then corrupt each
        # position: both schemes must catch every single-digit substitution.
        for base in ("78419901234567", "12345678901"):
            luhn = next(base + str(c) for c in range(10) if F.luhn_valid(base + str(c)))
            verh = next(base + str(c) for c in range(10) if F.verhoeff_valid(base + str(c)))
            for i in range(len(luhn)):
                assert not F.luhn_valid(_corrupt(luhn, i))
                assert not F.verhoeff_valid(_corrupt(verh, i))

    def test_thai_id_checks_every_position_but_the_third(self):
        """The scheme weights position i by 13 - i; at i = 2 that is 11 = 0 mod 11, so
        the third digit is not covered by the check at all. A property of the published
        scheme, pinned here so nobody relies on it catching that position."""
        base = "110170020345"
        valid = next(base + str(c) for c in range(10) if F.th_national_id_valid(base + str(c)))
        for i in range(12):
            caught = not F.th_national_id_valid(_corrupt(valid, i))
            assert caught == (i != 2), i

    def test_mrz_check_digit_icao_example(self):
        # ICAO 9303 specimen: document number L898902C3, check digit 6.
        assert F.mrz_check_digit("L898902C3") == 6

    def test_indonesian_nik_birth_date(self):
        assert F.id_nik_valid("3174015503900001")   # DD 55 = female, 15th
        assert not F.id_nik_valid("3174019913900001")  # month 13


class TestClassification:
    @pytest.mark.parametrize("token,kind", [
        ("S1234567D", "sg_nric"),
        ("11010519491231002X", "cn_resident_id"),
        ("ABCPE1234F", "in_pan"),
        ("GB82WEST12345698765432", "iban"),
        ("1,234.56", "amount"),
        ("1,00,000.00", "amount"),      # Indian lakh grouping
        ("1.250.000", "amount"),        # Indonesian / Vietnamese grouping
        ("25.000,50", "amount"),
        ("14/03/2026", "date"),
        ("14/03/2569", "date"),         # Thai Buddhist-era year
        ("2026-03-14", "date"),
        ("CHEQUE", "word"),
        ("Thương", "word"),
    ])
    def test_kinds(self, token, kind):
        assert F.token_kind(token) == kind

    def test_label_announces_a_number(self):
        assert F.token_kind("8823-1093", "A/C") == "number"
        assert F.token_kind("ABC", "A/C") == "word"  # no digits: still a word

    def test_only_words_use_the_language_model(self):
        assert F.uses_language_model("word")
        assert not any(F.uses_language_model(k) for k in ("sg_nric", "amount", "date", "number"))
