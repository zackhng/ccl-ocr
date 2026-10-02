"""Format grammars for fields a language model must not "correct".

An ID, an account number or an amount has no linguistic context. A character LM trained
on words would happily turn ``T9879554A`` into something pronounceable. For these
tokens the decoder cuts the LM weight to ~0 and instead *prefers* (never forces) a
candidate that satisfies the field's grammar — and where the field carries a check
digit, the checksum catches a single misread character outright.

**Registry, per jurisdiction.** Jurisdictions in scope: Singapore, China, India, UK,
Thailand, Vietnam, Indonesia, UAE. Checksums are implemented only for published,
well-established schemes; everything else is checked on format alone rather than on a
guessed rule:

==========  ======================  ==============================================
SG          NRIC/FIN (S T F G)      check letter (M series: format only)
CN          resident ID, 18 chars   ISO 7064 MOD 11-2 (GB 11643)
IN          Aadhaar, 12 digits      Verhoeff
IN          PAN                     format
UK          National Insurance no.  format
UK, AE, …   IBAN                    ISO 13616 mod-97
AE          Emirates ID 784-…       Luhn
TH          national ID, 13 digits  weighted mod-11
ID          NIK, 16 digits          format + embedded birth date
VN          CCCD, 12 digits         format
any         passport MRZ line       ICAO 9303 7-3-1 check digits
==========  ======================  ==============================================

Classification is deliberately conservative: a token is a *word* unless it clearly
looks structured, because mis-gating is asymmetric — an ID read with LM weight can be
corrupted, a word read without it merely loses help.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# ------------------------------------------------------------------------- checksums

_SG_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)
_SG_ST = "JZIHGFEDCBA"
_SG_FG = "XWUTRQPNMLK"


def sg_nric_check_letter(prefix: str, digits: str) -> str | None:
    """S/T/F/G check letter; ``None`` where the scheme is not implemented (M series,
    whose table is not reproduced here without a verified source)."""
    if prefix not in "STFG" or len(digits) != 7 or not digits.isdigit():
        return None
    total = sum(int(d) * w for d, w in zip(digits, _SG_WEIGHTS))
    if prefix in "TG":
        total += 4
    return (_SG_ST if prefix in "ST" else _SG_FG)[total % 11]


def sg_nric_valid(t: str) -> bool:
    expected = sg_nric_check_letter(t[0], t[1:8])
    return expected is None or expected == t[8]


_CN_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def cn_resident_id_valid(t: str) -> bool:
    """GB 11643 / ISO 7064 MOD 11-2 over the first 17 digits."""
    total = sum(int(d) * w for d, w in zip(t[:17], _CN_WEIGHTS))
    return "10X98765432"[total % 11] == t[17].upper()


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def iban_valid(t: str) -> bool:
    s = t.replace(" ", "").upper()
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def th_national_id_valid(t: str) -> bool:
    d = [int(c) for c in t]
    total = sum(d[i] * (13 - i) for i in range(12))
    return (11 - total % 11) % 10 == d[12]


def mrz_check_digit(field: str) -> int:
    def value(c: str) -> int:
        if c.isdigit():
            return int(c)
        if c == "<":
            return 0
        return ord(c.upper()) - 55  # A=10
    return sum(value(c) * (7, 3, 1)[i % 3] for i, c in enumerate(field)) % 10


def id_nik_valid(t: str) -> bool:
    """Format plus the birth date embedded at digits 7-12 (DDMMYY; DD + 40 for women)."""
    dd, mm = int(t[6:8]), int(t[8:10])
    dd = dd - 40 if dd > 40 else dd
    return 1 <= dd <= 31 and 1 <= mm <= 12


# -------------------------------------------------------------------------- registry


@dataclass(frozen=True, slots=True)
class FieldFormat:
    name: str
    jurisdiction: str
    pattern: re.Pattern
    check: Callable[[str], bool] | None = None
    """Checksum or semantic validation beyond the pattern; ``None`` = format only."""

    def matches_shape(self, token: str) -> bool:
        return bool(self.pattern.match(token))

    def valid(self, token: str) -> bool:
        if not self.pattern.match(token):
            return False
        return self.check is None or self.check(token)


FORMATS: tuple[FieldFormat, ...] = (
    FieldFormat("sg_nric", "SG", re.compile(r"^[STFGM]\d{7}[A-Z]$"), sg_nric_valid),
    FieldFormat("cn_resident_id", "CN", re.compile(r"^\d{17}[\dXx]$"), cn_resident_id_valid),
    FieldFormat("in_pan", "IN", re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")),
    FieldFormat("uk_nino", "UK", re.compile(
        r"^(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]$")),
    FieldFormat("iban", "*", re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$"), iban_valid),
    FieldFormat("ae_emirates_id", "AE", re.compile(r"^784-?\d{4}-?\d{7}-?\d$"),
                lambda t: luhn_valid(t.replace("-", ""))),
    FieldFormat("th_national_id", "TH", re.compile(r"^\d{13}$"), th_national_id_valid),
    FieldFormat("id_nik", "ID", re.compile(r"^\d{16}$"), id_nik_valid),
    FieldFormat("in_aadhaar", "IN", re.compile(r"^[2-9]\d{11}$"), verhoeff_valid),
    FieldFormat("vn_cccd", "VN", re.compile(r"^0\d{11}$")),
)
"""Ordered: when a token's shape fits several (a 12-digit number could be an Aadhaar or
a CCCD), the first *valid* one wins; with none valid, the first shape match decides the
kind and the decoder prefers candidates that validate."""

AMOUNT_RE = re.compile(
    r"^[$£€¥₹฿₫]?-?("
    r"\d{1,3}(,\d{3})+(\.\d{1,2})?"        # 1,234.56   (SG, UK, CN, TH, AE)
    r"|\d{1,2}(,\d{2})*,\d{3}(\.\d{2})?"   # 1,00,000.00 (IN lakh grouping)
    r"|\d{1,3}(\.\d{3})+(,\d{1,2})?"       # 1.250.000,50 (ID, VN)
    r"|\d+[.,]\d{2}"                       # 12.50 / 12,50
    r")[₫]?$"
)
DATE_RE = re.compile(
    r"^\d{1,2}[/.-]\d{1,2}[/.-](\d{2}|\d{4})$"  # DD/MM/YYYY (and Thai BE years 25xx)
    r"|^\d{4}[/.-]\d{1,2}[/.-]\d{1,2}$"         # YYYY-MM-DD (CN, ISO)
)

FIELD_LABELS = {
    "id": ("NRIC", "FIN", "NRIC/FIN", "IC", "ID", "NIK", "PAN", "AADHAAR", "NINO", "CCCD",
           "EMIRATES", "PASSPORT"),
    "account": ("A/C", "ACCOUNT", "ACC", "IBAN", "NO.REK", "REKENING"),
    "amount": ("AMOUNT", "TOTAL", "SUBTOTAL", "BALANCE", "JUMLAH", "TONG", "S$", "SGD", "RM",
               "RP", "IDR", "INR", "RS", "THB", "AED", "GBP", "VND", "$"),
    "date": ("DATE", "DOB", "BIRTH", "TANGGAL", "NGAY", "NGÀY"),
}


def match_format(token: str) -> FieldFormat | None:
    """The format a token belongs to: the first valid one, else the first shape match."""
    t = token.strip()
    shape = None
    for f in FORMATS:
        if f.matches_shape(t):
            if f.valid(t):
                return f
            shape = shape or f
    return shape


def token_kind(token: str, previous: str = "") -> str:
    """A format name from :data:`FORMATS`, or 'amount' | 'date' | 'number' | 'word'.

    ``token`` is the CNN's best reading; ``previous`` the preceding word on the line,
    whose label can announce the field ("NRIC", "A/C NO", "TOTAL")."""
    t = token.strip()
    f = match_format(t)
    if f is not None:
        return f.name
    if DATE_RE.match(t):
        return "date"
    if AMOUNT_RE.match(t):
        return "amount"
    digits = sum(c.isdigit() for c in t)
    if t and digits / len(t) >= 0.5:
        return "number"
    label = previous.strip().rstrip(":").upper()
    if digits and any(label in labels for labels in FIELD_LABELS.values()):
        return "number"
    return "word"


def satisfies(token: str, kind: str) -> bool:
    """Does ``token`` satisfy the grammar of ``kind``? Words always do."""
    for f in FORMATS:
        if f.name == kind:
            return f.valid(token)
    if kind == "amount":
        return bool(AMOUNT_RE.match(token))
    if kind == "date":
        return bool(DATE_RE.match(token))
    if kind == "number":
        return sum(c.isdigit() for c in token) >= len(token) / 2
    return True


def uses_language_model(kind: str) -> bool:
    """Only plain words take language-model weight."""
    return kind == "word"
