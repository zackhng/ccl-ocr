"""The Latin character set shared by the glyph CNN, the character LM and the decoder.

One definition, so the three models can never disagree about what index 37 means.

**Glyph classes** are the printable characters plus three *structural* classes, which
the CNN needs because ~18% of components are not one clean character (Phase 2b
dossier):

``<NONTEXT>``
    junk — speckle, a portrait's texture, a rule fragment. Decodes to nothing.
``<MULTI>``
    several characters fused into one component. The decoder expands it into
    column-cut hypotheses rather than guessing one letter.
``<PART>``
    a piece of a character (a broken stroke). The decoder may join it to a neighbour.

**LM tokens** are the printable characters plus a word boundary (space) and the
sequence markers. The LM never sees the structural classes.
"""

from __future__ import annotations

import string
import unicodedata


def _vietnamese_letters() -> str:
    """Every Vietnamese letter, composed (NFC): 12 vowel bases x 5 tone marks, the bare
    bases, and d-stroke; both cases. Generated rather than typed so no letter is missed."""
    bases = "aăâeêioôơuưy"
    tones = ("̀", "́", "̉", "̃", "̣")  # grave acute hook tilde dot
    letters = set(bases) | {"đ"}
    for b in bases:
        for t in tones:
            letters.add(unicodedata.normalize("NFC", b + t))
    letters |= {c.upper() for c in letters}
    return "".join(sorted(letters))


_LATIN1_LETTERS = "".join(
    chr(c) for c in range(0xC0, 0x100) if chr(c).isalpha() and chr(c) not in "ªº"
)
"""À..ÿ letters: accented names and words in UK, Indonesian and expatriate documents."""

ASCII = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
CURRENCY = "£€¥₹฿₫"
"""Jurisdictions in scope: GBP, EUR, CNY/JPY, INR, THB, VND. (RM, Rp, AED, SGD are
written in Latin letters.)"""

PRINTABLE = "".join(dict.fromkeys(ASCII + CURRENCY + _LATIN1_LETTERS + _vietnamese_letters()))
"""Every character the Latin recogniser can emit, without duplicates and in a stable
order. Case is kept: names, addresses and mixed-case receipts need it, and the CNN's
geometry inputs separate o/O, c/C.

Covers the Latin-script jurisdictions in scope — UK, Singapore, Indonesia, Vietnam —
and the Latin text inside Chinese, Indian, Thai and UAE documents. Han, Devanagari,
Thai and Arabic are Phase 7 recognisers, not classes here."""

NONTEXT, MULTI, PART = "<NONTEXT>", "<MULTI>", "<PART>"
STRUCTURAL = (NONTEXT, MULTI, PART)

GLYPH_CLASSES: tuple[str, ...] = tuple(PRINTABLE) + STRUCTURAL
GLYPH_INDEX: dict[str, int] = {c: i for i, c in enumerate(GLYPH_CLASSES)}
N_GLYPH_CLASSES = len(GLYPH_CLASSES)

PAD, BOS, EOS, SPACE = "<PAD>", "<BOS>", "<EOS>", " "
LM_TOKENS: tuple[str, ...] = (PAD, BOS, EOS, SPACE) + tuple(PRINTABLE)
LM_INDEX: dict[str, int] = {c: i for i, c in enumerate(LM_TOKENS)}
N_LM_TOKENS = len(LM_TOKENS)

# Characters PRINTABLE lacks are mapped to a close equivalent before training or
# scoring, so corpus text never carries a symbol the CNN cannot emit.
_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
    "—": "-", "−": "-", " ": " ", "…": "...", "×": "x",
    "•": "*", "·": ".",
}


def normalise(text: str) -> str:
    """Map text onto the charset: fold typographic variants, drop what cannot be
    emitted, collapse whitespace. Used for LM corpora and training-page text."""
    out = []
    # NFC first: Vietnamese text often arrives decomposed (base + combining marks),
    # and the charset holds composed letters.
    for ch in unicodedata.normalize("NFC", text):
        ch = _FOLD.get(ch, ch)
        for c in ch:
            if c in LM_INDEX and c not in (PAD, BOS, EOS):
                out.append(c)
            elif c.isspace():
                out.append(" ")
    return " ".join("".join(out).split())


def glyph_label(ch: str) -> int:
    """Class index of a ground-truth character; characters outside the set are
    NONTEXT for training purposes (they cannot be emitted anyway)."""
    return GLYPH_INDEX.get(_FOLD.get(ch, ch), GLYPH_INDEX[NONTEXT])


def lm_encode(text: str, bos: bool = True, eos: bool = True) -> list[int]:
    ids = [LM_INDEX[c] for c in normalise(text)]
    return ([LM_INDEX[BOS]] if bos else []) + ids + ([LM_INDEX[EOS]] if eos else [])


def glyph_to_lm(glyph_index: int) -> int | None:
    """LM token for a glyph class; ``None`` for the structural classes."""
    ch = GLYPH_CLASSES[glyph_index]
    return LM_INDEX.get(ch) if ch not in STRUCTURAL else None
