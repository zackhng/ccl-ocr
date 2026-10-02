"""Training pages for the Phase 3 recogniser, rendered from public text.

The benchmark templates draw their words from a tiny fixed vocabulary (20 surnames, 8
streets...). Training on those pages would teach the CNN — and through the decoder, the
whole recogniser — the benchmark's own words. These pages instead lay out *corpus* text
(Wikipedia en/id/vi, real receipt and form transcripts) and freshly generated
structured values, with the same fonts, renderer and capture degradations as the
benchmark, so the CNN learns the pipeline's crops without its vocabulary.

Layouts reproduce the hazards the isolation dossier found, so the CNN sees them:
prose, label/value forms with text sitting on rules, monospaced receipts with wide
letter spacing, and card-like pages with light text on a dark banner. Case is varied
(financial documents are mostly upper case; corpus text is mostly mixed).

Seeds here must never be the benchmark's (7) or the held-out tuning set's (11):
:data:`RESERVED_SEEDS` is checked.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from ocr.recog.charset import normalise
from ocr.types import BBox

from . import content
from .fonts import FontSpec, covers, fonts_for
from .render import DocumentSpec, Shape, TextLine

RESERVED_SEEDS = frozenset({7, 11})

LABELS = (
    "NAME", "Nama", "Tên", "ADDRESS", "Alamat", "Địa chỉ", "DATE", "Tanggal", "Ngày",
    "AMOUNT", "Jumlah", "Số tiền", "ACCOUNT NO", "No. Rekening", "REFERENCE", "TOTAL",
    "ID NO", "NIK", "PAN", "Passport No", "IBAN", "Phone", "Email", "Description",
)


class TextSource:
    """Lines from the corpora, plus generated structured values in the formats of the
    jurisdictions in scope."""

    def __init__(self, corpora: Path, financial_weight: float = 0.35) -> None:
        self.public = (corpora / "public.txt").read_text(encoding="utf-8").splitlines()
        self.financial = (corpora / "financial.txt").read_text(encoding="utf-8").splitlines()
        self.financial_weight = financial_weight

    def phrase(self, rng: random.Random, max_chars: int) -> str:
        pool = self.financial if rng.random() < self.financial_weight else self.public
        line = rng.choice(pool)
        if len(line) > max_chars:
            words = line.split()
            start = rng.randrange(max(1, len(words) - 2))
            out = []
            for w in words[start:]:
                if sum(map(len, out)) + len(out) + len(w) > max_chars:
                    break
                out.append(w)
            line = " ".join(out) or line[:max_chars]
        return line

    @staticmethod
    def value(rng: random.Random) -> str:
        """A structured value, fresh each time; formats mirror ocr.recog.formats."""
        kind = rng.choice(["nric", "amount", "amount_dot", "amount_lakh", "date", "date_iso",
                           "account", "phone", "pan", "iban", "nik", "ref"])
        if kind == "nric":
            return content.nric(rng)
        if kind == "amount":
            return rng.choice(["", "$", "S$", "RM ", "£", "AED "]) + f"{rng.uniform(1, 99999):,.2f}"
        if kind == "amount_dot":  # Indonesia / Vietnam: dots group thousands
            n = f"{rng.randrange(1000, 99_000_000):,}".replace(",", ".")
            return rng.choice(["Rp ", "", ""]) + n + rng.choice(["", "₫", ",00"])
        if kind == "amount_lakh":  # India: 1,00,000.00
            n = str(rng.randrange(100000, 99_999_999))
            head, tail = n[:-3], n[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            return "₹" + ",".join([head] + groups + [tail]) + f".{rng.randrange(100):02d}"
        if kind == "date":
            d = date(1950, 1, 1) + timedelta(days=rng.randrange(30000))
            return d.strftime(rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y"]))
        if kind == "date_iso":
            d = date(1990, 1, 1) + timedelta(days=rng.randrange(14000))
            return d.isoformat()
        if kind == "account":
            return content.account_number(rng)
        if kind == "phone":
            return rng.choice(["+65 ", "+62 ", "+84 ", "+91 ", "+44 ", "+971 ", "+66 "]) + \
                " ".join(str(rng.randrange(1000, 9999)) for _ in range(2))
        if kind == "pan":
            letters = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
            return f"{letters}{rng.randrange(10000):04d}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
        if kind == "iban":
            return rng.choice(["GB", "AE"]) + f"{rng.randrange(100):02d}" + \
                "".join(rng.choice("0123456789") for _ in range(rng.choice([18, 19])))
        if kind == "nik":
            return "".join(rng.choice("0123456789") for _ in range(16))
        return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10**7, 10**8)}"


def _case(rng: random.Random, text: str) -> str:
    r = rng.random()
    if r < 0.35:
        return text.upper()
    if r < 0.45:
        return text.title()
    return text


def _font(rng: random.Random, size: int, text: str, mono: bool = False) -> FontSpec | None:
    """A face that really has every glyph of ``text`` (Vietnamese especially): a missing
    glyph would put GT characters in the image that are not legible."""
    pool = (fonts_for("mono") if mono else []) or fonts_for("latin")
    for spec in rng.sample(pool, len(pool)):
        s = spec.at(size)
        if all(covers(s, ch) for ch in set(text) if not ch.isspace()):
            return s
    return None


def _add(lines: list[TextLine], rng, text: str, x: int, y: int, size: int, mono=False, fill=None) -> int:
    """Append a line if a covering font exists; return its advance height."""
    text = normalise(text)
    font = _font(rng, size, text, mono) if text else None
    if font is None:
        return 0
    lines.append(TextLine(text, x, y, font, fill=fill or (20 + rng.randrange(40),) * 3))
    return int(size * rng.uniform(1.35, 1.9))


def prose(rng: random.Random, src: TextSource) -> DocumentSpec:
    w, h = 1240, 1754
    size = rng.choice([18, 22, 26, 30, 34])
    lines: list[TextLine] = []
    y = 80
    while y < h - 2 * size:
        text = _case(rng, src.phrase(rng, rng.randrange(25, 70)))
        y += _add(lines, rng, text, 80, y, size) or size
    return DocumentSpec(w, h, lines=lines, dpi=150, meta={"template": "corpus_prose"})


def fields(rng: random.Random, src: TextSource) -> DocumentSpec:
    """Label/value pairs, half of them written on a rule — the cheque/form hazard."""
    w, h = 1240, 1754
    label_size, value_size = rng.choice([(18, 24), (20, 28), (22, 30)])
    lines: list[TextLine] = []
    shapes: list[Shape] = []
    y = 90
    while y < h - 120:
        col = rng.choice([80, 640]) if rng.random() < 0.4 else 80
        label = _case(rng, rng.choice(LABELS))
        value = src.value(rng) if rng.random() < 0.55 else _case(rng, src.phrase(rng, 32))
        _add(lines, rng, label, col, y, label_size, fill=(100, 100, 110))
        adv = _add(lines, rng, value, col + rng.choice([0, 220]), y + label_size + 6, value_size)
        if adv and rng.random() < 0.5:
            rule_y = y + label_size + 6 + int(value_size * 1.05)
            shapes.append(Shape(BBox(col, rule_y, rng.randrange(380, 520), 2), "rule",
                                fill=(70, 70, 75), annotate=True))
        y += label_size + (adv or value_size) + rng.randrange(18, 40)
    return DocumentSpec(w, h, lines=lines, shapes=shapes, dpi=150, meta={"template": "corpus_fields"})


def receipt(rng: random.Random, src: TextSource) -> DocumentSpec:
    """Narrow monospaced thermal receipt: wide letter spacing, low resolution."""
    w, h = 576, 1400
    size = rng.choice([16, 18, 20])
    lines: list[TextLine] = []
    y = 40
    while y < h - 60:
        if rng.random() < 0.6:
            item = _case(rng, src.phrase(rng, 18)).upper()
            amount = src.value(rng) if rng.random() < 0.3 else f"{rng.uniform(0.5, 999):.2f}"
            _add(lines, rng, item, 24, y, size, mono=True)
            adv = _add(lines, rng, amount, w - 24 - int(len(amount) * size * 0.6), y, size, mono=True)
        else:
            adv = _add(lines, rng, src.phrase(rng, 30).upper(), 24, y, size, mono=True)
        y += adv or size
    return DocumentSpec(w, h, background=(245, 244, 240), lines=lines, dpi=120,
                        meta={"template": "corpus_receipt"})


def card(rng: random.Random, src: TextSource) -> DocumentSpec:
    """Card layout: light text on a dark banner, small fields on a tint."""
    w, h = 1011, 638
    banner_color = rng.choice([(28, 52, 96), (96, 28, 36), (30, 70, 58), (40, 40, 40)])
    banner_h = rng.randrange(70, 110)
    shapes = [Shape(BBox(0, 0, w, banner_h), "panel", fill=banner_color, annotate=False)]
    lines: list[TextLine] = []
    _add(lines, rng, _case(rng, src.phrase(rng, 28)).upper(), 30, 16, 30, fill=(245, 245, 248))
    y = banner_h + 30
    while y < h - 60:
        x = rng.choice([40, 330, 620])
        _add(lines, rng, _case(rng, rng.choice(LABELS)).upper(), x, y, 20, fill=(110, 112, 125))
        value = src.value(rng) if rng.random() < 0.5 else _case(rng, src.phrase(rng, 22))
        y += 24 + (_add(lines, rng, value, x, y + 24, rng.choice([24, 28, 34])) or 30) + 10
    tint = rng.choice([(236, 240, 247), (243, 238, 230), (238, 245, 239), (250, 250, 250)])
    return DocumentSpec(w, h, background=tint, lines=lines, shapes=shapes, dpi=300,
                        meta={"template": "corpus_card"})


LAYOUTS = {"prose": prose, "fields": fields, "receipt": receipt, "card": card}
LAYOUT_WEIGHTS = {"prose": 0.3, "fields": 0.3, "receipt": 0.2, "card": 0.2}
PROFILE_WEIGHTS = {"clean_scan": 0.15, "scan": 0.35, "photo": 0.35, "hard_photo": 0.15}


def build_spec(seed: int, src: TextSource) -> tuple[DocumentSpec, str, random.Random]:
    """``(spec, profile_name, rng)`` for one training page."""
    if seed in RESERVED_SEEDS:
        raise ValueError(f"seed {seed} is reserved for evaluation")
    rng = random.Random(seed)
    layout = rng.choices(list(LAYOUT_WEIGHTS), weights=list(LAYOUT_WEIGHTS.values()))[0]
    profile = rng.choices(list(PROFILE_WEIGHTS), weights=list(PROFILE_WEIGHTS.values()))[0]
    return LAYOUTS[layout](rng, src), profile, rng
