"""Document templates.

Each template models a document class we actually have to read, and each one carries a
specific hazard for connected-component labelling:

* **ID card** — a dark header banner with light text (the inverted-region case), a dense
  portrait photo, and small tightly-spaced fields on a tinted ground.
* **Cheque** — a fixed layout whose highest-value fields sit *on* printed rules, so the
  glyphs touch a long horizontal component; plus a mono MICR line and a signature that
  must not be mistaken for a rule.
* **Receipt** — low-DPI thermal print, narrow column, condensed mono text, the regime
  where characters start to touch and strokes start to break.
* **Form** — ruled tables and boxed fields, i.e. a page full of components that look
  structural and must be routed, not classified.
* **Plain page** — dense body text, including the non-Latin scripts, for script-mix
  coverage ahead of Phase 7.
"""

from __future__ import annotations

import random
from typing import Callable

from ocr.types import BBox

from . import content
from .fonts import FontSpec, fonts_for
from .render import DocumentSpec, Shape, TextLine


def _pick(rng: random.Random, script: str, size: int) -> FontSpec:
    candidates = fonts_for(script)
    if not candidates:
        candidates = fonts_for("latin")
    return rng.choice(candidates).at(size)


def _mono(rng: random.Random, size: int) -> FontSpec:
    candidates = fonts_for("mono") or fonts_for("latin")
    return rng.choice(candidates).at(size)


# --------------------------------------------------------------------------- ID card


def id_card(rng: random.Random) -> DocumentSpec:
    """CR80 identity card at 300 DPI (85.6 x 54 mm)."""
    w, h = 1011, 638
    tint = rng.choice([(236, 240, 247), (243, 238, 230), (238, 245, 239)])
    banner_color = rng.choice([(28, 52, 96), (96, 28, 36), (30, 70, 58)])

    base = _pick(rng, "latin", 26)
    label_font = base.at(22)
    value_font = _pick(rng, "latin", 28)
    big_font = _mono(rng, 40)

    name = content.person_name(rng)
    number = content.nric(rng)
    dob = content.birth_date(rng)

    banner_h = 86
    shapes = [
        Shape(BBox(0, 0, w, banner_h), "panel", fill=banner_color, annotate=False),
        Shape(BBox(8, 8, w - 16, h - 16), "frame", outline=(150, 150, 160), width=3),
        Shape(BBox(48, banner_h + 48, 230, 290), "photo", seed=rng.randrange(1 << 30)),
        Shape(BBox(48, h - 96, 230, 46), "barcode", seed=rng.randrange(1 << 30)),
    ]

    lines = [
        # Light-on-dark: binarises into one solid banner unless polarity is recovered.
        TextLine("REPUBLIC OF SINGAPORE", 36, 18, base.at(30), fill=(248, 248, 250)),
        TextLine("IDENTITY CARD", 36, 52, label_font, fill=(226, 228, 236)),
        TextLine(number, 330, banner_h + 40, big_font, fill=(20, 20, 30), pii_type="nric"),
        TextLine("NAME", 330, banner_h + 108, label_font, fill=(110, 112, 125)),
        TextLine(name, 330, banner_h + 134, value_font, pii_type="name"),
        TextLine("RACE", 330, banner_h + 196, label_font, fill=(110, 112, 125)),
        TextLine(rng.choice(["CHINESE", "MALAY", "INDIAN", "EURASIAN"]), 330, banner_h + 222, value_font),
        TextLine("DATE OF BIRTH", 330, banner_h + 284, label_font, fill=(110, 112, 125)),
        TextLine(dob.strftime("%d-%m-%Y"), 330, banner_h + 310, value_font, pii_type="dob"),
        TextLine("SEX", 700, banner_h + 196, label_font, fill=(110, 112, 125)),
        TextLine(rng.choice(["M", "F"]), 700, banner_h + 222, value_font),
        TextLine("COUNTRY OF BIRTH", 620, banner_h + 284, label_font, fill=(110, 112, 125)),
        TextLine("SINGAPORE", 620, banner_h + 310, value_font),
    ]

    for i, addr_line in enumerate(content.address(rng)):
        lines.append(TextLine(addr_line, 330, h - 150 + i * 32, label_font.at(24), pii_type="address"))

    return DocumentSpec(
        width=w, height=h, background=tint, lines=lines, shapes=shapes,
        script="latin", dpi=300, meta={"template": "id_card"},
    )


# ---------------------------------------------------------------------------- cheque


def cheque(rng: random.Random) -> DocumentSpec:
    """Bank cheque at 300 DPI, roughly 170 x 80 mm."""
    w, h = 2008, 945
    bg = rng.choice([(247, 246, 240), (240, 245, 248), (250, 244, 240)])

    label = _pick(rng, "latin", 24)
    hand = _pick(rng, "latin", 34)
    bank_font = _pick(rng, "latin", 40)
    micr = _mono(rng, 38)

    value = content.amount(rng, 50000)
    payee = content.person_name(rng)
    acct = content.account_number(rng)
    cheque_no = f"{rng.randrange(100000, 999999)}"
    date = content.recent_date(rng)

    rule_color = (120, 125, 140)
    shapes = [
        Shape(BBox(60, 60, 110, 70), "logo", seed=rng.randrange(1 << 30)),
        # The payee and amount-in-words rules: the fields written on top of them are the
        # ones that matter, and their glyphs will touch these components.
        Shape(BBox(180, 356, 1400, 3), "rule", fill=rule_color),
        Shape(BBox(180, 470, 1600, 3), "rule", fill=rule_color),
        Shape(BBox(180, 560, 1600, 3), "rule", fill=rule_color),
        Shape(BBox(1620, 300, 330, 90), "frame", outline=(90, 95, 110), width=3),
        Shape(BBox(1450, 700, 460, 110), "signature", seed=rng.randrange(1 << 30)),
        Shape(BBox(1450, 820, 460, 3), "rule", fill=rule_color),
    ]

    lines = [
        TextLine(rng.choice(content.BANKS), 200, 70, bank_font),
        TextLine("CHEQUE NO " + cheque_no, 1620, 70, label),
        TextLine("DATE", 1620, 180, label, fill=(110, 115, 130)),
        TextLine(date.strftime("%d %m %Y"), 1700, 176, hand),
        TextLine("PAY", 80, 352, label, fill=(110, 115, 130)),
        TextLine(payee, 200, 318, hand, pii_type="name"),
        TextLine("OR BEARER", 1600, 352, label, fill=(110, 115, 130)),
        TextLine(content.amount_in_words(value), 200, 432, hand.at(30)),
        TextLine("SINGAPORE DOLLARS", 80, 496, label, fill=(110, 115, 130)),
        TextLine("$ " + content.money(value), 1650, 320, hand.at(42)),
        TextLine("A/C " + acct, 80, 600, label, pii_type="account"),
        TextLine("AUTHORISED SIGNATORY", 1480, 830, label, fill=(110, 115, 130)),
        # MICR: mono, fixed pitch, and the single most valuable line on the document.
        # The real E-13B transit/on-us symbols are not in any system font and render as
        # .notdef boxes, which would put glyphs in the image that the ground truth
        # claims are characters. ASCII stand-ins keep the band honest.
        TextLine(
            f"|:{cheque_no}:| |:{rng.randrange(1000, 9999)}:| {acct.replace('-', '')}|'",
            180, h - 120, micr, pii_type="account",
        ),
    ]

    return DocumentSpec(
        width=w, height=h, background=bg, lines=lines, shapes=shapes,
        script="latin", dpi=300, meta={"template": "cheque"},
    )


# --------------------------------------------------------------------------- receipt


def receipt(rng: random.Random) -> DocumentSpec:
    """Thermal receipt: narrow, low DPI, condensed mono text."""
    w = rng.choice([560, 600, 640])
    n_items = rng.randrange(5, 14)
    # Header block ends at y=256, items run 34 px each, then the totals block adds a
    # further 150 px plus a line of text. Undershooting this clips the last lines off
    # the canvas while the ground truth still claims their characters.
    h = 500 + n_items * 34

    body = _mono(rng, 20)
    header = _mono(rng, 26)

    merchant = rng.choice(content.MERCHANTS)
    date = content.recent_date(rng)
    lines = [
        TextLine(merchant, 40, 40, header),
        TextLine(content.address(rng)[0], 40, 80, body),
        TextLine(f"GST REG NO {rng.randrange(10000000, 99999999)}X", 40, 106, body),
        TextLine("-" * 42, 40, 140, body, annotate=False, nontext_type="rule"),
        TextLine(f"DATE {date.strftime('%d/%m/%Y')}  {rng.randrange(8, 22):02d}:{rng.randrange(60):02d}", 40, 168, body),
        TextLine(f"POS {rng.randrange(1, 9)}  TRACE {rng.randrange(100000, 999999)}", 40, 194, body),
        TextLine("-" * 42, 40, 222, body, annotate=False, nontext_type="rule"),
    ]

    total = 0.0
    y = 256
    for _ in range(n_items):
        price = content.amount(rng, 60)
        total += price
        item = rng.choice(content.ITEMS)
        lines.append(TextLine(item, 40, y, body))
        lines.append(TextLine(content.money(price), w - 150, y, body))
        y += 34

    gst = round(total * 0.09, 2)
    lines += [
        TextLine("-" * 42, 40, y + 10, body, annotate=False, nontext_type="rule"),
        TextLine("SUBTOTAL", 40, y + 44, body),
        TextLine(content.money(total), w - 150, y + 44, body),
        TextLine("GST 9%", 40, y + 72, body),
        TextLine(content.money(gst), w - 150, y + 72, body),
        TextLine("TOTAL", 40, y + 108, header.at(22)),
        TextLine(content.money(total + gst), w - 170, y + 108, header.at(22)),
        TextLine(f"CARD **** {rng.randrange(1000, 9999)}", 40, y + 150, body, pii_type="account"),
    ]

    return DocumentSpec(
        width=w, height=h, background=(250, 250, 248), lines=lines, shapes=[],
        script="latin", dpi=150, meta={"template": "receipt"},
    )


# ------------------------------------------------------------------------------ form


def form(rng: random.Random) -> DocumentSpec:
    """A4 account-opening form at 150 DPI: rules, boxed fields, a small table."""
    w, h = 1240, 1754
    label = _pick(rng, "latin", 22)
    value = _pick(rng, "latin", 24)
    title = _pick(rng, "latin", 34)

    shapes: list[Shape] = [Shape(BBox(60, 60, w - 120, h - 120), "frame", outline=(90, 90, 100), width=2)]
    lines = [
        TextLine(rng.choice(content.BANKS), 90, 90, title),
        TextLine("ACCOUNT OPENING APPLICATION", 90, 136, label.at(26)),
    ]

    fields = [
        ("FULL NAME", content.person_name(rng), "name"),
        ("NRIC / FIN", content.nric(rng), "nric"),
        ("DATE OF BIRTH", content.birth_date(rng).strftime("%d/%m/%Y"), "dob"),
        ("CONTACT NO", f"+65 {rng.randrange(8000, 9999)} {rng.randrange(1000, 9999)}", "phone"),
        ("RESIDENTIAL ADDRESS", content.address(rng)[0], "address"),
        ("POSTAL CODE", str(rng.randrange(100000, 829999)), "address"),
    ]
    y = 220
    for name, val, pii in fields:
        lines.append(TextLine(name, 100, y, label, fill=(100, 103, 115)))
        shapes.append(Shape(BBox(100, y + 28, w - 260, 44), "frame", outline=(140, 143, 155), width=2))
        lines.append(TextLine(val, 116, y + 36, value, pii_type=pii))
        y += 110

    # Ruled table: every rule here is a component the filter must route, and every cell
    # value is a component it must keep.
    table_top = y + 20
    rows, cols = 5, 4
    row_h, col_w = 52, (w - 200) // cols
    lines.append(TextLine("TRANSACTION HISTORY", 100, table_top - 34, label, fill=(100, 103, 115)))
    for r in range(rows + 1):
        shapes.append(Shape(BBox(100, table_top + r * row_h, col_w * cols, 2), "rule", fill=(120, 123, 135)))
    for c in range(cols + 1):
        shapes.append(Shape(BBox(100 + c * col_w, table_top, 2, row_h * rows), "rule", fill=(120, 123, 135)))

    headers = ["DATE", "DESCRIPTION", "DEBIT", "CREDIT"]
    for c, head in enumerate(headers):
        lines.append(TextLine(head, 112 + c * col_w, table_top + 14, label.at(20)))
    for r in range(1, rows):
        cells = [
            content.recent_date(rng).strftime("%d/%m/%y"),
            rng.choice(content.ITEMS),
            content.money(content.amount(rng, 900)),
            content.money(content.amount(rng, 900)),
        ]
        for c, cell in enumerate(cells):
            lines.append(TextLine(cell, 112 + c * col_w, table_top + r * row_h + 14, label.at(20)))

    sig_y = table_top + rows * row_h + 90
    shapes.append(Shape(BBox(100, sig_y, 420, 90), "signature", seed=rng.randrange(1 << 30)))
    shapes.append(Shape(BBox(100, sig_y + 100, 420, 2), "rule", fill=(120, 123, 135)))
    lines.append(TextLine("APPLICANT SIGNATURE", 100, sig_y + 110, label, fill=(100, 103, 115)))

    return DocumentSpec(
        width=w, height=h, background=(253, 253, 251), lines=lines, shapes=shapes,
        script="latin", dpi=150, meta={"template": "form"},
    )


# ------------------------------------------------------------------------ plain page

_SCRIPT_SAMPLES: dict[str, list[str]] = {
    "han": [
        "账户持有人姓名", "身份证号码", "交易记录与结余", "新加坡银行有限公司",
        "支票号码 四五六七八九", "请在此处签名确认", "存款金额 一万两千元整",
    ],
    "devanagari": [
        "खाता धारक का नाम", "पहचान संख्या", "लेनदेन का विवरण",
        "बैंक शाखा कार्यालय", "कुल राशि बारह हजार रुपये",
    ],
    "thai": [
        "ชื่อเจ้าของบัญชี", "เลขประจำตัวประชาชน", "รายการเดินบัญชี",
        "ธนาคารสาขาสำนักงานใหญ่", "จำนวนเงินรวมทั้งสิ้น",
    ],
    "arabic": [
        "اسم صاحب الحساب", "رقم الهوية الوطنية", "كشف الحساب البنكي",
        "المبلغ الإجمالي المستحق", "توقيع العميل المعتمد",
    ],
}


def plain_page(rng: random.Random, script: str = "latin") -> DocumentSpec:
    """Dense body text in one script. Script-mix coverage ahead of Phase 7."""
    w, h = 1240, 1754
    font = _pick(rng, script, rng.choice([24, 28, 32]))

    if script == "latin":
        pool = [
            "The account holder confirms the particulars stated above are true.",
            "Statement period 01 Jan 2026 to 31 Mar 2026, closing balance 12,480.55",
            "Please retain this advice for your records; no receipt will be issued.",
            "Reference number 8842-1097-33, settled via interbank GIRO on 14/02/2026.",
            "Fees and charges are levied in accordance with the published tariff.",
        ]
    else:
        pool = _SCRIPT_SAMPLES.get(script, [])
        if not pool:
            pool = ["placeholder"]

    lines = [TextLine("STATEMENT OF ACCOUNT", 100, 90, font.at(36))]
    y = 180
    while y < h - 140:
        lines.append(TextLine(rng.choice(pool), 100, y, font, script=script))
        y += int(font.size * 1.8)

    return DocumentSpec(
        width=w, height=h, background=(252, 252, 250), lines=lines, shapes=[],
        script=script, dpi=150, meta={"template": f"plain_{script}"},
    )


TEMPLATES: dict[str, Callable[[random.Random], DocumentSpec]] = {
    "id_card": id_card,
    "cheque": cheque,
    "receipt": receipt,
    "form": form,
    "plain_latin": lambda rng: plain_page(rng, "latin"),
    "plain_han": lambda rng: plain_page(rng, "han"),
    "plain_devanagari": lambda rng: plain_page(rng, "devanagari"),
    "plain_thai": lambda rng: plain_page(rng, "thai"),
    "plain_arabic": lambda rng: plain_page(rng, "arabic"),
}
