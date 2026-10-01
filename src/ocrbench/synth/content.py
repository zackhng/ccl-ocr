"""Seeded fake field values.

The values are *format*-realistic and content-fictional: a generated NRIC carries a
valid check letter so that a downstream validator behaves as it would in production,
but the digits are random and correspond to no person. Everything is driven by an
explicit :class:`random.Random` so a benchmark sample is reproducible from its seed.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

SURNAMES = [
    "TAN", "LIM", "LEE", "NG", "WONG", "CHAN", "KOH", "GOH", "TEO", "ONG", "CHUA",
    "YEO", "SIM", "CHEN", "LOW", "ANG", "TOH", "HENG", "SEAH", "QUEK",
]
GIVEN_NAMES = [
    "ZHAO HUI", "WEI MING", "JIA LING", "SIEW LAN", "KAI XIN", "MEI LING", "JUN JIE",
    "HUI YING", "CHEE KEONG", "LI WEN", "XIN YI", "YONG SHENG",
]
OTHER_NAMES = [
    "MUHAMMAD HAFIZ BIN OMAR", "NURUL AISYAH BINTE RAHMAN", "RAJESH S/O KUMAR",
    "PRIYA D/O SUBRAMANIAM", "DANIEL CHRISTOPHER WEE", "SARAH JANE MORRISON",
    "AHMAD FAIZAL BIN ISMAIL", "DEEPAK NAIR",
]

STREETS = [
    "ANG MO KIO AVE 3", "BEDOK NORTH ST 1", "CLEMENTI RD", "JURONG WEST ST 52",
    "TAMPINES ST 21", "SERANGOON CENTRAL", "BUKIT BATOK EAST AVE 6", "YISHUN RING RD",
]

BANKS = ["DBS BANK LTD", "OCBC BANK", "UNITED OVERSEAS BANK", "STANDARD CHARTERED BANK"]

MERCHANTS = [
    "FAIRPRICE FINEST", "SHENG SIONG SUPERMARKET", "COLD STORAGE", "GUARDIAN HEALTH",
    "KOPITIAM PTE LTD", "7-ELEVEN", "WATSONS PERSONAL CARE",
]

ITEMS = [
    "RICE 5KG", "MILK 1L", "EGGS 10S", "BREAD WHOLEMEAL", "COFFEE 200G", "SUGAR 1KG",
    "CHICKEN BREAST", "SOAP BAR 3S", "TISSUE 10R", "COOKING OIL 2L",
]

# Check-letter tables from the published NRIC/FIN checksum scheme.
_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)
_ST_LETTERS = "JZIHGFEDCBA"
_FG_LETTERS = "XWUTRQPNMLK"


def nric(rng: random.Random) -> str:
    """A format-valid, randomly-generated NRIC/FIN.

    Valid check letter so format validators behave realistically; the digits are random
    and identify nobody.
    """
    prefix = rng.choice("STFG")
    digits = [rng.randrange(10) for _ in range(7)]
    total = sum(d * w for d, w in zip(digits, _WEIGHTS))
    if prefix in "TG":
        total += 4
    table = _ST_LETTERS if prefix in "ST" else _FG_LETTERS
    return f"{prefix}{''.join(map(str, digits))}{table[total % 11]}"


def person_name(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"{rng.choice(SURNAMES)} {rng.choice(GIVEN_NAMES)}"
    return rng.choice(OTHER_NAMES)


def address(rng: random.Random) -> list[str]:
    blk = rng.randrange(1, 900)
    unit = f"#{rng.randrange(1, 25):02d}-{rng.randrange(1, 400):03d}"
    postal = f"SINGAPORE {rng.randrange(100000, 829999)}"
    return [f"BLK {blk} {rng.choice(STREETS)}", unit, postal]


def birth_date(rng: random.Random) -> date:
    start = date(1950, 1, 1)
    return start + timedelta(days=rng.randrange(0, 365 * 55))


def recent_date(rng: random.Random) -> date:
    return date(2026, 1, 1) + timedelta(days=rng.randrange(0, 300))


def account_number(rng: random.Random) -> str:
    return f"{rng.randrange(100, 999)}-{rng.randrange(100000, 999999)}-{rng.randrange(0, 9)}"


def amount(rng: random.Random, max_value: int = 20000) -> float:
    return round(rng.uniform(1.0, max_value), 2)


def money(value: float) -> str:
    return f"{value:,.2f}"


def amount_in_words(value: float) -> str:
    """Spell out a cheque amount. Covers the 0-999,999 range we generate."""
    units = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
             "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
             "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
    tens = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]

    def under_thousand(n: int) -> str:
        if n < 20:
            return units[n]
        if n < 100:
            return tens[n // 10] + ("" if n % 10 == 0 else " " + units[n % 10])
        rest = n % 100
        return units[n // 100] + " HUNDRED" + ("" if rest == 0 else " AND " + under_thousand(rest))

    whole = int(value)
    cents = int(round((value - whole) * 100))
    parts = []
    if whole >= 1000:
        parts.append(under_thousand(whole // 1000) + " THOUSAND")
        whole %= 1000
    if whole or not parts:
        parts.append(under_thousand(whole))
    words = " ".join(parts) + " DOLLARS"
    if cents:
        words += f" AND {under_thousand(cents)} CENTS"
    return words + " ONLY"


def mrz_lines(rng: random.Random, surname: str, given: str, doc_number: str) -> list[str]:
    """Two TD3 machine-readable-zone lines, 44 characters each.

    Mono-spaced, fixed-length, chevron-padded — a component-isolation case unlike any
    body text on the page, and the one a passport reader actually cares about.
    """
    def pad(s: str, n: int) -> str:
        s = "".join(ch if ch.isalnum() else "<" for ch in s.upper())
        return (s + "<" * n)[:n]

    line1 = "P<SGP" + pad(f"{surname}<<{given}", 39)
    dob = birth_date(rng).strftime("%y%m%d")
    expiry = (recent_date(rng) + timedelta(days=365 * 5)).strftime("%y%m%d")
    line2 = (
        pad(doc_number, 9) + str(rng.randrange(10)) + "SGP" + dob + str(rng.randrange(10))
        + rng.choice("MF") + expiry + str(rng.randrange(10)) + "<" * 14
        + str(rng.randrange(10)) + str(rng.randrange(10))
    )
    return [line1[:44], line2[:44]]
