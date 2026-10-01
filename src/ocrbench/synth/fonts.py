"""Font discovery, per script.

Phase 1 is Latin-only, but the generator is built multi-script from the start so Phase 7
inherits a working renderer instead of needing a new one. Fonts are discovered from the
system at import time and missing scripts degrade to "that script is unavailable" rather
than raising — a machine without a Thai font should still be able to generate the Latin
benchmark.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# Latin glyphs map one-to-one onto connected components (modulo touching pairs), so we
# can emit per-character ground truth. Devanagari, Arabic and Thai reorder, join and
# stack glyphs during shaping, so "the box of character i" is not a well-defined thing
# and those samples carry word-level ground truth only.
SIMPLE_SCRIPTS = frozenset({"latin", "han", "mono"})

SHAPED_SCRIPTS = frozenset({"devanagari", "arabic", "thai"})
"""Scripts that need a real shaping engine (HarfBuzz, via Pillow's Raqm backend).

Without it Pillow maps codepoints straight to glyphs: no contextual forms, no
reordering, no bidi. Arabic comes out in isolated forms and left-to-right; Devanagari
loses its conjuncts and matra reordering; Thai loses mark positioning. The result looks
like text and is not, so these strata are *skipped* rather than generated wrong —
a benchmark sample labelled ``script=arabic`` that contains mis-shaped glyphs would make
every Phase 7 number meaningless."""

_FONT_DIRS = [
    Path(r"C:\Windows\Fonts"),
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path(__file__).resolve().parents[3] / "assets" / "fonts",
]

# Candidate filenames per script, in rough preference order. Several are TrueType
# collections (.ttc) where the face we want is not index 0.
_CANDIDATES: dict[str, list[tuple[str, int]]] = {
    "latin": [
        ("arial.ttf", 0), ("arialbd.ttf", 0), ("ariali.ttf", 0),
        ("calibri.ttf", 0), ("calibrib.ttf", 0),
        ("times.ttf", 0), ("timesbd.ttf", 0),
        ("georgia.ttf", 0), ("verdana.ttf", 0), ("tahoma.ttf", 0),
        ("segoeui.ttf", 0), ("trebuc.ttf", 0),
        ("DejaVuSans.ttf", 0), ("LiberationSans-Regular.ttf", 0),
    ],
    # Fixed-pitch faces for the machine-readable parts of a document: the MRZ on a
    # passport, the MICR line on a cheque. These are the highest-value characters on
    # the page and they look nothing like body text.
    "mono": [
        ("consola.ttf", 0), ("consolab.ttf", 0), ("cour.ttf", 0), ("courbd.ttf", 0),
        ("DejaVuSansMono.ttf", 0), ("LiberationMono-Regular.ttf", 0),
    ],
    "han": [("msyh.ttc", 0), ("simsun.ttc", 0), ("simhei.ttf", 0), ("NotoSansCJK-Regular.ttc", 0)],
    "devanagari": [("Nirmala.ttc", 0), ("mangal.ttf", 0), ("NotoSansDevanagari-Regular.ttf", 0)],
    "thai": [("LeelawUI.ttf", 0), ("leelawui.ttf", 0), ("tahoma.ttf", 0), ("NotoSansThai-Regular.ttf", 0)],
    "arabic": [("segoeui.ttf", 0), ("arial.ttf", 0), ("NotoSansArabic-Regular.ttf", 0)],
}


@dataclass(frozen=True, slots=True)
class FontSpec:
    path: str
    size: int
    index: int = 0

    def load(self) -> ImageFont.FreeTypeFont:
        return _load_font(self.path, self.size, self.index)

    def at(self, size: int) -> FontSpec:
        return FontSpec(self.path, size, self.index)

    @property
    def family(self) -> str:
        return Path(self.path).stem


@lru_cache(maxsize=256)
def _load_font(path: str, size: int, index: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def _find(filename: str) -> Path | None:
    for directory in _FONT_DIRS:
        if not directory.exists():
            continue
        candidate = directory / filename
        if candidate.exists():
            return candidate
        # Linux nests fonts in per-family subdirectories.
        if directory.is_dir():
            for found in directory.rglob(filename):
                return found
    return None


@lru_cache(maxsize=1)
def available_fonts() -> dict[str, list[FontSpec]]:
    """Map script -> usable font faces, at a nominal size of 24 px."""
    out: dict[str, list[FontSpec]] = {}
    for script, candidates in _CANDIDATES.items():
        specs = []
        for name, index in candidates:
            path = _find(name)
            if path is not None:
                specs.append(FontSpec(str(path), 24, index))
        if specs:
            out[script] = specs
    return out


def fonts_for(script: str) -> list[FontSpec]:
    fonts = available_fonts()
    if script in fonts:
        return fonts[script]
    # Latin is the only script we insist on; without it nothing can be generated.
    if script == "latin":
        raise RuntimeError(
            "no Latin font found. Searched: "
            + ", ".join(str(d) for d in _FONT_DIRS)
            + ". Drop a .ttf into assets/fonts/ to proceed."
        )
    return []


@lru_cache(maxsize=1)
def has_shaping() -> bool:
    """Is Pillow built with Raqm (HarfBuzz + FriBiDi)?"""
    from PIL import features

    return bool(features.check("raqm"))


def supported_scripts() -> list[str]:
    """Scripts we can render *correctly* — font present, and shaping if required."""
    have = available_fonts()
    shaping = has_shaping()
    return [
        s for s in _CANDIDATES
        if s in have and (s not in SHAPED_SCRIPTS or shaping)
    ]


def covers(spec: FontSpec, text: str) -> bool:
    """Does this face actually have glyphs for ``text``?

    A font silently substitutes .notdef (an empty or boxed glyph) for missing
    characters, which would produce ground truth claiming characters that are not
    legible in the image. Checking the ink mask is the cheap way to catch it.
    """
    font = spec.load()
    try:
        mask = font.getmask(text, mode="L")
    except OSError:
        return False
    return mask.getbbox() is not None


def report() -> str:
    usable = set(supported_scripts())
    lines = [
        f"font search paths: {', '.join(str(d) for d in _FONT_DIRS if d.exists())}",
        f"text shaping (Pillow Raqm): {'yes' if has_shaping() else 'NO'}",
    ]
    for script, specs in sorted(available_fonts().items()):
        mark = " " if script in usable else "!"
        lines.append(
            f" {mark}{script:<12} {len(specs):>2} face(s): {', '.join(s.family for s in specs[:6])}"
        )
    missing = sorted(set(_CANDIDATES) - set(available_fonts()))
    if missing:
        lines.append(f"  no font: {', '.join(missing)}")
    blocked = sorted(SHAPED_SCRIPTS & set(available_fonts()) - usable)
    if blocked:
        lines.append(
            f"  skipped (needs shaping): {', '.join(blocked)} -- install a Pillow build "
            "with Raqm/libraqm to enable these strata"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(), file=sys.stderr)
