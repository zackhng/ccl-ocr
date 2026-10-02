"""Correct rendering of shaped scripts (Devanagari, Thai, Arabic, and Han/Latin too),
for Phase 7 training data.

Pillow on this machine has no Raqm, so it draws Devanagari without conjuncts or vowel
reordering and Arabic unjoined and left-to-right — glyphs no real document contains.
Here HarfBuzz (``uharfbuzz``) does the shaping (cluster formation, reordering, bidi
direction for a single-direction run, contextual forms) and FreeType rasterises each
shaped glyph at its HarfBuzz position.

Deliberately *not* wired into the benchmark generator: turning shaping on there would
make the published seed-7 benchmark regenerate with the shaped strata it previously
skipped, and no longer match its committed manifest. Phase 7 builds its own data with
this module explicitly.

Ground truth is line and word level only. In a shaped script "the box of character i"
is not well defined (a Devanagari conjunct is several characters in one glyph; an
Arabic word is one connected stroke), so no character boxes are claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ocr.types import BBox


@dataclass(slots=True)
class ShapedLine:
    mask: np.ndarray
    """uint8 coverage (0..255), tight to the line's ink, ``h x w``."""
    origin: tuple[int, int]
    """Where the mask's top-left lands on the page, given the requested pen position."""
    words: list[tuple[BBox, str]]
    """Word boxes in page coordinates, in *logical* (reading) order, with their text."""
    direction: str
    """'ltr' or 'rtl' as HarfBuzz determined it."""


@lru_cache(maxsize=64)
def _faces(path: str, index: int):
    import freetype
    import uharfbuzz as hb

    blob = hb.Blob.from_file_path(path)
    return freetype.Face(path, index=index), hb.Face(blob, index)


def shape_line(text: str, x: int, y: int, font_path: str, size: int, index: int = 0) -> ShapedLine:
    """Shape and rasterise one line whose top-left pen position is ``(x, y)``.

    ``y`` is the top of the line box (ascender line), matching Pillow's convention in
    ``render.TextLine`` so shaped and simple lines lay out alike.
    """
    import freetype
    import uharfbuzz as hb

    ft_face, hb_face = _faces(font_path, index)
    ft_face.set_char_size(size * 64)
    font = hb.Font(hb_face)
    upem = hb_face.upem
    scale = size / upem

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)

    ascender = ft_face.size.ascender / 64.0
    baseline = y + ascender
    pen = float(x)
    pieces = []  # (left, top, bitmap, cluster)
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        g = ft_face.glyph
        bm = g.bitmap
        if bm.width and bm.rows:
            arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width]
            left = int(round(pen + pos.x_offset * scale + g.bitmap_left))
            top = int(round(baseline - pos.y_offset * scale - g.bitmap_top))
            pieces.append((left, top, arr, info.cluster))
        pen += pos.x_advance * scale

    if not pieces:
        return ShapedLine(np.zeros((1, 1), np.uint8), (x, y), [], str(buf.direction))
    x0 = min(p[0] for p in pieces)
    y0 = min(p[1] for p in pieces)
    x1 = max(p[0] + p[2].shape[1] for p in pieces)
    y1 = max(p[1] + p[2].shape[0] for p in pieces)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for left, top, arr, _ in pieces:
        region = mask[top - y0 : top - y0 + arr.shape[0], left - x0 : left - x0 + arr.shape[1]]
        np.maximum(region, arr, out=region)

    # Words: map each glyph's cluster (a character offset) to the word containing it.
    starts, word_texts, offset = [], [], 0
    for token in text.split(" "):
        if token:
            starts.append((offset, offset + len(token)))
            word_texts.append(token)
        offset += len(token) + 1
    boxes: list[list[int] | None] = [None] * len(word_texts)
    for left, top, arr, cluster in pieces:
        for wi, (a, b) in enumerate(starts):
            if a <= cluster < b:
                r = [left, top, left + arr.shape[1], top + arr.shape[0]]
                cur = boxes[wi]
                boxes[wi] = r if cur is None else [min(cur[0], r[0]), min(cur[1], r[1]),
                                                   max(cur[2], r[2]), max(cur[3], r[3])]
                break
    words = [(BBox(b[0], b[1], b[2] - b[0], b[3] - b[1]), t)
             for b, t in zip(boxes, word_texts) if b is not None]
    return ShapedLine(mask, (x0, y0), words, str(buf.direction))


def draw_shaped(image, line: ShapedLine, fill=(25, 25, 28)) -> None:
    """Composite a shaped line onto a PIL RGB image (in place)."""
    from PIL import Image

    h, w = line.mask.shape
    ink = Image.new("RGB", (w, h), fill)
    image.paste(ink, line.origin, Image.fromarray(line.mask, "L"))
