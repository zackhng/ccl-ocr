"""Rendering a document spec to pixels *and* to ground truth at the same time.

This is the whole reason the synthetic generator exists. Character-level ground truth
is the only way to measure "did CCL isolate this character", and no public dataset of
financial documents has it. Here we get it for free: the renderer knows exactly where
it put every glyph.

The honesty constraint is :data:`~ocrbench.synth.fonts.SIMPLE_SCRIPTS`. For Latin and
Han, one character maps to one glyph at one position, so per-character boxes are real.
For Devanagari, Arabic and Thai, shaping reorders, joins and stacks glyphs — "the box
of character i" is not a well-defined object, and emitting one anyway would be
fabricated ground truth that quietly inflates every metric computed against it. Those
samples get word-level ground truth and ``chars = None``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ocr.types import BBox
from ocrbench.gt import GTBox, GTChar

from .fonts import SIMPLE_SCRIPTS, FontSpec

RGB = tuple[int, int, int]


@dataclass(slots=True)
class TextLine:
    """One rendered line of text, with its semantic role attached."""

    text: str
    x: int
    y: int
    font: FontSpec
    fill: RGB = (25, 25, 28)
    pii_type: str = ""
    script: str = "latin"
    annotate: bool = True
    """Whether this line is content the engine is expected to isolate.

    False for printed decoration — the row of dashes a thermal receipt uses as a
    separator. The ink is rendered because the pipeline has to cope with it, but
    annotating 42 dashes as 42 characters would demand that CCL isolate glyphs that
    touch by design, and would score correct rule-routing as 42 misses."""

    nontext_type: str = ""
    """When ``annotate`` is False, record the line's extent as a non-text region of this
    type instead, so the filter's routing can still be checked against it."""


@dataclass(slots=True)
class Shape:
    """A non-text element: the things component filtering must route away."""

    bbox: BBox
    type: str
    fill: RGB | None = None
    outline: RGB | None = None
    width: int = 1
    seed: int = 0
    annotate: bool = True
    """Whether this shape is a non-text *region* in the ground truth.

    False for things that are background rather than content — the dark banner behind a
    card header is painted, but it is not a logo or a photo, and annotating it as one
    would make the non-text ground truth a lie."""


@dataclass(slots=True)
class DocumentSpec:
    width: int
    height: int
    background: RGB = (252, 251, 248)
    lines: list[TextLine] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)
    script: str = "latin"
    dpi: int = 300
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class RenderResult:
    image: np.ndarray
    """BGR uint8, for OpenCV."""

    lines: list[GTBox]
    words: list[GTBox]
    chars: list[GTChar] | None
    pii: list[GTBox]
    regions_nontext: list[GTBox]
    text: str


def _box_from_corners(x0: float, y0: float, x1: float, y1: float) -> BBox | None:
    x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1 - x0, y1 - y0)


def _char_boxes(draw: ImageDraw.ImageDraw, line: TextLine) -> list[tuple[int, BBox, str]]:
    """Per-character ink boxes, as (index_in_line, box, char).

    Offsets come from ``textlength`` over the *prefix*, so kerning accumulated up to
    each character is accounted for. The per-glyph ink extents come from
    ``font.getbbox``. Characters that render no ink (spaces) are skipped — a ground
    truth box around nothing would be scored as a miss forever.
    """
    font = line.font.load()
    out: list[tuple[int, BBox, str]] = []
    for i, ch in enumerate(line.text):
        if ch.isspace():
            continue
        advance = draw.textlength(line.text[:i], font=font)
        ink = font.getbbox(ch)
        if ink is None:
            continue
        box = _box_from_corners(
            line.x + advance + ink[0], line.y + ink[1],
            line.x + advance + ink[2], line.y + ink[3],
        )
        if box is not None:
            out.append((i, box, ch))
    return out


def _word_boxes(draw: ImageDraw.ImageDraw, line: TextLine) -> list[GTBox]:
    """Word boxes from prefix advances, which works for shaped scripts too."""
    font = line.font.load()
    words: list[GTBox] = []
    cursor = 0
    for token in line.text.split(" "):
        if not token:
            cursor += 1
            continue
        start_adv = draw.textlength(line.text[:cursor], font=font)
        end_adv = draw.textlength(line.text[: cursor + len(token)], font=font)
        ink = font.getbbox(token)
        if ink is not None:
            box = _box_from_corners(
                line.x + start_adv, line.y + ink[1], line.x + end_adv, line.y + ink[3]
            )
            if box is not None:
                words.append(GTBox(box, text=token))
        cursor += len(token) + 1
    return words


def _draw_photo(draw: ImageDraw.ImageDraw, shape: Shape, canvas: Image.Image) -> None:
    """A portrait placeholder: dense enough to behave like a real photo under CCL.

    Painted into its own tile and pasted, so the shoulders are clipped at the photo
    border instead of bleeding across the card — the ground-truth box says the photo
    ends here, and the pixels must agree.
    """
    b = shape.bbox
    rng = random.Random(shape.seed)
    tile = Image.new("RGB", (b.w, b.h))
    td = ImageDraw.Draw(tile)

    base = rng.randrange(110, 165)
    for i in range(b.h):
        v = int(base - 35 + 70 * (i / max(1, b.h)))
        td.line([(0, i), (b.w, i)], fill=(v, v - 4, v - 10))

    head_w, head_h = int(b.w * 0.46), int(b.h * 0.40)
    hx = (b.w - head_w) // 2
    hy = int(b.h * 0.14)
    td.ellipse([hx, hy, hx + head_w, hy + head_h], fill=(196, 168, 146))

    sh_w = int(b.w * 0.86)
    sx = (b.w - sh_w) // 2
    td.ellipse([sx, int(b.h * 0.62), sx + sh_w, b.h * 2], fill=(70, 78, 96))

    canvas.paste(tile, (b.x, b.y))


def _draw_signature(draw: ImageDraw.ImageDraw, shape: Shape, canvas: Image.Image) -> None:
    """A sparse connected scrawl — the case that must NOT be classified as a rule."""
    b = shape.bbox
    rng = random.Random(shape.seed)
    points = []
    n = rng.randrange(14, 26)
    for i in range(n):
        t = i / (n - 1)
        x = b.x + t * b.w
        y = b.cy + math.sin(t * rng.uniform(6, 14) + rng.random()) * b.h * rng.uniform(0.15, 0.4)
        points.append((x, y))
    draw.line(points, fill=(30, 40, 120), width=max(1, b.h // 18), joint="curve")


def _draw_barcode(draw: ImageDraw.ImageDraw, shape: Shape, canvas: Image.Image) -> None:
    b = shape.bbox
    rng = random.Random(shape.seed)
    x = b.x
    while x < b.x2 - 2:
        w = rng.choice([1, 1, 2, 3])
        if rng.random() < 0.55:
            draw.rectangle([x, b.y, x + w, b.y2], fill=(10, 10, 10))
        x += w + rng.choice([1, 1, 2])


def _draw_logo(draw: ImageDraw.ImageDraw, shape: Shape, canvas: Image.Image) -> None:
    b = shape.bbox
    rng = random.Random(shape.seed)
    color = (rng.randrange(20, 90), rng.randrange(40, 110), rng.randrange(90, 180))
    draw.ellipse([b.x, b.y, b.x + b.h, b.y2], fill=color)
    draw.polygon(
        [(b.x + b.h * 0.6, b.y), (b.x2, b.y + b.h * 0.5), (b.x + b.h * 0.6, b.y2)], fill=color
    )


_SHAPE_PAINTERS = {
    "photo": _draw_photo,
    "signature": _draw_signature,
    "barcode": _draw_barcode,
    "logo": _draw_logo,
}


def render(spec: DocumentSpec) -> RenderResult:
    """Rasterise a spec and emit the matching ground truth."""
    image = Image.new("RGB", (spec.width, spec.height), spec.background)
    draw = ImageDraw.Draw(image)

    nontext: list[GTBox] = []
    for shape in spec.shapes:
        painter = _SHAPE_PAINTERS.get(shape.type)
        if painter is not None:
            painter(draw, shape, image)
        elif shape.fill is not None:
            draw.rectangle(
                [shape.bbox.x, shape.bbox.y, shape.bbox.x2, shape.bbox.y2], fill=shape.fill
            )
        else:
            draw.rectangle(
                [shape.bbox.x, shape.bbox.y, shape.bbox.x2, shape.bbox.y2],
                outline=shape.outline or (120, 120, 120),
                width=shape.width,
            )
        if shape.annotate:
            nontext.append(GTBox(shape.bbox, type=shape.type))

    gt_lines: list[GTBox] = []
    gt_words: list[GTBox] = []
    gt_chars: list[GTChar] = []
    gt_pii: list[GTBox] = []
    text_parts: list[str] = []
    char_gt_valid = True

    for line in spec.lines:
        if not line.text.strip():
            continue
        font = line.font.load()
        draw.text((line.x, line.y), line.text, font=font, fill=line.fill)

        words = _word_boxes(draw, line)
        if not words:
            continue

        line_box = words[0].bbox
        for w in words[1:]:
            line_box = line_box.union(w.bbox)

        if not line.annotate:
            nontext.append(GTBox(line_box, type=line.nontext_type or "rule"))
            continue

        gt_words.extend(words)
        gt_lines.append(GTBox(line_box, text=line.text))
        text_parts.append(line.text)

        if line.script in SIMPLE_SCRIPTS:
            for _, box, ch in _char_boxes(draw, line):
                gt_chars.append(GTChar(box, ch))
        else:
            # One shaped line poisons character ground truth for the whole sample:
            # a partially-populated `chars` list would be read as "these are all the
            # characters", and every unlisted glyph would score as junk.
            char_gt_valid = False

        if line.pii_type:
            gt_pii.append(GTBox(line_box, text=line.text, type=line.pii_type))

    for shape in spec.shapes:
        if shape.type in ("photo", "signature"):
            gt_pii.append(GTBox(shape.bbox, type="face" if shape.type == "photo" else "signature"))

    bgr = np.array(image)[:, :, ::-1].copy()
    return RenderResult(
        image=bgr,
        lines=gt_lines,
        words=gt_words,
        chars=gt_chars if (char_gt_valid and gt_chars) else None,
        pii=gt_pii,
        regions_nontext=nontext,
        text="\n".join(text_parts),
    )
