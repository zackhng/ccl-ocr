"""CORD v2 — real Indonesian receipts (naver-clova-ix/cord-v2, CC BY 4.0).

Phone photographs and scans of restaurant and shop receipts from Indonesia: the
Indonesian jurisdiction's real financial documents, with prices in local grouping
(``75,000`` / ``75.000``). Every word carries a quadrilateral and its transcript
(``valid_line[].words[]``), grouped into receipt lines.

Ground truth is complete for text inside the receipt's region of interest; regions the
annotators marked ``dontcare`` are not transcribed, so ``gt_complete`` is False (junk
and spurious rates are not computed against CORD).
"""

from __future__ import annotations

import json

from ocr.types import BBox

from ocrbench.gt import GTBox, Sample

from .base import validate

LICENCE = "CC BY 4.0 (naver-clova-ix/cord-v2)"


def _quad_box(q: dict) -> BBox | None:
    xs = [q[f"x{i}"] for i in range(1, 5)]
    ys = [q[f"y{i}"] for i in range(1, 5)]
    box = BBox.from_points(list(zip(xs, ys)))
    return box if box.w > 1 and box.h > 1 else None


def sample_from_record(ground_truth: str, split: str, index: int) -> Sample:
    gt = json.loads(ground_truth)
    words: list[GTBox] = []
    lines: list[GTBox] = []
    for line in gt.get("valid_line", []):
        line_words = []
        for w in line.get("words", []):
            text = str(w.get("text", "")).strip()
            box = _quad_box(w.get("quad", {}))
            if text and box is not None:
                line_words.append(GTBox(box, text=text))
        if not line_words:
            continue
        words += line_words
        lb = line_words[0].bbox
        for w in line_words[1:]:
            lb = lb.union(w.bbox)
        lines.append(GTBox(lb, text=" ".join(w.text for w in sorted(line_words, key=lambda w: w.bbox.x))))

    image_id = gt.get("meta", {}).get("image_id", index)
    return validate(Sample(
        sample_id=f"cord_{split}_{image_id}",
        source="cord",
        capture="photo",
        script="latin",
        dpi=150,
        text="\n".join(l.text for l in lines),
        lines=lines,
        words=words,
        chars=None,
        meta={"template": "receipt", "split": split, "jurisdiction": "ID",
              "gt_complete": False, "licence": LICENCE},
    ))
