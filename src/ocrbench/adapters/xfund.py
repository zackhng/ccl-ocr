"""XFUND — real scanned forms in seven languages (nnul/xfund-multilingual mirror).

The multilingual sibling of FUNSD: real forms, every word with an absolute-pixel box and
its transcript. The European languages (de, es, fr, it, pt) exercise accented Latin —
the recogniser's extended charset — on real scans; Chinese (zh) and Japanese (ja) are
kept apart for the Phase 7 recognisers and are not converted here by default.

Licence: XFUND is released for research use (non-commercial); approved for internal
development only. Words are complete within each form, so ``gt_complete`` is True.
Lines are not annotated; word metrics only.
"""

from __future__ import annotations

from ocr.types import BBox

from ocrbench.gt import GTBox, Sample

from .base import validate

LATIN_LANGUAGES = ("de", "es", "fr", "it", "pt")
LICENCE = "XFUND, research use only (internal development approved)"


def sample_from_record(record_id: str, words: list[str], bboxes: list[list[int]]) -> Sample:
    lang = record_id.split("_")[0]
    gt_words = []
    for text, b in zip(words, bboxes):
        text = str(text).strip()
        x0, y0, x1, y1 = (int(v) for v in b)
        if text and x1 > x0 and y1 > y0:
            gt_words.append(GTBox(BBox(x0, y0, x1 - x0, y1 - y0), text=text))
    return validate(Sample(
        sample_id=f"xfund_{record_id}",
        source="xfund",
        capture="scan",
        script="latin" if lang in LATIN_LANGUAGES else lang,
        dpi=300,
        text=" ".join(w.text for w in gt_words),
        lines=[],
        words=gt_words,
        chars=None,
        meta={"template": "form", "language": lang, "gt_complete": True, "licence": LICENCE},
    ))
