"""ICDAR-2019 SROIE — scanned receipts, the closest public analogue to our documents.

Licence on the mirror we use (``jsdnrs/ICDAR2019-SROIE``): **CC-BY-4.0**. That is the
most permissive terms of any real dataset in this benchmark, which is why it carries the
real-document slice.

Granularity caveat: the dataset's ``words`` column is misnamed. Its entries are text
*segments* — "TAN CHAY YEE", "TEL:07-388 2218 FAX:07-388 8218" — one per annotated
region, which is a line, not a word. They are loaded into :attr:`Sample.lines` and the
region metric reports ``granularity="line"`` for them. Treating them as words would make
SROIE's coverage look mechanically better than FUNSD's for no real reason.

Reads the local cache written by ``scripts/fetch_sroie.py``::

    bench/raw/sroie/rows.jsonl      one JSON row per line (key, bboxes, words, entities)
    bench/raw/sroie/images/<key>.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, Sample

from .base import AdapterError, read_image, require_dir, validate

FETCH_HINT = "python scripts/fetch_sroie.py --limit 50"

# Receipt fields that are personal or financial. Annotated, never acted on in Phases 0-2.
PII_ENTITIES = {"company": "name", "address": "address", "total": "account"}


def _bbox(v: list[int]) -> BBox | None:
    """SROIE boxes are [x0, y0, x1, y1]."""
    if len(v) != 4:
        return None
    x0, y0, x1, y1 = (int(round(float(c))) for c in v)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1 - x0, y1 - y0)


def sample_from_row(row: dict, image_path: Path) -> Sample:
    key = row["key"]
    texts = row.get("words") or []
    boxes = row.get("bboxes") or []
    if len(texts) != len(boxes):
        raise AdapterError(
            f"{key}: {len(texts)} transcripts but {len(boxes)} boxes -- the dataset "
            "layout has changed; re-check the column names before trusting any metric"
        )

    lines: list[GTBox] = []
    for text, raw in zip(texts, boxes):
        box = _bbox(raw)
        if box is not None and str(text).strip():
            lines.append(GTBox(box, text=str(text)))

    entities = row.get("entities") or {}
    pii: list[GTBox] = []
    for field, pii_type in PII_ENTITIES.items():
        value = (entities.get(field) or "").strip()
        if not value:
            continue
        # Entity values carry no geometry of their own; locate them by matching the
        # transcript of an annotated line. Lines that do not match are simply not
        # annotated as PII rather than guessed at.
        for line in lines:
            if value and value in line.text:
                pii.append(GTBox(line.bbox, text=value, type=pii_type))
                break

    return Sample(
        sample_id=f"sroie_{key}",
        source="sroie",
        capture="scan",
        script="latin",
        dpi=150,  # Not recorded per image; receipts here are ~460-1200 px wide.
        text="\n".join(l.text for l in lines),
        lines=lines,
        words=[],
        chars=None,
        pii=pii,
        regions_nontext=[],
        meta={
            "template": "receipt",
            "gt_complete": True,
            "granularity_note": "source 'words' column is line-level",
            "licence": "CC-BY-4.0 (jsdnrs/ICDAR2019-SROIE mirror)",
            "image_file": image_path.name,
        },
    )


def convert(raw_dir: str | Path, store: BenchmarkStore, limit: int | None = None) -> list[Sample]:
    root = require_dir(raw_dir, "SROIE cache", FETCH_HINT)
    rows_path = root / "rows.jsonl"
    if not rows_path.exists():
        raise AdapterError(f"{rows_path} missing. Fetch it first: {FETCH_HINT}")

    samples: list[Sample] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = root / "images" / f"{row['key']}.jpg"
        if not image_path.exists():
            continue
        sample = sample_from_row(row, image_path)
        store.write(validate(sample), read_image(image_path))
        samples.append(sample)
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise AdapterError(f"converted 0 samples from {root}; re-run {FETCH_HINT}")
    return samples
