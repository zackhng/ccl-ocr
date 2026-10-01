"""FUNSD — 199 noisy scanned forms with word-level boxes and transcripts.

Licence: non-commercial, research and educational use only. Fine for benchmarking;
check before any of it informs a shipped model.

Annotation layout (``annotations/<id>.json``)::

    {"form": [{"box": [x0, y0, x1, y1], "text": "...", "label": "question",
               "words": [{"box": [x0, y0, x1, y1], "text": "..."}], ...}]}

Boxes are ``[left, top, right, bottom]``, not x/y/w/h — a conversion worth getting
wrong exactly once.

No character annotations, so these samples exercise the region-level metric only.
``gt_complete`` is True: FUNSD annotates every word on the page, so a surviving
component that lands on no word really is junk.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, Sample

from .base import AdapterError, read_image, require_dir, validate

FETCH_HINT = "python scripts/fetch_funsd.py"


def _bbox(v: list[int]) -> BBox | None:
    x0, y0, x1, y1 = (int(round(float(c))) for c in v)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1 - x0, y1 - y0)


def convert(
    raw_dir: str | Path,
    store: BenchmarkStore,
    limit: int | None = None,
    split: str = "training_data",
) -> list[Sample]:
    root = require_dir(raw_dir, "FUNSD", FETCH_HINT)
    ann_dir = root / split / "annotations"
    img_dir = root / split / "images"
    if not ann_dir.exists():
        # The zip has been repackaged several ways over the years; find it rather than
        # guessing one more path.
        found = next((p for p in root.rglob("annotations") if p.is_dir()), None)
        if found is None:
            raise AdapterError(
                f"no annotations/ directory under {root}. Expected "
                f"{split}/annotations/*.json -- re-run {FETCH_HINT}"
            )
        ann_dir, img_dir = found, found.parent / "images"

    samples: list[Sample] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        image_path = img_dir / f"{ann_path.stem}.png"
        if not image_path.exists():
            image_path = img_dir / f"{ann_path.stem}.jpg"
        if not image_path.exists():
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        lines: list[GTBox] = []
        words: list[GTBox] = []
        for entry in data.get("form", []):
            box = _bbox(entry["box"])
            if box is not None:
                lines.append(GTBox(box, text=entry.get("text", ""), type=entry.get("label", "")))
            for word in entry.get("words", []):
                wbox = _bbox(word["box"])
                if wbox is not None and word.get("text", "").strip():
                    words.append(GTBox(wbox, text=word["text"]))

        image = read_image(image_path)
        sample = Sample(
            sample_id=f"funsd_{ann_path.stem}",
            source="funsd",
            capture="scan",
            script="latin",
            dpi=200,  # FUNSD scans are ~1000x750 letter pages; not recorded per file.
            text="\n".join(l.text for l in lines if l.text),
            lines=lines,
            words=words,
            chars=None,
            pii=[],
            regions_nontext=[],
            meta={
                "template": "form",
                "gt_complete": True,
                "licence": "non-commercial research/educational only",
                "source_file": ann_path.name,
            },
        )
        store.write(validate(sample), image)
        samples.append(sample)
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise AdapterError(f"converted 0 samples from {root}; is the archive complete?")
    return samples
