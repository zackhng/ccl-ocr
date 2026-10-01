"""MIDV-2020 — 1000 mock identity documents (2000 scans, 1000 photos, 1000 clips).

**Status: written to the documented format, NOT verified against the real archive.**
The dataset is 124 GB behind a Google Form and an sFTP server at the University of La
Rochelle, so this adapter has not been run on real data. It parses VIA (VGG Image
Annotator) polygon JSON, which is what the MIDV releases use, and raises
:class:`AdapterError` with the actual keys it found when the layout differs. Confirm the
layout before trusting any number this produces.

Access: https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html (form + licence acceptance)

What it provides, and does not:

* **Does**: document quadrangle and face quadrangle per image — real ID-card geometry,
  and a genuine test of whether the filter routes a portrait photo to ``BLOB``.
* **Does**: ideal text field *values*, which become ``Sample.text``.
* **Does not**: boxes for those values in image coordinates. MIDV gives field geometry
  in *template* coordinates; projecting it through the document quadrangle homography is
  possible but is reconstruction, not ground truth, so it is not done here.

Consequence: like the cheque adapter, these samples carry no line or word boxes and are
absent from the region-coverage table by design. They contribute latency, routing, and
face-region ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, Sample

from .base import AdapterError, read_image, require_dir, validate

FETCH_HINT = "see scripts/fetch_midv.py -- access is gated, this cannot be automated"

FACE_KEYS = {"face", "photo", "portrait"}
DOCUMENT_KEYS = {"document", "doc", "card"}


def _polygon_bbox(shape: dict) -> BBox | None:
    xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
    if xs and ys:
        return BBox.from_points(list(zip(xs, ys)))
    if {"x", "y", "width", "height"} <= shape.keys():
        return BBox(int(shape["x"]), int(shape["y"]), int(shape["width"]), int(shape["height"]))
    return None


def _regions(data: dict) -> list[dict]:
    """Pull the region list out of either VIA layout.

    VIA has shipped two shapes over the years: a top-level ``regions`` list, and a
    ``_via_img_metadata`` map keyed by filename. Both appear in the wild.
    """
    if isinstance(data.get("regions"), list):
        return data["regions"]
    meta = data.get("_via_img_metadata")
    if isinstance(meta, dict):
        for entry in meta.values():
            if isinstance(entry, dict) and isinstance(entry.get("regions"), list):
                return entry["regions"]
    raise AdapterError(
        "unrecognised MIDV annotation layout; expected VIA JSON with 'regions' or "
        f"'_via_img_metadata'. Top-level keys found: {sorted(data)[:12]}"
    )


def sample_from_annotation(ann_path: Path, image_path: Path, doc_type: str) -> Sample:
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    pii: list[GTBox] = []
    nontext: list[GTBox] = []
    field_values: list[str] = []

    for region in _regions(data):
        shape = region.get("shape_attributes", {})
        attrs = region.get("region_attributes", {})
        name = str(attrs.get("field_name") or attrs.get("name") or "").lower()
        box = _polygon_bbox(shape)
        if box is None:
            continue
        if name in FACE_KEYS:
            nontext.append(GTBox(box, type="photo"))
            pii.append(GTBox(box, type="face"))
        elif name in DOCUMENT_KEYS:
            nontext.append(GTBox(box, type="frame"))
        else:
            value = str(attrs.get("value") or "")
            if value:
                field_values.append(value)
            pii.append(GTBox(box, text=value, type="name" if "name" in name else "passport"))

    return Sample(
        sample_id=f"midv_{doc_type}_{ann_path.stem}",
        source="midv",
        capture="photo" if "photo" in str(image_path).lower() else "scan",
        script="latin",
        dpi=300,
        text="\n".join(field_values),
        lines=[],
        words=[],
        chars=None,
        pii=pii,
        regions_nontext=nontext,
        meta={
            "template": "id_card",
            "gt_complete": False,
            "doc_type": doc_type,
            "adapter_status": "UNVERIFIED -- written to spec, never run on the real archive",
            "granularity_note": "document/face quadrangles only; no text geometry",
            "licence": "MIDV-2020 licence, accepted via the access form",
        },
    )


def convert(raw_dir: str | Path, store: BenchmarkStore, limit: int | None = None) -> list[Sample]:
    root = require_dir(raw_dir, "MIDV-2020", FETCH_HINT)
    ann_paths = sorted(root.rglob("*.json"))
    if not ann_paths:
        raise AdapterError(f"no annotation JSON under {root}. {FETCH_HINT}")

    samples: list[Sample] = []
    for ann_path in ann_paths:
        image_path = next(
            (p for ext in (".jpg", ".png", ".tif")
             for p in root.rglob(f"{ann_path.stem}{ext}")),
            None,
        )
        if image_path is None:
            continue
        doc_type = ann_path.parent.name
        sample = sample_from_annotation(ann_path, image_path, doc_type)
        store.write(validate(sample), read_image(image_path))
        samples.append(sample)
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise AdapterError(
            f"converted 0 samples from {root}. Either the archive is incomplete or the "
            "layout differs from the documented one -- this adapter is UNVERIFIED."
        )
    return samples
