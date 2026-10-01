"""Indian bank cheques — ``jaganadhg/cheque-synthetic-images`` (Apache-2.0).

295 cheques composited from real IDRBT field crops onto bank-specific canvases, with
six field boxes each: ``date, amount, ifsc, acno, sign, name``.

**What these samples can and cannot measure.** The annotations are *field regions*, not
text lines — the ``name`` box routinely spans the full 2365 px width of the cheque to
cover a payee line with a dozen characters in it. Scoring region coverage against a box
that is mostly empty paper would produce a number that looks like a failure and means
nothing. So these samples deliberately emit **no** ``lines`` or ``words``: they
contribute latency, component-routing behaviour, and PII/signature region ground truth,
and they are correctly absent from the region-coverage table.

What they are genuinely good for is the one thing synthetic data cannot fake — real
cheque paper, real print, real scanner artefacts, real signatures — against which the
filter's BLOB/RULE routing can be checked.

The companion dataset ``jaganadhg/cheque-field-annotations`` carries the same six fields
for the 112 *real* IDRBT cheques but ships no images; those must be obtained from
idrbt.ac.in separately, and IDRBT states no licence. Not wired up here for that reason.

Reads the local cache written by ``scripts/fetch_cheques.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, Sample

from .base import AdapterError, read_image, require_dir, validate

FETCH_HINT = "python scripts/fetch_cheques.py --limit 40"

FIELDS = ("date", "amount", "ifsc", "acno", "sign", "name")

# How each field maps into our schema. 'sign' is the only one that is not text.
PII_TYPE = {
    "name": "name",
    "acno": "account",
    "amount": "account",
    "ifsc": "account",
    "date": "dob",
    "sign": "signature",
}


def _bbox(d: dict | None) -> BBox | None:
    if not d:
        return None
    x0, y0 = int(d["xmin"]), int(d["ymin"])
    x1, y1 = int(d["xmax"]), int(d["ymax"])
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1 - x0, y1 - y0)


def sample_from_row(row: dict, image_path: Path) -> Sample:
    pii: list[GTBox] = []
    nontext: list[GTBox] = []

    for field in FIELDS:
        box = _bbox(row.get(field))
        if box is None:
            continue
        if field == "sign":
            nontext.append(GTBox(box, type="signature"))
            pii.append(GTBox(box, type="signature"))
        else:
            pii.append(GTBox(box, type=PII_TYPE[field]))

    if not nontext and not pii:
        raise AdapterError(
            f"{row.get('image_id')}: no field boxes parsed -- the dataset schema has "
            f"changed; expected keys {FIELDS} with xmin/ymin/xmax/ymax"
        )

    return Sample(
        sample_id=f"cheque_{row['image_id']}",
        source="cheque",
        capture="scan",
        script="latin",
        dpi=300,  # IDRBT scans are ~2365x1080 at roughly 300 DPI.
        text="",
        lines=[],
        words=[],
        chars=None,
        pii=pii,
        regions_nontext=nontext,
        meta={
            "template": "cheque",
            "gt_complete": False,
            "bank": row.get("bank", ""),
            "granularity_note": (
                "field regions only, much larger than the text they contain -- "
                "intentionally contributes no line/word ground truth"
            ),
            "licence": "Apache-2.0 (jaganadhg/cheque-synthetic-images)",
        },
    )


def convert(raw_dir: str | Path, store: BenchmarkStore, limit: int | None = None) -> list[Sample]:
    root = require_dir(raw_dir, "cheque cache", FETCH_HINT)
    rows_path = root / "rows.jsonl"
    if not rows_path.exists():
        raise AdapterError(f"{rows_path} missing. Fetch it first: {FETCH_HINT}")

    samples: list[Sample] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = root / "images" / f"{row['image_id']}.jpg"
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
