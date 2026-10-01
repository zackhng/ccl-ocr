"""DDI-100 — 99,870 distorted document images with **character-level boxes**.

**Status: written to the documented format, NOT verified against the real archive.**

Two things make this dataset unusual, and both need stating up front.

*It is the only large real source with character boxes.* Every other real dataset here
stops at words or lines. That makes DDI-100 the one external check on the isolation
metric, which otherwise only ever runs on data we generated ourselves — a benchmark
grading its own homework.

*It is Russian.* The pages are Cyrillic documents, and this is an English-first build.
Cyrillic exercises CCL geometry fine — touching glyphs, broken strokes, stamps over
text, heavy distortion — but it is not our script, so these numbers **must be reported
separately and never folded into headline accuracy**. The adapter sets
``script="cyrillic"`` precisely so the report's per-script slice keeps them apart
automatically.

Source: https://github.com/machine-intelligence-laboratory/DDI-100

Documented layout per page::

    <part>/orig_texts/<id>.pkl    list of {"text": str, "box": [[x,y], ...]}
    <part>/gen_imgs/<id>.png      the distorted image

``box`` is a polygon in image coordinates. Unpickling is gated behind an explicit flag
because loading a pickle executes arbitrary code — a real consideration for a 100k-file
third-party archive.
"""

from __future__ import annotations

from pathlib import Path

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, GTChar, Sample

from .base import AdapterError, read_image, require_dir, validate

FETCH_HINT = "python scripts/fetch_ddi100.py  (large download; see the docstring)"


def _load_pickle(path: Path) -> list[dict]:
    import pickle

    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        raise AdapterError(
            f"{path}: expected a list of annotation dicts, got {type(data).__name__}. "
            "This adapter is UNVERIFIED -- confirm the layout before using the numbers."
        )
    return data


def _box(entry: dict) -> BBox | None:
    raw = entry.get("box")
    if raw is None:
        return None
    try:
        return BBox.from_points(raw)
    except (TypeError, IndexError, ValueError):
        return None


def convert(
    raw_dir: str | Path,
    store: BenchmarkStore,
    limit: int | None = None,
    allow_pickle: bool = False,
) -> list[Sample]:
    root = require_dir(raw_dir, "DDI-100", FETCH_HINT)
    if not allow_pickle:
        raise AdapterError(
            "DDI-100 annotations are Python pickles, and unpickling executes arbitrary "
            "code. Pass allow_pickle=True (or --allow-pickle) once you are satisfied "
            "with the archive's provenance."
        )

    pickles = sorted(root.rglob("orig_texts/*.pkl"))
    if not pickles:
        raise AdapterError(f"no orig_texts/*.pkl under {root}. {FETCH_HINT}")

    samples: list[Sample] = []
    for pkl in pickles:
        image_path = pkl.parents[1] / "gen_imgs" / f"{pkl.stem}.png"
        if not image_path.exists():
            continue

        entries = _load_pickle(pkl)
        chars: list[GTChar] = []
        words: list[GTBox] = []
        for entry in entries:
            box = _box(entry)
            text = str(entry.get("text", ""))
            if box is None or not text:
                continue
            if len(text) == 1:
                chars.append(GTChar(box, text))
            else:
                words.append(GTBox(box, text=text))

        sample = Sample(
            sample_id=f"ddi100_{pkl.stem}",
            source="ddi100",
            capture="scan",
            # Not 'latin'. The per-script slice must keep these out of the headline.
            script="cyrillic",
            dpi=300,
            text=" ".join(w.text for w in words),
            lines=[],
            words=words,
            chars=chars or None,
            pii=[],
            regions_nontext=[],
            meta={
                "template": "plain_cyrillic",
                "gt_complete": True,
                "adapter_status": "UNVERIFIED -- written to spec, never run on the real archive",
                "script_note": "Cyrillic: geometry stress test only, not headline accuracy",
                "licence": "see the DDI-100 repository",
            },
        )
        store.write(validate(sample), read_image(image_path))
        samples.append(sample)
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise AdapterError(
            f"converted 0 samples from {root}; layout may differ -- this adapter is UNVERIFIED."
        )
    return samples
