"""Shared adapter plumbing.

An adapter's whole job is to turn one dataset's idiosyncratic annotation into
:class:`~ocrbench.gt.Sample`, and to be *explicit about what it could not supply*.
That second half matters more than it sounds: if an adapter quietly emits empty word
lists, the sample silently contributes nothing to the metrics while still being counted
in "150 samples benchmarked". Hence :func:`validate`, which refuses to write a sample
that carries no usable annotation at all.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocrbench.gt import Sample


class AdapterError(RuntimeError):
    """Raised when a raw dataset is missing or does not match its documented layout.

    Deliberately loud. The alternative — returning zero samples — looks identical to
    "this dataset contributed nothing interesting" in the final report.
    """


def require_dir(path: str | Path, what: str, fetch_hint: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise AdapterError(f"{what} not found at {p}.\n  Fetch it first: {fetch_hint}")
    return p


def read_image(path: str | Path) -> np.ndarray:
    """imdecode-based read: tolerates non-ASCII paths on Windows."""
    buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise AdapterError(f"could not decode image: {path}")
    return image


def validate(sample: Sample) -> Sample:
    if not (sample.chars or sample.words or sample.lines or sample.regions_nontext):
        raise AdapterError(
            f"{sample.sample_id}: no usable annotation. An adapter must emit at least "
            "one of chars/words/lines/regions, or the sample inflates the benchmark "
            "count while measuring nothing."
        )
    return sample


def flatten_text(sample: Sample) -> str:
    if sample.lines:
        return "\n".join(l.text for l in sample.lines if l.text)
    return " ".join(w.text for w in sample.words if w.text)
