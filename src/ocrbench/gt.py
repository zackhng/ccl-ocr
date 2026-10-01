"""The unified ground-truth format.

Every source — SROIE, FUNSD, cheques, MIDV, DDI-100, our own synthetic generator —
is normalised to this one schema by an adapter, so the runner and the metrics never
learn where a sample came from. That is what makes "slice the results by capture mode"
a one-liner instead of a per-dataset special case.

The one field that needs care is :attr:`Sample.chars`. It is ``None`` whenever the
source does not annotate individual characters, which is most real datasets. Metrics
must therefore *always* have a word-level fallback — a metric that silently skips
samples with no char ground truth would quietly compute its headline number over
synthetic data only, which is exactly the self-congratulating benchmark we are trying
not to build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ocr.types import BBox

CAPTURE_MODES = ("scan", "photo", "digital")
"""``digital`` means born-digital (rendered, never through a lens or a sensor)."""

PII_TYPES = ("nric", "passport", "account", "name", "address", "dob", "phone", "face", "signature")

NONTEXT_TYPES = ("photo", "logo", "signature", "rule", "frame", "barcode", "stamp")


@dataclass(slots=True)
class GTBox:
    bbox: BBox
    text: str = ""
    type: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bbox": self.bbox.as_list()}
        if self.text:
            out["text"] = self.text
        if self.type:
            out["type"] = self.type
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GTBox:
        return cls(BBox.from_list(d["bbox"]), d.get("text", ""), d.get("type", ""))


@dataclass(slots=True)
class GTChar:
    bbox: BBox
    char: str

    def to_json(self) -> dict[str, Any]:
        return {"bbox": self.bbox.as_list(), "char": self.char}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GTChar:
        return cls(BBox.from_list(d["bbox"]), d["char"])


@dataclass(slots=True)
class Sample:
    sample_id: str
    source: str
    capture: str
    script: str
    dpi: int
    text: str = ""
    lines: list[GTBox] = field(default_factory=list)
    words: list[GTBox] = field(default_factory=list)
    chars: list[GTChar] | None = None
    pii: list[GTBox] = field(default_factory=list)
    regions_nontext: list[GTBox] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_char_gt(self) -> bool:
        return bool(self.chars)

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "capture": self.capture,
            "script": self.script,
            "dpi": self.dpi,
            "text": self.text,
            "lines": [b.to_json() for b in self.lines],
            "words": [b.to_json() for b in self.words],
            "chars": None if self.chars is None else [c.to_json() for c in self.chars],
            "pii": [b.to_json() for b in self.pii],
            "regions_nontext": [b.to_json() for b in self.regions_nontext],
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Sample:
        chars = d.get("chars")
        return cls(
            sample_id=d["sample_id"],
            source=d["source"],
            capture=d["capture"],
            script=d["script"],
            dpi=int(d.get("dpi", 0)),
            text=d.get("text", ""),
            lines=[GTBox.from_json(x) for x in d.get("lines", [])],
            words=[GTBox.from_json(x) for x in d.get("words", [])],
            chars=None if chars is None else [GTChar.from_json(x) for x in chars],
            pii=[GTBox.from_json(x) for x in d.get("pii", [])],
            regions_nontext=[GTBox.from_json(x) for x in d.get("regions_nontext", [])],
            meta=d.get("meta", {}),
        )


class BenchmarkStore:
    """On-disk benchmark: ``images/<id>.png`` beside ``gt/<id>.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.gt_dir = self.root / "gt"

    def ensure_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, sample_id: str) -> Path:
        return self.images_dir / f"{sample_id}.png"

    def gt_path(self, sample_id: str) -> Path:
        return self.gt_dir / f"{sample_id}.json"

    def write(self, sample: Sample, image=None) -> None:
        """Persist ground truth, and the image if one is supplied.

        Adapters for datasets we may not redistribute call this with ``image=None`` and
        leave the pixels wherever they were downloaded.
        """
        self.ensure_dirs()
        self.gt_path(sample.sample_id).write_text(
            json.dumps(sample.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if image is not None:
            import cv2

            ok, buf = cv2.imencode(".png", image)
            if not ok:
                raise ValueError(f"could not encode image for {sample.sample_id}")
            self.image_path(sample.sample_id).write_bytes(buf.tobytes())

    def read(self, sample_id: str) -> Sample:
        return Sample.from_json(json.loads(self.gt_path(sample_id).read_text(encoding="utf-8")))

    def sample_ids(self) -> list[str]:
        if not self.gt_dir.exists():
            return []
        return sorted(p.stem for p in self.gt_dir.glob("*.json"))

    def iter_samples(self) -> Iterator[Sample]:
        for sid in self.sample_ids():
            yield self.read(sid)

    # -- manifest -------------------------------------------------------------
    # Pixels are gitignored (size, and redistribution terms on the real sources).
    # The manifest is committed so a benchmark run is reproducible from a list of
    # sample ids plus the fetch scripts.

    def write_manifest(self) -> Path:
        entries = []
        for s in self.iter_samples():
            entries.append(
                {
                    "sample_id": s.sample_id,
                    "source": s.source,
                    "capture": s.capture,
                    "script": s.script,
                    "dpi": s.dpi,
                    "has_char_gt": s.has_char_gt,
                    "n_words": len(s.words),
                    "image_present": self.image_path(s.sample_id).exists(),
                }
            )
        path = self.root / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"samples": entries}, indent=1), encoding="utf-8")
        return path
