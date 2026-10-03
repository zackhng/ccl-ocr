"""Real line crops for the Latin sequence recogniser (CRNN), from annotated documents.

    uv run python scripts/build_real_lines.py --store data/real/train --out data/lines/real_latin

The CRNN reads a whole line crop, so real documents need no character alignment: every
annotated line *is* a training example. Datasets that annotate lines (SROIE, CORD) give
them directly; word-only datasets (FUNSD, XFUND) have their words grouped into line
*segments* first — same row, gap under 2.5 word heights — matching the segments CCL's
grouping (Phase 5) produces at inference, so training and inference crops look alike.

Decoder-tuning documents (``ocrbench.glyphs.is_tune_doc``) are excluded. Output uses the
strip format of ``scripts/build_lines.py`` (40 px high), so ``train_seq.py`` reads both.
``--max-per-source`` caps very large sources (XFUND has ~900 words per form) to keep
the dataset — held in RAM during training — bounded.

``profiles`` holds each line's source: SROIE transcripts are upper-cased whatever the
print, so the trainer scores those lines case-insensitively (``ocrbench.metrics.
CASE_FOLDED_SOURCES``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.engine import load_image  # noqa: E402
from ocr.recog.charset import normalise  # noqa: E402
from ocr.recog.seq import LINE_HEIGHT, MAX_WIDTH  # noqa: E402
from ocr.types import BBox  # noqa: E402
from ocrbench.glyphs import is_tune_doc  # noqa: E402
from ocrbench.gt import BenchmarkStore, GTBox  # noqa: E402


def segments_from_words(words: list[GTBox], gap_ratio: float = 2.5) -> list[GTBox]:
    """Group word boxes into line segments: same row, horizontally close."""
    words = [w for w in words if w.text.strip()]
    words.sort(key=lambda w: (w.bbox.cy, w.bbox.x))
    rows: list[list[GTBox]] = []
    for w in words:
        for row in rows:
            ref = row[-1].bbox
            if abs(w.bbox.cy - ref.cy) <= 0.5 * min(w.bbox.h, ref.h):
                row.append(w)
                break
        else:
            rows.append([w])
    out = []
    for row in rows:
        row.sort(key=lambda w: w.bbox.x)
        seg = [row[0]]
        for w in row[1:]:
            prev = seg[-1].bbox
            if w.bbox.x - prev.x2 <= gap_ratio * max(prev.h, w.bbox.h):
                seg.append(w)
            else:
                out.append(_join(seg))
                seg = [w]
        out.append(_join(seg))
    return out


def _join(ws: list[GTBox]) -> GTBox:
    box = ws[0].bbox
    for w in ws[1:]:
        box = box.union(w.bbox)
    return GTBox(box, text=" ".join(w.text for w in ws))


def crop_line(gray: np.ndarray, b: BBox) -> np.ndarray | None:
    m = max(2, b.h // 6)
    crop = gray[max(0, b.y - m): b.y2 + m, max(0, b.x - m): b.x2 + m]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    new_w = max(8, min(MAX_WIDTH, int(round(crop.shape[1] * LINE_HEIGHT / crop.shape[0]))))
    return cv2.resize(crop, (new_w, LINE_HEIGHT), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="data/real/train")
    ap.add_argument("--out", default="data/lines/real_latin")
    ap.add_argument("--max-per-source", type=int, default=60000)
    ap.add_argument("--shard", type=int, default=10000)
    args = ap.parse_args()

    store = BenchmarkStore(args.store)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    ids = [s for s in store.sample_ids() if not is_tune_doc(s)]
    rng.shuffle(ids)  # so a per-source cap samples documents evenly

    per_source: Counter = Counter()
    buf, shard, skipped = [], 0, Counter()

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        widths = np.array([c.shape[1] for c, _, _ in buf], np.int32)
        np.savez_compressed(
            out / f"shard_{shard:04d}.npz", strip=np.concatenate([c for c, _, _ in buf], axis=1),
            widths=widths, offsets=np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64),
            texts=np.array([t for _, t, _ in buf], dtype=object), labels=np.array([t for _, t, _ in buf], dtype=object),
            profiles=np.array([src for _, _, src in buf], dtype=object), script="latin")
        shard += 1
        buf = []

    for k, sid in enumerate(ids, 1):
        sample = store.read(sid)
        if per_source[sample.source] >= args.max_per_source:
            continue
        units = sample.lines or segments_from_words(sample.words)
        gray = cv2.cvtColor(load_image(str(store.image_path(sid))), cv2.COLOR_BGR2GRAY)
        for u in units:
            text = normalise(unicodedata.normalize("NFC", u.text))
            if not text or len(text) > 120:
                skipped["empty_or_long"] += 1
                continue
            crop = crop_line(gray, u.bbox)
            if crop is None:
                skipped["tiny"] += 1
                continue
            buf.append((crop, text, sample.source))
            per_source[sample.source] += 1
            if len(buf) >= args.shard:
                flush()
        if k % 200 == 0:
            print(f"  {k}/{len(ids)} docs, {sum(per_source.values())} lines {dict(per_source)}", flush=True)
    flush()
    summary = {"lines": sum(per_source.values()), "per_source": dict(per_source), "skipped": dict(skipped)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
