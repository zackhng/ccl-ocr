"""Line images for the Phase 7 sequence recognisers and the script classifier.

    uv run python scripts/build_lines.py --script devanagari --lines 60000 --out data/lines/devanagari
    uv run python scripts/build_lines.py --script latin --lines 60000 --out data/lines/latin

Each line: text sampled from that script's corpus (``data/corpora/script_<s>.txt``;
Latin uses ``public.txt`` + ``financial.txt``), with Latin amounts, dates and codes
mixed in as real documents do; shaped and rasterised correctly
(``ocrbench.synth.shaped``: HarfBuzz + FreeType); degraded with the benchmark's capture
profiles; cropped to the line and resized to 40 px high.

Seeds are derived from ``--seed0`` and the line index; never 7 or 11 (the benchmark and
the held-out tuning seeds). Output: ``shard_XXXX.npz`` with a horizontal strip of all
crops (uint8, 40 x total_width), per-line offsets/widths, texts (logical) and labels
(visual order: RTL reversed — see ``ocr.recog.seq.logical_to_visual``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog.seq import LINE_HEIGHT, MAX_WIDTH, logical_to_visual  # noqa: E402

PROFILES = {"clean_scan": 0.2, "scan": 0.35, "photo": 0.3, "hard_photo": 0.15}
_S: dict = {}


def _init(script: str, corpora: str) -> None:
    root = Path(corpora)
    if script == "latin":
        lines = (root / "public.txt").read_text(encoding="utf-8").splitlines()
        lines += (root / "financial.txt").read_text(encoding="utf-8").splitlines()
    else:
        lines = (root / f"script_{script}.txt").read_text(encoding="utf-8").splitlines()
    _S["lines"] = lines
    _S["script"] = script
    from ocrbench.synth.fonts import fonts_for

    _S["fonts"] = [f for f in fonts_for(script if script != "latin" else "latin")]
    if script == "latin":
        _S["fonts"] += fonts_for("mono")


def _token(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return f"{rng.uniform(1, 999999):,.2f}"
    if kind < 0.55:
        return f"{rng.randrange(1, 29):02d}/{rng.randrange(1, 13):02d}/{rng.randrange(1990, 2030)}"
    if kind < 0.75:
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(rng.randrange(4, 12)))
    return rng.choice(["AED", "INR", "THB", "CNY", "RMB", "Rs.", "No.", "Tel", "A/C", "TOTAL", "Date"])


def _text(rng: random.Random) -> str:
    script = _S["script"]
    src = rng.choice(_S["lines"])
    if script == "han":
        n = rng.randrange(4, 26)
        start = rng.randrange(max(1, len(src) - n))
        text = src[start:start + n].strip()
    else:
        words = src.split()
        n = rng.randrange(1, 9)
        start = rng.randrange(max(1, len(words) - n))
        text = " ".join(words[start:start + n])
    if script != "latin" and rng.random() < 0.3:
        tok = _token(rng)
        text = f"{text} {tok}" if rng.random() < 0.5 else f"{tok} {text}"
    return text[:80]


def _one(seed: int):
    from PIL import Image

    from ocr.types import BBox
    from ocrbench.synth.degrade import PROFILES as DEG
    from ocrbench.synth.degrade import degrade
    from ocrbench.synth.fonts import covers
    from ocrbench.synth.shaped import draw_shaped, shape_line

    if seed in (7, 11):
        return None
    rng = random.Random(seed)
    text = _text(rng)
    if not text.strip():
        return None
    size = rng.choice([22, 26, 30, 34, 40, 48])
    fonts = rng.sample(_S["fonts"], len(_S["fonts"]))
    font = next((f for f in fonts if all(covers(f.at(size), c) for c in set(text) if not c.isspace())), None)
    if font is None:
        return None
    line = shape_line(text, 40, 30, font.path, size, font.index)
    h, w = line.mask.shape
    canvas_w, canvas_h = line.origin[0] + w + 40, line.origin[1] + h + 30
    bg = tuple(rng.randrange(225, 256) for _ in range(3))
    img = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw_shaped(img, line, fill=tuple(rng.randrange(0, 70) for _ in range(3)))
    box = BBox(line.origin[0], line.origin[1], w, h)
    profile = rng.choices(list(PROFILES), weights=list(PROFILES.values()))[0]
    res = degrade(np.array(img)[:, :, ::-1].copy(), [box], DEG[profile], rng)
    b = res.boxes[0]
    if b is None or b.w < 4 or b.h < 4:
        return None
    gray = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY)
    m = max(2, b.h // 6)
    crop = gray[max(0, b.y - m): b.y2 + m, max(0, b.x - m): b.x2 + m]
    new_w = max(8, min(MAX_WIDTH, int(round(crop.shape[1] * LINE_HEIGHT / max(1, crop.shape[0])))))
    crop = cv2.resize(crop, (new_w, LINE_HEIGHT), interpolation=cv2.INTER_AREA)
    return crop, text, logical_to_visual(text, _S["script"]), profile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True, choices=["latin", "han", "devanagari", "thai", "arabic"])
    ap.add_argument("--lines", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=10000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf, shard, kept = [], 0, 0

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        widths = np.array([c.shape[1] for c, *_ in buf], np.int32)
        np.savez_compressed(
            out / f"shard_{shard:04d}.npz", strip=np.concatenate([c for c, *_ in buf], axis=1),
            widths=widths, offsets=np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64),
            texts=np.array([t for _, t, _, _ in buf], dtype=object),
            labels=np.array([v for _, _, v, _ in buf], dtype=object),
            profiles=np.array([p for *_, p in buf], dtype=object), script=args.script)
        shard += 1
        buf = []

    seeds = range(args.seed0, args.seed0 + int(args.lines * 1.15))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.script, args.corpora)) as pool:
        for res in pool.map(_one, seeds, chunksize=64):
            if res is None:
                continue
            buf.append(res)
            kept += 1
            if len(buf) >= args.shard:
                flush()
                print(f"  {kept} lines", flush=True)
            if kept >= args.lines:
                break
    flush()
    (out / "summary.json").write_text(json.dumps({"script": args.script, "lines": kept}, indent=1))
    print(f"{args.script}: {kept} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
