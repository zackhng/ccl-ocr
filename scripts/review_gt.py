"""Semi-automatic ground-truth review for real samples that ship without transcripts.

The cheque and MIDV adapters deliberately emit no line or word boxes, because their
source annotations are coarse field regions. This tool is how those samples get real
text ground truth: bootstrap with an off-the-shelf OCR engine, correct by hand, apply.

    # 1. propose boxes + text, and render an overlay to check them against
    uv run python scripts/review_gt.py bootstrap --source cheque

    # 2. edit bench/review/<id>.json by hand -- fix text, delete junk, add misses
    #    (open bench/review/<id>.png alongside it)

    # 3. merge the corrections into the benchmark ground truth
    uv run python scripts/review_gt.py apply --source cheque

Bootstrapping needs an OCR engine that is not a dependency of this project:

    uv sync --extra baselines      # PaddlePaddle + PaddleOCR
    pip install pytesseract        # plus the Tesseract binary on PATH

``apply`` needs neither, so corrections can be reviewed on one machine and applied on
another. Reviewed samples are marked ``meta.gt_source = "reviewed"`` and
``meta.gt_complete = true`` — only then do they start counting toward the junk rate,
which would otherwise punish the pipeline for the dataset's missing annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from ocr.engine import load_image  # noqa: E402
from ocr.types import BBox  # noqa: E402
from ocrbench.gt import BenchmarkStore, GTBox, Sample  # noqa: E402


# ------------------------------------------------------------------ OCR backends


_PADDLE = None


def _paddle_boxes(image) -> list[tuple[BBox, str]] | None:
    # Reuses the Phase 6 baseline engine so there is one place that knows the
    # PaddleOCR 3.x API; constructed once because model loading takes seconds.
    global _PADDLE
    try:
        from ocr.paddle_engine import PaddleEngine
        if _PADDLE is None:
            _PADDLE = PaddleEngine()
    except ImportError:
        return None
    return [(ln.bbox, ln.text) for ln in _PADDLE.run(image).lines]


def _tesseract_boxes(image) -> list[tuple[BBox, str]] | None:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return None
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    out: list[tuple[BBox, str]] = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        out.append((
            BBox(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            text,
        ))
    return out


def bootstrap_boxes(image) -> tuple[list[tuple[BBox, str]], str]:
    for name, fn in (("paddleocr", _paddle_boxes), ("tesseract", _tesseract_boxes)):
        boxes = fn(image)
        if boxes is not None:
            return boxes, name
    raise SystemExit(
        "no OCR backend available for bootstrapping.\n"
        "  uv sync --extra baselines     (PaddleOCR)\n"
        "  pip install pytesseract       (plus the Tesseract binary on PATH)\n"
        "Or write bench/review/<id>.json by hand and run 'apply'."
    )


# --------------------------------------------------------------------- commands


def cmd_bootstrap(args: argparse.Namespace) -> int:
    store = BenchmarkStore(args.bench)
    review_dir = Path(args.bench) / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        sid for sid in store.sample_ids()
        if store.read(sid).source == args.source and store.image_path(sid).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"no samples with source={args.source!r} and an image present")
        return 1

    backend = ""
    for sid in todo:
        out_json = review_dir / f"{sid}.json"
        if out_json.exists() and not args.overwrite:
            continue
        image = load_image(str(store.image_path(sid)))
        boxes, backend = bootstrap_boxes(image)

        out_json.write_text(
            json.dumps(
                {
                    "sample_id": sid,
                    "bootstrapped_with": backend,
                    "instructions": (
                        "Fix 'text', delete wrong entries, add missed ones. "
                        "bbox is [x, y, w, h]. Then run: review_gt.py apply"
                    ),
                    "words": [{"bbox": b.as_list(), "text": t} for b, t in boxes],
                },
                indent=1, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        overlay = image.copy()
        for b, _ in boxes:
            cv2.rectangle(overlay, (b.x, b.y), (b.x2, b.y2), (0, 180, 0), 2)
        ok, buf = cv2.imencode(".png", overlay)
        if ok:
            (review_dir / f"{sid}.png").write_bytes(buf.tobytes())
        print(f"  {sid}: {len(boxes)} proposed")

    print(f"\nwrote proposals to {review_dir} (backend: {backend})")
    print("Correct the JSON files, then: review_gt.py apply --source " + args.source)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    store = BenchmarkStore(args.bench)
    review_dir = Path(args.bench) / "review"
    if not review_dir.exists():
        print(f"nothing to apply: {review_dir} does not exist")
        return 1

    applied = 0
    for path in sorted(review_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data["sample_id"]
        if not store.gt_path(sid).exists():
            print(f"  skip {sid}: no ground truth to merge into")
            continue
        sample: Sample = store.read(sid)
        if args.source and sample.source != args.source:
            continue

        words = [
            GTBox(BBox.from_list(w["bbox"]), text=w.get("text", ""))
            for w in data.get("words", [])
            if w.get("text", "").strip()
        ]
        if not words:
            print(f"  skip {sid}: no words left after correction")
            continue

        sample.words = words
        sample.text = " ".join(w.text for w in words)
        sample.meta["gt_source"] = "reviewed"
        sample.meta["gt_complete"] = True
        sample.meta["bootstrapped_with"] = data.get("bootstrapped_with", "")
        store.write(sample)
        applied += 1
        print(f"  {sid}: {len(words)} words")

    store.write_manifest()
    print(f"\napplied {applied} reviewed sample(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap", help="propose boxes with an OCR engine")
    p.add_argument("--source", required=True, help="e.g. cheque, midv")
    p.add_argument("--bench", default="bench")
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("apply", help="merge corrected proposals into ground truth")
    p.add_argument("--source", help="restrict to one source")
    p.add_argument("--bench", default="bench")
    p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
