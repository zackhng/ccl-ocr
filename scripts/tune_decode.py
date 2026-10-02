"""Tune the decoder (LM weight, format bonus) on the reserved real tuning documents.

    uv run python scripts/tune_decode.py --store data/real/train

Uses only documents with ``ocrbench.glyphs.is_tune_doc`` — never trained on by the CNN,
never in the test benchmark. Each page is prepared once (CCL, crops, one CNN pass) and
re-decoded per configuration, so the grid costs decoding time only.

Reports CER and word F1 for every configuration, with lambda = 0 (CNN only) as the
baseline row, and the best setting by CER.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.engine import CCLEngine, load_image  # noqa: E402
from ocr.recog.decode import DecodeConfig  # noqa: E402
from ocr.recog.recognizer import default_recognizer  # noqa: E402
from ocr.types import PageResult  # noqa: E402
from ocrbench.glyphs import is_tune_doc  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402
from ocrbench.metrics import text_metrics  # noqa: E402
from ocrbench.runner import aggregate_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="data/real/train")
    ap.add_argument("--lm-weights", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--format-bonus", nargs="+", type=float, default=[0.0, 2.0, 4.0])
    args = ap.parse_args()

    store = BenchmarkStore(args.store)
    ids = [sid for sid in store.sample_ids() if is_tune_doc(sid)]
    rec = default_recognizer()
    engine = CCLEngine()
    pages = []
    for sid in ids:
        sample = store.read(sid)
        image = load_image(str(store.image_path(sid)))
        page = engine.run(image)
        pages.append((sample, page, rec.prepare(image, page)))
    print(f"{len(pages)} tuning documents from {store.root}\n", flush=True)

    best = None
    for lam, bonus in itertools.product(args.lm_weights, args.format_bonus):
        cfg = replace(rec.cfg.decode, lm_weight=lam, format_bonus=bonus)
        metrics = []
        for sample, page, prep in pages:
            promoted = rec.decode(prep, page, cfg, lm=rec.lm if lam > 0 else None)
            view = PageResult(width=page.width, height=page.height, components=page.components,
                              lines=page.lines + promoted)
            metrics.append(text_metrics(sample, view))
        agg = aggregate_text(metrics)
        print(f"  lm_weight {lam:4.2f}  format_bonus {bonus:3.1f}  CER {agg.cer:6.2%}  word F1 {agg.word_f1:6.2%}", flush=True)
        if best is None or agg.cer < best[0]:
            best = (agg.cer, lam, bonus, agg.word_f1)
    print(f"\nbest: lm_weight {best[1]} format_bonus {best[2]} -> CER {best[0]:.2%}, word F1 {best[3]:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
