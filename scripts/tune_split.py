"""Sweep the merge-splitter's thresholds on a *held-out* synthetic set.

The published benchmark (seed 7) is the evaluation set. Tuning on it would report a
number fitted to it, so the splitter is tuned here, on a second synthetic set from the
same generator with a different seed, and the published set is then run once.

    # build the held-out set once (never committed; regenerates byte-identically)
    uv run python -m ocrbench.cli synth --count 150 --seed 11 --out bench_tune

    uv run python scripts/tune_split.py --bench bench_tune

Prints isolation / merge / miss / over-segmentation for photos and scans, plus
small-mark retention and the P50 of the whole pipeline, one row per configuration.
Only character-level metrics are used, so only samples with character GT count.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.config import DEFAULT_CONFIG, PipelineConfig  # noqa: E402
from ocr.engine import CCLEngine, load_image  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402
from ocrbench.metrics import char_metrics  # noqa: E402
from ocrbench.runner import aggregate_chars  # noqa: E402

GRID = {
    "small_block_ratio": (0.4, 0.5, 0.7),
    "suspect_width_ratio": (1.2, 1.3, 1.6),
    "min_part_height": (0.4, 0.5, 0.65),
}
COLUMN_GRID = {
    "cut_max_ink": (0.2, 0.3, 0.45),
    "min_piece_width": (0.5, 0.65, 0.8),
}


def evaluate(cfg: PipelineConfig, data) -> dict:
    engine = CCLEngine(cfg)
    by_capture: dict[str, list] = {"photo": [], "scan": []}
    totals = []
    for sample, image in data:
        result = engine.run(image)
        by_capture.setdefault(sample.capture, []).append(char_metrics(sample, result))
        totals.append(result.timings_ms["total"])
    out = {"p50": float(np.median(totals))}
    for capture, metrics in by_capture.items():
        out[capture] = aggregate_chars(metrics)
    return out


def row(label: str, r: dict) -> str:
    p, s = r["photo"], r["scan"]
    return (
        f"{label:52} photo iso {p.isolation_recall:5.1%} mrg {p.merge_rate:5.1%} "
        f"miss {p.miss_rate:5.1%} ovs {p.over_segmentation_rate:4.1%} small {p.small_retention or 0:5.1%} | "
        f"scan iso {s.isolation_recall:5.1%} mrg {s.merge_rate:5.1%} miss {s.miss_rate:5.1%} | "
        f"p50 {r['p50']:.0f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default="bench_tune")
    parser.add_argument("--columns", action="store_true",
                        help="also sweep the column cut, at the best re-threshold setting")
    args = parser.parse_args()

    if Path(args.bench).resolve() == Path("bench").resolve():
        print("refusing to tune on the published benchmark; use the held-out set", file=sys.stderr)
        return 2

    store = BenchmarkStore(args.bench)
    data = [(s, load_image(str(store.image_path(s.sample_id))))
            for s in (store.read(i) for i in store.sample_ids()) if s.has_char_gt]
    print(f"{len(data)} samples with character GT from {store.root}\n", flush=True)

    base = DEFAULT_CONFIG
    print(row("split disabled (Phase 0-2)", evaluate(replace(base, split=replace(base.split, enabled=False)), data)), flush=True)

    best_label, best_cfg, best_score = "", base.split, -1.0
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        split = replace(base.split, **dict(zip(keys, values)))
        r = evaluate(replace(base, split=split), data)
        label = " ".join(f"{k}={v}" for k, v in zip(keys, values))
        print(row(label, r), flush=True)
        # Photos are the target, but a setting that buys photos by losing scans is not
        # a win: score on the two isolation recalls together.
        score = r["photo"].isolation_recall + r["scan"].isolation_recall
        if score > best_score:
            best_label, best_cfg, best_score = label, split, score
    print(f"\nbest re-threshold setting: {best_label}\n", flush=True)

    if args.columns:
        ckeys = list(COLUMN_GRID)
        for values in itertools.product(*(COLUMN_GRID[k] for k in ckeys)):
            split = replace(best_cfg, split_touching_columns=True, **dict(zip(ckeys, values)))
            r = evaluate(replace(base, split=split), data)
            print(row("columns " + " ".join(f"{k}={v}" for k, v in zip(ckeys, values)), r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
