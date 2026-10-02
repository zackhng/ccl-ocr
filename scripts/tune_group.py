"""Sweep the grouping thresholds on the *held-out* synthetic set (Phase 5).

Same discipline as ``tune_split.py``: tune on ``bench_tune`` (seed 11), report on the
published benchmark (seed 7) once.

    uv run python scripts/tune_group.py --bench bench_tune

The engine runs once per image; grouping is then re-run on the same filtered
components for every configuration, since it depends on nothing else. Scores are line
and word F1 by capture, plus the grouping stage's own time.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.config import DEFAULT_CONFIG, GroupConfig  # noqa: E402
from ocr.engine import CCLEngine, load_image  # noqa: E402
from ocr.group import group_components  # noqa: E402
from ocr.types import PageResult  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402
from ocrbench.metrics import grouping_metrics  # noqa: E402
from ocrbench.runner import aggregate_grouping  # noqa: E402

GRID = {
    "line_gap_ratio": (2.0, 2.5, 3.0),
    "word_gap_ratio": (0.25, 0.3, 0.4),
    "word_gap_median_factor": (0.0, 1.5, 2.0, 2.5),
    "core_fraction": (0.5, 0.6),
}
JUNK_GRID = {
    "junk_max_height_ratio": (0.6, 0.7),
    "junk_single_height_ratio": (0.8, 0.9),
    "junk_median_height_ratio": (0.5, 0.6),
    "junk_barcode_aspect": (0.0, 0.15, 0.25),
}


def evaluate(cfg: GroupConfig, data) -> dict:
    by_cap: dict[str, list] = {}
    ms = []
    for sample, result, glyph_h in data:
        t0 = time.perf_counter()
        lines, _, _ = group_components(result.components, result.width, result.height, glyph_h, cfg)
        ms.append((time.perf_counter() - t0) * 1000)
        page = PageResult(width=result.width, height=result.height,
                          components=result.components, lines=lines)
        by_cap.setdefault(sample.capture, []).append(grouping_metrics(sample, page))
    out = {"p95": float(np.percentile(ms, 95))}
    for cap, ms_ in by_cap.items():
        agg = aggregate_grouping(ms_)
        out[cap] = (agg.line_prf[2] or 0.0, agg.word_prf[2] or 0.0)
    return out


def row(label: str, r: dict) -> str:
    return (f"{label:70} photo line {r['photo'][0]:5.1%} word {r['photo'][1]:5.1%} | "
            f"scan line {r['scan'][0]:5.1%} word {r['scan'][1]:5.1%} | group P95 {r['p95']:.1f} ms")


def score(r: dict) -> float:
    return sum(r[c][0] + r[c][1] for c in ("photo", "scan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="bench_tune")
    args = ap.parse_args()
    if Path(args.bench).resolve() == Path("bench").resolve():
        print("refusing to tune on the published benchmark; use the held-out set", file=sys.stderr)
        return 2

    store = BenchmarkStore(args.bench)
    engine = CCLEngine(replace(DEFAULT_CONFIG, group=replace(DEFAULT_CONFIG.group, enabled=False)))
    data = []
    for sid in store.sample_ids():
        s = store.read(sid)
        if not (s.lines or s.words):
            continue
        r = engine.run(load_image(str(store.image_path(sid))))
        # Components are in original coordinates here, so the glyph height is too.
        glyph_h = (r.meta["filter"].get("median_height") or 0.0) / (r.scale or 1.0)
        data.append((s, r, glyph_h))
    print(f"{len(data)} samples from {store.root}\n", flush=True)

    base = DEFAULT_CONFIG.group
    best, best_score = base, -1.0
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        cfg = replace(base, **dict(zip(keys, values)))
        r = evaluate(cfg, data)
        print(row(" ".join(f"{k}={v}" for k, v in zip(keys, values)), r), flush=True)
        if score(r) > best_score:
            best, best_score = cfg, score(r)
    print("\nbest gaps:", {k: getattr(best, k) for k in GRID}, "\n")

    keys = list(JUNK_GRID)
    best_j, best_score = best, -1.0
    for values in itertools.product(*(JUNK_GRID[k] for k in keys)):
        cfg = replace(best, **dict(zip(keys, values)))
        r = evaluate(cfg, data)
        print(row(" ".join(f"{k}={v}" for k, v in zip(keys, values)), r), flush=True)
        if score(r) > best_score:
            best_j, best_score = cfg, score(r)
    print("\nbest:", {k: getattr(best_j, k) for k in list(GRID) + list(JUNK_GRID)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
