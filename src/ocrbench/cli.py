"""Benchmark CLI.

    python -m ocrbench.cli fonts                        # what can this machine render?
    python -m ocrbench.cli synth --count 150            # build the synthetic set
    python -m ocrbench.cli run --repeats 5              # measure
    python -m ocrbench.cli overlays --limit 30          # look at the failures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ocr.config import DEFAULT_CONFIG, PipelineConfig
from ocr.engine import CCLEngine, load_image
from ocr.visualize import binary_preview, draw_components, save_image, side_by_side

from . import report, runner
from .gt import BenchmarkStore
from .synth import fonts
from .synth.generate import DEFAULT_STRATA, generate


def _config(path: str | None) -> PipelineConfig:
    if not path:
        return DEFAULT_CONFIG
    return PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_fonts(args: argparse.Namespace) -> int:
    print(fonts.report())
    print(f"\nusable scripts: {', '.join(fonts.supported_scripts())}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    store = BenchmarkStore(args.out)
    print(f"generating {args.count} synthetic samples into {store.root} (seed {args.seed})")
    samples = generate(store, args.count, seed=args.seed, strata=DEFAULT_STRATA)

    by_template: dict[str, int] = {}
    by_capture: dict[str, int] = {}
    with_chars = 0
    for s in samples:
        by_template[str(s.meta.get("template"))] = by_template.get(str(s.meta.get("template")), 0) + 1
        by_capture[s.capture] = by_capture.get(s.capture, 0) + 1
        with_chars += bool(s.has_char_gt)

    manifest = store.write_manifest()
    print(f"\n  templates: {by_template}")
    print(f"  capture:   {by_capture}")
    print(f"  with character-level GT: {with_chars}/{len(samples)}")
    print(f"  manifest:  {manifest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = BenchmarkStore(args.bench)
    rc = runner.RunConfig(
        repeats=args.repeats,
        warmup=args.warmup,
        limit=args.limit,
        sources=tuple(args.sources or ()),
        captures=tuple(args.captures or ()),
        scripts=tuple(args.scripts or ()),
    )
    cfg = _config(args.config)
    print(f"running {CCLEngine(cfg).name} over {store.root}")
    result = runner.run(store, engine=CCLEngine(cfg), run_config=rc, pipeline_config=cfg)

    out_dir = runner.save(result, args.results)
    summary_path = report.write(result, out_dir)

    lat = result.latency.get("total", {})
    print(f"\n  {len(result.samples)} samples")
    print(f"  end-to-end P50 {lat.get('p50', 0):.1f} ms | "
          f"P95 {lat.get('p95', 0):.1f} ms | P99 {lat.get('p99', 0):.1f} ms")

    chars = runner.aggregate_chars([s.chars for s in result.samples if s.chars is not None])
    if chars.n_gt:
        print(f"  isolation recall {chars.isolation_recall:.1%} over {chars.n_gt} GT chars "
              f"(over-seg {chars.over_segmentation_rate:.1%}, merged {chars.merge_rate:.1%}, "
              f"missed {chars.miss_rate:.1%})")
    words = runner.aggregate_words([s.words for s in result.samples])
    print(f"  word hit rate {words.hit_rate:.1%}, coverage {words.mean_coverage:.1%}")
    print(f"\n  {summary_path}")
    return 0


def cmd_overlays(args: argparse.Namespace) -> int:
    """Render overlays for a sample of the benchmark — the Phase 1 eyeball gate."""
    store = BenchmarkStore(args.bench)
    engine = CCLEngine(_config(args.config))
    ids = [sid for sid in store.sample_ids() if store.image_path(sid).exists()]
    if not ids:
        print(f"no images under {store.images_dir}", file=sys.stderr)
        return 1

    if args.limit and len(ids) > args.limit:
        # Spread across the benchmark rather than taking the first N, which would be
        # one template in alphabetical order.
        ids = random.Random(args.seed).sample(ids, args.limit)
        ids.sort()

    out_dir = Path(args.out)
    for sid in ids:
        image = load_image(str(store.image_path(sid)))
        result, debug = engine.run_with_debug(image)
        overlay = draw_components(image, result)
        panel = side_by_side(image, binary_preview(debug.binary), overlay) if args.stages else overlay
        save_image(out_dir / f"{sid}.png", panel)
    print(f"wrote {len(ids)} overlay(s) to {out_dir}")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    store = BenchmarkStore(args.bench)
    path = store.write_manifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [s["sample_id"] for s in data["samples"] if not s["image_present"]]
    print(f"{len(data['samples'])} samples -> {path}")
    if missing:
        print(f"  {len(missing)} without images (run the fetch scripts): {missing[:5]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocrbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench_arg = argparse.ArgumentParser(add_help=False)
    bench_arg.add_argument("--bench", default="bench", help="benchmark root")
    bench_arg.add_argument("--config", help="JSON PipelineConfig override")

    p = sub.add_parser("fonts", help="report renderable scripts")
    p.set_defaults(func=cmd_fonts)

    p = sub.add_parser("synth", help="generate synthetic samples")
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="bench")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("run", parents=[bench_arg], help="run the benchmark")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--sources", nargs="+")
    p.add_argument("--captures", nargs="+")
    p.add_argument("--scripts", nargs="+")
    p.add_argument("--results", default="bench/results")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("overlays", parents=[bench_arg], help="render overlays for inspection")
    p.add_argument("--out", default="bench/overlays")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stages", action="store_true")
    p.set_defaults(func=cmd_overlays)

    p = sub.add_parser("manifest", parents=[bench_arg], help="rewrite the manifest")
    p.set_defaults(func=cmd_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
