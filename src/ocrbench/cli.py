"""Benchmark CLI.

    python -m ocrbench.cli fonts                        # what can this machine render?
    python -m ocrbench.cli synth --count 150            # build the synthetic set
    python -m ocrbench.cli run --repeats 5              # measure
    python -m ocrbench.cli compare --engines ccl paddle-det paddle   # Phase 6 bake-off
    python -m ocrbench.cli report --runs <dir> <dir> --out <dir>     # re-compare saved runs
    python -m ocrbench.cli overlays --limit 30          # look at the failures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from ocr.config import DEFAULT_CONFIG, PipelineConfig
from ocr.engine import CCLEngine, Engine, load_image
from ocr.visualize import binary_preview, draw_components, save_image, side_by_side

from . import compare, report, runner
from .gt import BenchmarkStore
from .synth import fonts
from .synth.generate import DEFAULT_STRATA, generate


def _config(path: str | None) -> PipelineConfig:
    if not path:
        return DEFAULT_CONFIG
    return PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


ENGINES = ("ccl", "ccl-cnn", "ccl-cnn-nolm", "ccl-crnn", "ccl-glyph", "paddle", "paddle-det")


def make_engine(name: str, cfg: PipelineConfig, threads: int | None) -> Engine:
    """Build an engine by name. Paddle is imported only when asked for."""
    if name == "ccl":
        return CCLEngine(cfg)
    if name in ("ccl-cnn", "ccl-cnn-nolm", "ccl-crnn", "ccl-glyph"):
        from ocr.recog.recognizer import default_recognizer

        # ccl-cnn: glyph CNN + LM, and the Latin line CRNN per line by confidence when
        # its checkpoint exists. ccl-crnn / ccl-glyph: one Latin reader only (ablations).
        route = {"ccl-crnn": "crnn", "ccl-glyph": "cnn"}.get(name, "confidence")
        engine = CCLEngine(cfg, recognizer=default_recognizer(lm=name != "ccl-cnn-nolm", latin_route=route))
        engine.name = name
        return engine
    from ocr.paddle_engine import PaddleConfig, PaddleEngine

    kw = {"cpu_threads": threads} if threads else {}
    return PaddleEngine(PaddleConfig(mode="full" if name == "paddle" else "det", **kw))


def _apply_threads(threads: int | None) -> None:
    # One thread budget for every engine; otherwise the comparison measures the
    # thread pools, not the pipelines.
    if threads:
        import cv2

        cv2.setNumThreads(threads)


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
        note=args.note or "",
    )
    cfg = _config(args.config)
    _apply_threads(args.threads)
    engine = make_engine(args.engine, cfg, args.threads)
    print(f"running {engine.name} over {store.root}")
    result = runner.run(store, engine=engine, run_config=rc,
                        pipeline_config=cfg if args.engine.startswith("ccl") else None)

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
    if words.n_gt:
        print(f"  word hit rate {words.hit_rate:.1%}, coverage {words.mean_coverage:.1%}")
    regions = runner.aggregate_regions(s.regions for s in result.samples)
    print(f"  line recall {regions.line_recall:.1%} over {regions.n_gt} GT lines")
    text = runner.aggregate_text(s.text for s in result.samples)
    if text.n_gt_chars:
        print(f"  CER {text.cer:.1%}, word F1 {text.word_f1:.1%}")
    grouping = runner.aggregate_grouping(s.grouping for s in result.samples)
    if grouping.n_gt_words:
        lp, lr, lf = grouping.line_prf
        wp, wr, wf = grouping.word_prf
        print(f"  grouping: line F1 {lf or 0:.1%} (P {lp or 0:.1%} R {lr:.1%}, "
              f"ceiling {grouping.line_recall_ceiling:.1%}), "
              f"word F1 {wf or 0:.1%} (P {wp or 0:.1%} R {wr:.1%}, "
              f"ceiling {grouping.word_recall_ceiling:.1%})")
    print(f"\n  {summary_path}")
    return 0


def _print_checks(results: dict[str, runner.RunResult], roles: compare.Roles) -> None:
    for c in compare.checks(results, roles):
        value = "n/a" if c.value is None else f"{c.value:.2f}"
        status = "n/a" if c.passed is None else ("pass" if c.passed else "FAIL")
        print(f"  [{status:4}] {c.slice:8} {c.name}: {value} (limit {c.limit:.2f}) - {c.detail}")


def _roles(args: argparse.Namespace) -> compare.Roles:
    return compare.Roles(candidate=args.candidate, baseline=args.baseline,
                         baseline_det=args.baseline_det)


def cmd_compare(args: argparse.Namespace) -> int:
    """Phase 6: run engines over the same documents, one comparison report.

    Each engine's ``results.json`` and ``summary.md`` are written the moment that
    engine finishes, not after the last one: a full-Paddle pass takes tens of minutes,
    and a killed or crashed run must not throw away the engines that completed.
    Engines whose results already exist in ``--out`` are loaded instead of re-run, so
    re-invoking the same command resumes where it stopped.
    """
    store = BenchmarkStore(args.bench)
    rc = runner.RunConfig(
        repeats=args.repeats,
        warmup=args.warmup,
        limit=args.limit,
        sources=tuple(args.sources or ()),
        captures=tuple(args.captures or ()),
        scripts=tuple(args.scripts or ()),
        note=args.note or "",
    )
    cfg = _config(args.config)
    _apply_threads(args.threads)

    if args.out:
        out_root = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = Path(args.results) / f"{stamp}-compare"
    print(f"results -> {out_root}", flush=True)

    results: dict[str, runner.RunResult] = {}
    for name in args.engines:
        engine_dir = out_root / name
        if (engine_dir / "results.json").exists():
            results[name] = runner.load(engine_dir)
            print(f"  [{name}] already measured ({len(results[name].samples)} samples), "
                  "loaded from disk", flush=True)
            continue
        engine = make_engine(name, cfg, args.threads)
        print(f"running {engine.name} over {store.root}", flush=True)
        res = runner.run(store, engine=engine, run_config=rc,
                         pipeline_config=cfg if name == "ccl" else None)
        report.write(res, runner.save(res, out_root, subdir=name))
        lat = res.latency.get("total", {})
        print(f"  [{name}] done: {len(res.samples)} samples in {res.run_config.get('wall_s')}s, "
              f"P50 {lat.get('p50', 0):.1f} ms, P95 {lat.get('p95', 0):.1f} ms "
              f"-> {engine_dir}", flush=True)
        results[name] = res

    path = compare.write(results, out_root, out_root.name, _roles(args))
    print()
    _print_checks(results, _roles(args))
    print(f"\n  {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild a comparison from saved runs, without measuring anything.

    Each ``--runs`` entry is a folder (or ``results.json``) written by ``run`` or
    ``compare``, optionally prefixed ``name=`` to override the engine name — e.g. to
    set today's CCL run against a Paddle baseline measured last week:

        python -m ocrbench.cli report --candidate ccl-cnn --out bench/results/<x>
            --runs bench/results/<old>-compare/paddle ccl-cnn=bench/results/<new>
    """
    results: dict[str, runner.RunResult] = {}
    for spec in args.runs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = "", spec
        res = runner.load(path)
        key = name or res.engine
        if key in results:
            print(f"two runs named {key!r}; prefix one with name=", file=sys.stderr)
            return 2
        results[key] = res
        print(f"  {key}: run {res.run_id}, {len(res.samples)} samples ({path})")

    out_dir = Path(args.out)
    path = compare.write(results, out_dir, out_dir.name, _roles(args))
    print()
    _print_checks(results, _roles(args))
    print(f"\n  {path}")
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
    p.add_argument("--engine", choices=ENGINES, default="ccl")
    p.add_argument("--threads", type=int, help="thread budget for cv2 and Paddle")
    p.add_argument("--note", help="stored with the run, e.g. 'machine busy: GPU training'")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", parents=[bench_arg], help="Phase 6: run several engines, compare")
    p.add_argument("--engines", nargs="+", choices=ENGINES, default=["ccl", "paddle-det", "paddle"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--sources", nargs="+")
    p.add_argument("--captures", nargs="+")
    p.add_argument("--scripts", nargs="+")
    p.add_argument("--results", default="bench/results")
    p.add_argument("--threads", type=int, default=8, help="thread budget for cv2 and Paddle")
    p.add_argument("--out", help="output folder; engines already saved there are loaded, not re-run")
    p.add_argument("--note", help="stored with each run, e.g. 'machine busy: GPU training'")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report", help="rebuild a comparison from saved runs")
    p.add_argument("--runs", nargs="+", required=True,
                   help="run folders or results.json files, optionally name=path")
    p.add_argument("--out", required=True, help="folder to write comparison.md into")
    p.set_defaults(func=cmd_report)

    # Who is under test and who is the bar, for both compare and report.
    for name in ("compare", "report"):
        sp = sub.choices[name]
        sp.add_argument("--candidate", default="ccl", help="engine under test (default: ccl)")
        sp.add_argument("--baseline", default="paddle", help="full baseline (default: paddle)")
        sp.add_argument("--baseline-det", default="paddle-det",
                        help="detection-only baseline (default: paddle-det)")

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
