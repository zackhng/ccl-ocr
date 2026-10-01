"""Command line entry point for the engine itself.

    python -m ocr.cli visualize bench/images/*.png --out overlays/
    python -m ocr.cli run path/to/doc.png --json

Benchmark-wide commands live in ``ocrbench.cli``; this one operates on single images
and is what you reach for when an overlay looks wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig
from .engine import CCLEngine, load_image
from .types import ComponentKind
from .visualize import binary_preview, draw_components, save_image, side_by_side


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in glob.glob(pattern)]
        if matched:
            paths.extend(sorted(matched))
        elif Path(pattern).exists():
            paths.append(Path(pattern))
        else:
            print(f"warning: no match for {pattern!r}", file=sys.stderr)
    return paths


def _load_config(path: str | None) -> PipelineConfig:
    if not path:
        return DEFAULT_CONFIG
    return PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_visualize(args: argparse.Namespace) -> int:
    engine = CCLEngine(_load_config(args.config))
    out_dir = Path(args.out)
    kinds = None
    if args.only:
        kinds = {ComponentKind(k) for k in args.only}

    paths = _expand(args.images)
    if not paths:
        print("no images matched", file=sys.stderr)
        return 1

    for path in paths:
        image = load_image(str(path))
        result, debug = engine.run_with_debug(image)
        overlay = draw_components(image, result, kinds=kinds, label_reasons=args.label_reasons)

        panel = overlay
        if args.stages:
            # Source, binary, overlay. When a page goes wrong it is almost always
            # visible in the binary, not in the boxes.
            panel = side_by_side(image, binary_preview(debug.binary), overlay)

        save_image(out_dir / f"{path.stem}.png", panel)
        counts = result.kind_counts()
        print(
            f"{path.name}: {len(result.components)} components "
            f"(text={counts['text']} diacritic={counts['diacritic']} "
            f"rule={counts['rule']} blob={counts['blob']} noise={counts['noise']}) "
            f"{result.timings_ms['total']:.1f} ms"
        )

    print(f"\nwrote {len(paths)} overlay(s) to {out_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    engine = CCLEngine(_load_config(args.config))
    image = load_image(args.image)
    result = engine.run(image)

    payload = {
        "width": result.width,
        "height": result.height,
        "scale": result.scale,
        "timings_ms": result.timings_ms,
        "meta": result.meta,
        "counts": result.kind_counts(),
    }
    if args.components:
        payload["components"] = [
            {
                "id": c.id,
                "bbox": c.bbox.as_list(),
                "kind": c.kind.value,
                "reason": c.reason,
                "pixel_area": c.pixel_area,
                "fill_ratio": round(c.fill_ratio, 3),
            }
            for c in result.components
        ]

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{result.width}x{result.height} scale={result.scale:.3f}")
        print(f"counts: {payload['counts']}")
        print(f"timings (ms): {result.timings_ms}")
        print(f"meta: {result.meta}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="JSON file with a PipelineConfig override")

    vis = sub.add_parser("visualize", parents=[common], help="render component overlays")
    vis.add_argument("images", nargs="+", help="image paths or globs")
    vis.add_argument("--out", default="overlays", help="output directory")
    vis.add_argument(
        "--only", nargs="+", choices=[k.value for k in ComponentKind],
        help="draw only these component kinds",
    )
    vis.add_argument("--stages", action="store_true", help="also show source and binary panels")
    vis.add_argument("--label-reasons", action="store_true", help="annotate each box with its gate")
    vis.set_defaults(func=cmd_visualize)

    run = sub.add_parser("run", parents=[common], help="run the engine on one image")
    run.add_argument("image")
    run.add_argument("--json", action="store_true")
    run.add_argument("--components", action="store_true", help="include every component")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
