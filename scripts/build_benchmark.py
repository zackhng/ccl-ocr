"""Assemble the full benchmark: synthetic + every real dataset that is reachable.

    uv run python scripts/build_benchmark.py --total 250

Runs the synthetic generator and each fetch script in turn. A source that is gated
(MIDV-2020) or needs a manual download (DDI-100) is reported as skipped rather than
silently omitted — the point of the summary at the end is that you can see exactly what
the benchmark is made of before you read a single metric off it.

Default composition at ``--total 250``, which follows the strata in the plan:

====================  =====  =====================================================
slice                 share  why
====================  =====  =====================================================
synthetic             ~64%   the only source of character-level ground truth
SROIE receipts        ~16%   real scanned financial documents, CC-BY-4.0
FUNSD forms           ~10%   real degraded scans with word boxes
synthetic cheques     ~10%   real cheque paper, print and signatures
====================  =====  =====================================================
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.gt import BenchmarkStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (label, script, share of total, extra args)
SOURCES = [
    ("SROIE receipts", "fetch_sroie.py", 0.16, []),
    ("FUNSD forms", "fetch_funsd.py", 0.10, []),
    ("cheques", "fetch_cheques.py", 0.10, []),
]


def run_script(name: str, args: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "scripts" / name), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=250)
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-network", action="store_true",
                        help="synthetic only; useful offline or in CI")
    args = parser.parse_args()

    store = BenchmarkStore(args.bench)
    report: list[tuple[str, str]] = []

    real_share = sum(s[2] for s in SOURCES) if not args.skip_network else 0.0
    synth_count = max(1, round(args.total * (1.0 - real_share)))

    # Generated in-process: it is the one source with no external dependency, and a
    # subprocess here would only bury its traceback in captured output.
    from ocrbench.synth.generate import generate

    print(f"== synthetic ({synth_count}) ==")
    generate(store, synth_count, seed=args.seed)
    report.append(("synthetic", f"{synth_count} generated"))

    if not args.skip_network:
        for label, script, share, extra in SOURCES:
            count = max(1, round(args.total * share))
            print(f"\n== {label} ({count}) ==")
            ok, output = run_script(script, ["--limit", str(count), "--bench", args.bench, *extra])
            tail = output.splitlines()[-1] if output else ""
            print(output[-800:] if output else "")
            report.append((label, f"{'ok' if ok else 'FAILED'}: {tail}"))

    report.append(("MIDV-2020 ID cards", "skipped: gated, 124 GB -- scripts/fetch_midv.py"))
    report.append(("DDI-100 char boxes", "skipped: manual download -- scripts/fetch_ddi100.py"))

    manifest = store.write_manifest()
    samples = list(store.iter_samples())
    by_source: dict[str, int] = {}
    with_chars = 0
    for s in samples:
        by_source[s.source] = by_source.get(s.source, 0) + 1
        with_chars += bool(s.has_char_gt)

    print("\n" + "=" * 68)
    print("benchmark composition")
    print("=" * 68)
    for label, status in report:
        print(f"  {label:<22} {status}")
    print(f"\n  total samples: {len(samples)}  {by_source}")
    print(f"  with character-level GT: {with_chars}/{len(samples)} "
          f"({with_chars / max(1, len(samples)):.0%})")
    print(f"  manifest: {manifest}")
    print("\n  next: uv run python -m ocrbench.cli run --repeats 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
