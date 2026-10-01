"""Make sure the benchmark data is present locally; fetch only what is missing.

    uv run python scripts/ensure_data.py            # materialise everything missing
    uv run python scripts/ensure_data.py --check    # report only, change nothing
    uv run python scripts/ensure_data.py --only synth sroie

Idempotent by design: run it as often as you like. Each source is inspected first and
skipped if it is already complete, so a second run costs a directory listing rather than
a download.

The benchmark data is not in the repository — 249 MB, and FUNSD's licence restricts
redistribution. Nothing is lost, because every sample is reproducible:

* **synthetic** regenerates byte-identically from its seed (pinned at 7)
* **real** sources re-download through the same fetch scripts

``bench/manifest.json`` *is* committed, so this script knows exactly which 245 samples
the published results were measured against and can tell you when what you have on disk
does not match.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench.gt import BenchmarkStore  # noqa: E402


@dataclass(frozen=True)
class Source:
    name: str
    expected: int
    """Sample count in the published benchmark."""

    how: str
    """Human-readable description of where it comes from."""

    network: bool = True


SOURCES = (
    Source("synth", 150, "generated locally from seed 7", network=False),
    Source("sroie", 40, "HF datasets-server, jsdnrs/ICDAR2019-SROIE (CC-BY-4.0)"),
    Source("funsd", 25, "crc.nd.edu/~pmoreira/funsd.zip (non-commercial research only)"),
    Source("cheque", 30, "HF datasets-server, jaganadhg/cheque-synthetic-images (Apache-2.0)"),
)

# Sources that cannot be automated at all, reported so their absence is explicit rather
# than invisible.
MANUAL = (
    ("midv", "gated behind an access form, 124 GB -- scripts/fetch_midv.py"),
    ("ddi100", "manual multi-GB download -- scripts/fetch_ddi100.py"),
)

SEED = 7


def present(store: BenchmarkStore, prefix: str) -> tuple[int, int]:
    """(samples with ground truth, samples that also have their image) for a source."""
    ids = [sid for sid in store.sample_ids() if sid.startswith(f"{prefix}_")]
    with_images = sum(1 for sid in ids if store.image_path(sid).exists())
    return len(ids), with_images


def run(script: str, args: list[str]) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"    $ {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode == 0


def fetch(source: Source, bench: str) -> bool:
    if source.name == "synth":
        from ocrbench.synth.generate import generate

        generate(BenchmarkStore(bench), source.expected, seed=SEED)
        return True
    script = {
        "sroie": "fetch_sroie.py",
        "funsd": "fetch_funsd.py",
        "cheque": "fetch_cheques.py",
    }[source.name]
    return run(script, ["--limit", str(source.expected), "--bench", bench])


def manifest_expectation(bench: str) -> dict[str, int] | None:
    path = Path(bench) / "manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for entry in data.get("samples", []):
        counts[entry["source"]] = counts.get(entry["source"], 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--check", action="store_true", help="report only; download nothing")
    parser.add_argument("--only", nargs="+", choices=[s.name for s in SOURCES],
                        help="restrict to these sources")
    parser.add_argument("--offline", action="store_true",
                        help="synthetic only; skip anything needing the network")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even where the data looks complete")
    args = parser.parse_args()

    store = BenchmarkStore(args.bench)
    expected = manifest_expectation(args.bench)
    wanted = [s for s in SOURCES if not args.only or s.name in args.only]
    if args.offline:
        wanted = [s for s in wanted if not s.network]

    print(f"benchmark root: {store.root.resolve()}")
    if expected:
        print("  (expected counts taken from the committed manifest)")
    print()

    plan: list[Source] = []
    print(f"  {'source':<10} {'on disk':>16}  {'want':>5}  status")
    print(f"  {'-' * 10} {'-' * 16:>16}  {'-' * 5:>5}  {'-' * 34}")
    for source in wanted:
        target = (expected or {}).get(source.name, source.expected)
        n_gt, n_img = present(store, source.name)
        complete = n_img >= target and n_gt >= target

        if args.force:
            status, need = "forced re-fetch", True
        elif complete:
            status, need = "ok", False
        elif n_gt and not n_img:
            status, need = "ground truth but no images", True
        elif n_gt:
            status, need = f"incomplete ({n_img}/{target})", True
        else:
            status, need = "missing", True

        print(f"  {source.name:<10} {f'{n_gt} gt / {n_img} img':>16}  {target:>5}  {status}")
        if need:
            plan.append(source)

    for name, note in MANUAL:
        n_gt, _ = present(store, name)
        print(f"  {name:<10} {f'{n_gt} gt':>16}  {'-':>5}  not automated: {note}")

    if not plan:
        print("\nall requested data is present -- nothing to do.")
        print("  next: uv run python -m ocrbench.cli run --repeats 5")
        return 0

    if args.check:
        print(f"\n--check: would fetch {', '.join(s.name for s in plan)}")
        return 1

    failed: list[str] = []
    for source in plan:
        print(f"\n== {source.name} ==\n    {source.how}")
        try:
            if not fetch(source, args.bench):
                failed.append(source.name)
        except Exception as exc:  # noqa: BLE001
            print(f"    failed: {exc}")
            failed.append(source.name)

    store.write_manifest()
    total = len(store.sample_ids())
    print(f"\n{'=' * 60}")
    print(f"benchmark now holds {total} samples")
    if failed:
        # Loud, and a non-zero exit: a quietly shrunken benchmark still reports a
        # confident-looking number over whatever happened to download.
        print(f"FAILED: {', '.join(failed)} -- results will not match the published run")
        return 1
    print("  next: uv run python -m ocrbench.cli run --repeats 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
