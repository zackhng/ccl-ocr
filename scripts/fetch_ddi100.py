"""DDI-100 conversion — the only large real source of character-level boxes.

    uv run python scripts/fetch_ddi100.py --raw /path/to/DDI-100 --allow-pickle --limit 40

Download is manual: the archive is distributed in multi-gigabyte parts linked from
https://github.com/machine-intelligence-laboratory/DDI-100 . Grab one part; 40 pages is
plenty for a geometry check.

Two warnings the adapter enforces rather than merely documents:

1. **Cyrillic.** These are Russian document pages. They stress CCL geometry — touching
   glyphs, distortion, stamps over text — but they are not our script. The adapter tags
   them ``script="cyrillic"`` so the report's per-script slice keeps them out of the
   headline automatically. Do not average them in.

2. **Pickles.** Annotations are Python pickles, and unpickling executes arbitrary code.
   ``--allow-pickle`` is required and deliberately not the default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.adapters import ddi100  # noqa: E402
from ocrbench.adapters.base import AdapterError  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="path to an extracted DDI-100 part")
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--allow-pickle", action="store_true",
                        help="required: annotations are pickles and unpickling runs code")
    args = parser.parse_args()

    if not args.raw:
        print(__doc__)
        print("\nNothing to do: pass --raw once you have an extracted part.")
        return 0

    store = BenchmarkStore(args.bench)
    try:
        samples = ddi100.convert(
            args.raw, store, limit=args.limit, allow_pickle=args.allow_pickle
        )
    except AdapterError as exc:
        print(f"\nadapter failed (it is UNVERIFIED by design):\n  {exc}", file=sys.stderr)
        return 1

    store.write_manifest()
    chars = sum(len(s.chars or []) for s in samples)
    print(f"  converted {len(samples)} DDI-100 samples ({chars} character boxes)")
    print("  note: Cyrillic -- report these separately, never in headline accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
