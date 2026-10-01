"""MIDV-2020 access instructions, and conversion once you have the archive.

    uv run python scripts/fetch_midv.py --raw /path/to/midv2020 --limit 60

**This cannot be automated.** The dataset is gated: you must accept the licence through
a Google Form, after which you get credentials for an sFTP server at the University of
La Rochelle. The full release is 124 GB.

Request access:  https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html

Pull only what the benchmark needs — the ``scan`` and ``photo`` subsets. The 1000 video
clips are the bulk of the 124 GB and are useless here: this is a still-image pipeline.

The adapter is **UNVERIFIED** (see ``ocrbench.adapters.midv``). It is written against
the documented VIA annotation format and fails loudly on a mismatch rather than
producing plausible-looking nonsense.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.adapters import midv  # noqa: E402
from ocrbench.adapters.base import AdapterError  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402

INSTRUCTIONS = __doc__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="path to an already-downloaded MIDV-2020 tree")
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    if not args.raw:
        print(INSTRUCTIONS)
        print("\nNothing to do: pass --raw once you have the archive.")
        return 0

    store = BenchmarkStore(args.bench)
    try:
        samples = midv.convert(args.raw, store, limit=args.limit)
    except AdapterError as exc:
        print(f"\nadapter failed (it is UNVERIFIED by design):\n  {exc}", file=sys.stderr)
        return 1

    store.write_manifest()
    print(f"  converted {len(samples)} MIDV samples into {store.root}")
    print("  note: document/face geometry only -- no text boxes, so no region coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
