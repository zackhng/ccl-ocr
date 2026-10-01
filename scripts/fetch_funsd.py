"""Download and unpack FUNSD, then convert it into the benchmark.

    uv run python scripts/fetch_funsd.py --limit 30

FUNSD is 199 noisy scanned forms with word-level boxes and transcripts, licensed for
**non-commercial research and educational use only**. That is fine for benchmarking an
internal experiment and is a question worth answering before any of it informs a model
that ships.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.adapters import funsd  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402

URL = "https://www.crc.nd.edu/~pmoreira/funsd.zip"


def download(dest: Path) -> Path:
    archive = dest / "funsd.zip"
    if archive.exists():
        print(f"  already downloaded: {archive}")
        return archive
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive.write_bytes(response.read())
    print(f"  {archive.stat().st_size / 1e6:.1f} MB -> {archive}")
    return archive


def extract(archive: Path, dest: Path) -> Path:
    marker = dest / "dataset"
    if marker.exists():
        print(f"  already extracted: {marker}")
        return marker
    print(f"  extracting into {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    if marker.exists():
        return marker
    # Some repackagings drop the dataset/ level; fall back to whatever holds the
    # annotations rather than hard-coding a layout that has changed before.
    found = next((p.parent for p in dest.rglob("annotations") if p.is_dir()), dest)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="bench/raw/funsd", help="download/extract location")
    parser.add_argument("--bench", default="bench", help="benchmark root")
    parser.add_argument("--limit", type=int, help="convert at most N documents")
    parser.add_argument("--split", default="training_data")
    args = parser.parse_args()

    raw = Path(args.raw)
    archive = download(raw)
    root = extract(archive, raw)

    store = BenchmarkStore(args.bench)
    samples = funsd.convert(root, store, limit=args.limit, split=args.split)
    store.write_manifest()

    words = sum(len(s.words) for s in samples)
    print(f"\n  converted {len(samples)} FUNSD samples ({words} word boxes) into {store.root}")
    print("  note: no character-level ground truth -- these exercise the region metric only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
