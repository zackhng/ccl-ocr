"""Download a sample of ICDAR-2019 SROIE and convert it into the benchmark.

    uv run python scripts/fetch_sroie.py --limit 50

Pulls rows through the Hugging Face datasets-server rows API rather than the parquet
files. The parquet split is ~500 MB for 987 receipts; we want ~50, and the rows endpoint
serves annotations as JSON plus a signed image URL. No pyarrow dependency, no half-gig
download for a 50-document slice.

Source: https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE  (CC-BY-4.0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.adapters import sroie  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402

DATASET = "jsdnrs/ICDAR2019-SROIE"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100
"""The rows endpoint caps a request at 100 rows."""


def _get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ocr-engine-benchmark/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_rows(split: str, limit: int, raw: Path) -> list[dict]:
    """Page the rows API, downloading each image as we go.

    Image URLs are signed and expire, so they are fetched in the same pass rather than
    collected and resolved later.
    """
    images_dir = raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0

    while len(rows) < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET, "config": "default", "split": split,
                "offset": offset, "length": min(PAGE, limit - len(rows)),
            }
        )
        payload = json.loads(_get(f"{ROWS_URL}?{query}"))
        if "error" in payload:
            raise SystemExit(f"datasets-server error: {payload['error']}")
        batch = payload.get("rows", [])
        if not batch:
            break

        for item in batch:
            row = item["row"]
            key = row["key"]
            image_path = images_dir / f"{key}.jpg"
            if not image_path.exists():
                src = row["image"]["src"] if isinstance(row["image"], dict) else row["image"]
                try:
                    image_path.write_bytes(_get(src))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {key}: image download failed ({exc})")
                    continue
            row.pop("image", None)  # the signed URL is useless once it expires
            rows.append(row)
            if len(rows) >= limit:
                break

        offset += len(batch)
        print(f"  {len(rows)} rows")
        time.sleep(0.2)  # be polite to a free public endpoint

    (raw / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="bench/raw/sroie")
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    args = parser.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    if (raw / "rows.jsonl").exists():
        print(f"  using cached rows at {raw / 'rows.jsonl'}")
    else:
        print(f"  fetching up to {args.limit} rows from {DATASET} [{args.split}]")
        fetch_rows(args.split, args.limit, raw)

    store = BenchmarkStore(args.bench)
    samples = sroie.convert(raw, store, limit=args.limit)
    store.write_manifest()

    regions = sum(len(s.lines) for s in samples)
    print(f"\n  converted {len(samples)} SROIE samples ({regions} line regions) into {store.root}")
    print("  note: line-level ground truth, no characters -- region metric reports "
          "granularity='line'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
