"""Download synthetic Indian bank cheques and convert them into the benchmark.

    uv run python scripts/fetch_cheques.py --limit 40

Source: https://huggingface.co/datasets/jaganadhg/cheque-synthetic-images (Apache-2.0)
295 cheques composited from real IDRBT field crops, each with six field boxes.

These contribute real cheque paper, print and signatures — the things the synthetic
generator cannot fake — but their annotations are coarse field regions, so they
deliberately carry no line/word ground truth. See ``ocrbench.adapters.cheque``.

The companion real-cheque annotations (``jaganadhg/cheque-field-annotations``, 112
IDRBT cheques) ship without images and IDRBT states no licence, so they are not wired
up here.
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

from ocrbench.adapters import cheque  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402

DATASET = "jaganadhg/cheque-synthetic-images"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def _get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ocr-engine-benchmark/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_rows(limit: int, raw: Path) -> list[dict]:
    images_dir = raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0

    while len(rows) < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET, "config": "default", "split": "train",
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
            image_path = images_dir / f"{row['image_id']}.jpg"
            if not image_path.exists():
                src = row["image"]["src"] if isinstance(row["image"], dict) else row["image"]
                try:
                    image_path.write_bytes(_get(src))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {row['image_id']}: image download failed ({exc})")
                    continue
            row.pop("image", None)
            rows.append(row)
            if len(rows) >= limit:
                break

        offset += len(batch)
        print(f"  {len(rows)} rows")
        time.sleep(0.2)

    (raw / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="bench/raw/cheques")
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    if (raw / "rows.jsonl").exists():
        print(f"  using cached rows at {raw / 'rows.jsonl'}")
    else:
        print(f"  fetching up to {args.limit} rows from {DATASET}")
        fetch_rows(args.limit, raw)

    store = BenchmarkStore(args.bench)
    samples = cheque.convert(raw, store, limit=args.limit)
    store.write_manifest()

    print(f"\n  converted {len(samples)} cheque samples into {store.root}")
    print("  note: field-region annotations only -- these contribute latency, component")
    print("        routing and PII/signature regions, but no line or word coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
