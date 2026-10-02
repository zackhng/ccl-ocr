"""Real-document datasets for Phase 3: a test benchmark and a training pool.

    uv run python scripts/fetch_real.py                  # everything below
    uv run python scripts/fetch_real.py --only cord

bench_real/          TEST splits only — the real-document benchmark.
    sroie test (MY receipts), funsd testing_data (forms), cord test (ID receipts),
    xfund val in de/es/fr/it/pt (accented-Latin forms).
data/real/train/     TRAIN splits only — for the CNN (via alignment) and the LM.
    sroie train, funsd training_data, cord train+validation, xfund train (Latin langs).

Splits never cross: a document is in exactly one of the two, by its dataset's own split,
and ``tests/test_real_splits.py`` asserts the stores share no image. Documents already
in the published ``bench`` are excluded from training as well.

Images and ground truth are written in the benchmark's own format
(``BenchmarkStore``), so every existing tool — the runner, metrics, overlays, the
comparison report — works on them unchanged. Parquet shards are streamed row group by
row group from the Hugging Face hub, so the download is the data itself (~3.6 GB), not
a dataset cache.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrbench.adapters import cord, funsd, sroie, xfund  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402


def _decode(image_field) -> np.ndarray:
    import cv2

    data = image_field["bytes"] if isinstance(image_field, dict) else image_field
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


def _shards(dataset: str, split: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet") and f.rsplit("/", 1)[1].startswith(f"{split}-"))


def _rows(dataset: str, split: str, columns: list[str]):
    """Whole-shard downloads through the hub CDN (cached under data/hf_cache), then
    read locally. Streaming row groups through HfFileSystem issues many small range
    requests and ran at a crawl on 450 MB shards."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for shard in _shards(dataset, split):
        local = hf_hub_download(dataset, shard, repo_type="dataset", cache_dir="data/hf_cache")
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=columns).to_pylist()


def fetch_split(dataset: str, split: str, store: BenchmarkStore, exclude: set[str], tmp: Path) -> int:
    """Convert one split of one dataset into ``store``. Returns documents written."""
    n = 0
    if dataset == "cord":
        for i, row in enumerate(_rows("naver-clova-ix/cord-v2", split, ["image", "ground_truth"])):
            store.write(cord.sample_from_record(row["ground_truth"], split, i), _decode(row["image"]))
            n += 1
    elif dataset == "sroie":
        for row in _rows("jsdnrs/ICDAR2019-SROIE", split, ["image", "key", "words", "bboxes", "entities"]):
            if f"sroie_{row['key']}" in exclude and store.root.name != "bench_real":
                continue
            sample = sroie.sample_from_row(row, tmp / f"{row['key']}.jpg")
            sample.meta["split"] = split
            store.write(sample, _decode(row["image"]))
            n += 1
    elif dataset == "xfund":
        for row in _rows("nnul/xfund-multilingual", split, ["id", "words", "bboxes", "image"]):
            if row["id"].split("_")[0] not in xfund.LATIN_LANGUAGES:
                continue
            sample = xfund.sample_from_record(row["id"], row["words"], row["bboxes"])
            sample.meta["split"] = split
            store.write(sample, _decode(row["image"]))
            n += 1
    return n


PLAN = (
    # Test splits first, so the real benchmark is complete early on a slow link.
    ("cord", "test", "test"), ("sroie", "test", "test"), ("xfund", "val", "test"),
    ("sroie", "train", "train"), ("cord", "train", "train"), ("cord", "validation", "train"),
    ("xfund", "train", "train"),
)
"""(dataset, split, destination). Each finished step leaves a marker in
``data/real/.done/``, so an interrupted fetch resumes where it stopped."""


def fetch_funsd(test: BenchmarkStore, train: BenchmarkStore, exclude: set[str], root: Path) -> dict:
    t = funsd.convert(root, test, split="testing_data")
    all_train = funsd.convert(root, train, split="training_data")
    # The published bench's 25 FUNSD forms come from testing_data, so this normally
    # drops nothing; it is kept so the guarantee does not depend on that.
    dropped = 0
    for s in all_train:
        if s.sample_id in exclude:
            train.gt_path(s.sample_id).unlink(missing_ok=True)
            train.image_path(s.sample_id).unlink(missing_ok=True)
            dropped += 1
    print(f"  funsd test: {len(t)}, train: {len(all_train) - dropped} ({dropped} bench forms excluded)", flush=True)
    return {"test": len(t), "train": len(all_train) - dropped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-out", default="bench_real")
    ap.add_argument("--train-out", default="data/real/train")
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--funsd", default="bench/raw/funsd/FUNSD")
    ap.add_argument("--only", nargs="+", choices=["cord", "sroie", "xfund", "funsd"])
    args = ap.parse_args()

    test, train = BenchmarkStore(args.test_out), BenchmarkStore(args.train_out)
    test.ensure_dirs()
    train.ensure_dirs()
    exclude = {s["sample_id"] for s in json.loads(Path(args.bench, "manifest.json").read_text())["samples"]}
    tmp = Path(args.train_out).parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    done_dir = Path(args.train_out).parent / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    if not args.only or "funsd" in args.only:
        if not (done_dir / "funsd").exists():
            report["funsd"] = fetch_funsd(test, train, exclude, Path(args.funsd))
            (done_dir / "funsd").touch()
    for dataset, split, dest in PLAN:
        if args.only and dataset not in args.only:
            continue
        marker = done_dir / f"{dataset}_{split}"
        if marker.exists():
            print(f"  {dataset} {split}: already fetched", flush=True)
            continue
        n = fetch_split(dataset, split, test if dest == "test" else train, exclude, tmp)
        marker.touch()
        report[f"{dataset}_{split}"] = n
        print(f"  {dataset} {split} -> {dest}: {n}", flush=True)
        (test if dest == "test" else train).write_manifest()

    test.write_manifest()
    train.write_manifest()
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
