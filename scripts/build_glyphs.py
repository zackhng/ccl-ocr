"""Build the glyph dataset for the Phase 3 CNN from corpus pages run through CCL.

    uv run python scripts/build_glyphs.py --pages 4000 --out data/glyphs/train --seed0 1000
    uv run python scripts/build_glyphs.py --pages 300  --out data/glyphs/val   --seed0 900000
    uv run python scripts/build_glyphs.py --bench bench_tune --out data/glyphs/heldout
    uv run python scripts/build_glyphs.py --real data/real/train --out data/glyphs/real

``--real`` labels *real* documents, which annotate words, not characters, by aligning
CCL clusters to each word's transcript (``ocrbench.glyphs.label_by_alignment``): a word
contributes only when cluster and character counts agree. Real crops are the primary
training data (user requirement); corpus pages supplement charset gaps.

Each page is generated (``ocrbench.synth.corpus_page``), degraded with the benchmark's
capture profiles, run through the real ``CCLEngine``, and every glyph cluster is cropped
(``ocr.recog.crops``) and labelled from ground truth (``ocrbench.glyphs``). The CNN
therefore trains on exactly the crops it will see at inference.

``--bench`` labels an existing benchmark directory instead (the held-out seed-11 set:
for calibration and tuning, never for training). Never pass the published ``bench``.

Output: ``shard_XXXX.npz`` files with crops (uint8 N x 32 x 32), geometry (float16 N x 5),
labels (int16), profile ids and page seeds, plus ``summary.json`` with class counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.engine import CCLEngine, load_image  # noqa: E402
from ocr.recog.charset import GLYPH_CLASSES  # noqa: E402
from ocr.recog.crops import extract, to_gray  # noqa: E402
from ocrbench.glyphs import doc_seed, is_tune_doc, label_by_alignment, label_clusters  # noqa: E402
from ocrbench.gt import BenchmarkStore  # noqa: E402

PROFILES = ("clean_scan", "scan", "photo", "hard_photo")
_STATE: dict = {}


def _init(corpora: str) -> None:
    from ocrbench.synth.corpus_page import TextSource

    _STATE["engine"] = CCLEngine()
    if corpora:
        _STATE["src"] = TextSource(Path(corpora))


def _page(seed: int):
    from ocrbench.synth.corpus_page import build_spec
    from ocrbench.synth.generate import sample_from_spec

    spec, profile, rng = build_spec(seed, _STATE["src"])
    sample, image = sample_from_spec(f"corpus_{seed}", spec, profile, rng, seed)
    return _crop(sample, image, profile, seed)


def _bench_page(args):
    root, sid = args
    store = BenchmarkStore(root)
    sample = store.read(sid)
    image = load_image(str(store.image_path(sid)))
    return _crop(sample, image, str(sample.meta.get("profile")), int(sample.meta.get("seed", 0)))


def _real_page(args):
    root, sid = args
    store = BenchmarkStore(root)
    sample = store.read(sid)
    image = load_image(str(store.image_path(sid)))
    return _crop(sample, image, sample.capture, doc_seed(sid), align=True)


def _crop(sample, image, profile, seed, align=False):
    if not (sample.words or sample.lines) if align else not sample.chars:
        return None
    result = _STATE["engine"].run(image)
    crops, geom, clusters = extract(to_gray(image), result)
    labeller = label_by_alignment if align else label_clusters
    labels = np.array(labeller(sample, clusters), dtype=np.int16)
    keep = labels >= 0
    pid = PROFILES.index(profile) if profile in PROFILES else -1
    return (crops[keep], geom[keep].astype(np.float16), labels[keep],
            np.full(int(keep.sum()), pid, np.int8), np.full(int(keep.sum()), seed, np.int64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--bench", help="label an existing synthetic benchmark directory instead")
    ap.add_argument("--real", help="label a real-document store by word alignment")
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=250, help="pages per shard")
    args = ap.parse_args()


    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for root in (args.bench, args.real):
        if root and Path(root).resolve() in (Path("bench").resolve(), Path("bench_real").resolve()):
            print("refusing to build training data from a test benchmark", file=sys.stderr)
            return 2
    if args.real:
        # The decoder's tuning documents (is_tune_doc) never train the CNN.
        jobs = [(args.real, sid) for sid in BenchmarkStore(args.real).sample_ids() if not is_tune_doc(sid)]
        fn, init_arg = _real_page, ""
    elif args.bench:
        jobs = [(args.bench, sid) for sid in BenchmarkStore(args.bench).sample_ids()]
        fn, init_arg = _bench_page, ""
    else:
        jobs = list(range(args.seed0, args.seed0 + args.pages))
        fn, init_arg = _page, args.corpora

    counts: Counter = Counter()
    buf: list = []
    shard = 0

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        cat = [np.concatenate(x) for x in zip(*buf)]
        np.savez_compressed(out / f"shard_{shard:04d}.npz", crops=cat[0], geom=cat[1],
                            labels=cat[2], profile=cat[3], seed=cat[4])
        shard += 1
        buf = []

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(init_arg,)) as pool:
        for k, res in enumerate(pool.map(fn, jobs, chunksize=4), 1):
            if res is not None:
                buf.append(res)
                counts.update(res[2].tolist())
            if k % args.shard == 0:
                flush()
                print(f"  {k}/{len(jobs)} pages, {sum(counts.values())} crops", flush=True)
    flush()

    total = sum(counts.values())
    by_name = {GLYPH_CLASSES[i]: n for i, n in counts.most_common()}
    lower = sum(n for c, n in by_name.items() if len(c) == 1 and c.islower())
    accented = sum(n for c, n in by_name.items() if len(c) == 1 and ord(c) > 127 and c.isalpha())
    summary = {
        "pages": len(jobs), "crops": total, "classes_seen": len(by_name),
        "share": {k: round(by_name.get(k, 0) / max(1, total), 4) for k in ("<NONTEXT>", "<MULTI>", "<PART>")},
        "lowercase_share": round(lower / max(1, total), 4),
        "accented_share": round(accented / max(1, total), 4),
        "counts": by_name,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
