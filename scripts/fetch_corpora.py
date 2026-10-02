"""Text corpora for the Phase 3 character LM and training pages.

    uv run python scripts/fetch_corpora.py          # -> data/corpora/

public.txt
    Wikipedia (wikimedia/wikipedia 20231101) in the Latin-script languages of the
    jurisdictions in scope — English, Indonesian, Vietnamese — read lazily from the
    parquet shards (only the first row groups are fetched). Case and diacritics are
    preserved. CC BY-SA 4.0.
financial.txt
    Real financial-document text for fine-tuning:
    - SROIE receipts, *train* split (the benchmark uses the *test* split);
    - FUNSD forms, training_data only (testing_data is the real-document benchmark);
    - CORD and XFUND *train* transcripts, read from ``data/real/train``.
    FUNSD is non-commercial research only — fine for this internal model, a question
    for a shipped one (docs/DATASETS.md).
manifest.json
    Sources, line counts, and the benchmark sample ids that were *excluded*.

**Nothing from either benchmark may reach the LM**, or its CER on those documents would
be grading memorisation. Exclusion is by id against ``bench/manifest.json`` *and*
``bench_real/manifest.json``, asserted by ``tests/test_real_splits.py``. The synthetic
generator's vocabulary is not used anywhere here for the same reason.
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

from ocr.recog.charset import normalise  # noqa: E402

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def _rows(dataset: str, config: str, split: str, limit: int, fields: tuple[str, ...]):
    offset, n = 0, 0
    while n < limit:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": min(PAGE, limit - n)})
        req = urllib.request.Request(f"{ROWS_URL}?{q}", headers={"User-Agent": "ocr-engine/0.1"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload = json.loads(r.read())
                break
            except Exception as exc:  # noqa: BLE001 - flaky public endpoint
                if attempt == 3:
                    raise SystemExit(f"rows API failed for {dataset}: {exc}")
                time.sleep(2 * (attempt + 1))
        batch = payload.get("rows", [])
        if not batch:
            return
        for item in batch:
            yield {k: item["row"].get(k) for k in fields}
        n += len(batch)
        offset += len(batch)
        time.sleep(0.15)


def benchmark_ids(bench: Path) -> set[str]:
    data = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    return {s["sample_id"] for s in data["samples"]}


PUBLIC_LANGUAGES = {"en": 20_000_000, "id": 6_000_000, "vi": 6_000_000}
"""Characters per language. English dominates financial documents in every
jurisdiction in scope; Indonesian and Vietnamese cover the local-language Latin text."""


def public_corpus(budget: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    lines: list[str] = []
    per_lang: dict[str, int] = {}
    for lang, max_chars in budget.items():
        path = f"datasets/wikimedia/wikipedia/20231101.{lang}/train-00000-of-*.parquet"
        shard = sorted(fs.glob(path))[0]
        total = 0
        with fs.open(shard, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                    for para in (text or "").splitlines():
                        t = normalise(para)
                        if len(t) < 40:  # headings, list stubs
                            continue
                        lines.append(t)
                        total += len(t)
                    if total >= max_chars:
                        break
                print(f"  {lang}: {total / 1e6:.1f}M chars", flush=True)
                if total >= max_chars:
                    break
        per_lang[lang] = total
    return lines, per_lang


def sroie_train(excluded: set[str]) -> tuple[list[str], list[str]]:
    lines, skipped = [], []
    for row in _rows("jsdnrs/ICDAR2019-SROIE", "default", "train", 10**5, ("key", "words")):
        key = f"sroie_{row['key']}"
        if key in excluded:
            skipped.append(key)
            continue
        lines += [normalise(str(w)) for w in (row["words"] or []) if str(w).strip()]
    return lines, skipped


def funsd_forms(root: Path, excluded: set[str]) -> tuple[list[str], list[str]]:
    lines, skipped = [], []
    for split in ("training_data",):
        for path in sorted((root / split / "annotations").glob("*.json")):
            if f"funsd_{path.stem}" in excluded:
                skipped.append(f"funsd_{path.stem}")
                continue
            form = json.loads(path.read_text(encoding="utf-8")).get("form", [])
            lines += [normalise(e.get("text", "")) for e in form if e.get("text", "").strip()]
    return lines, skipped


SCRIPT_WIKIS = {"han": "zh", "devanagari": "hi", "thai": "th", "arabic": "ar"}
"""Phase 7: one Wikipedia per non-Latin script in scope (CN, IN, TH, UAE)."""


def script_corpus(lang: str, max_chars: int) -> list[str]:
    """Lines of one Wikipedia, NFC and whitespace-normalised only — the Latin
    charset filter (``normalise``) would erase these scripts."""
    import re
    import unicodedata

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shard = sorted(fs.glob(f"datasets/wikimedia/wikipedia/20231101.{lang}/train-00000-of-*.parquet"))[0]
    lines, total = [], 0
    with fs.open(shard, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                for para in (text or "").splitlines():
                    t = " ".join(unicodedata.normalize("NFC", para).split())
                    if len(t) < 20 or re.fullmatch(r"[\W\d_]+", t):
                        continue
                    lines.append(t)
                    total += len(t)
                if total >= max_chars:
                    return lines
    return lines


def excluded_roots(bench: str) -> list[Path]:
    return [p for p in (Path(bench), Path("bench_real")) if (p / "gt").exists()]


def benchmark_lines(roots: list[Path]) -> set[str]:
    """Every normalised transcript line/word of every test document."""
    from ocrbench.gt import BenchmarkStore

    out: set[str] = set()
    for root in roots:
        store = BenchmarkStore(root)
        for sid in store.sample_ids():
            s = store.read(sid)
            for g in s.lines + s.words:
                if g.text:
                    out.add(normalise(g.text))
    return out


def real_train_text(root: Path, excluded: set[str]) -> tuple[list[str], list[str]]:
    """Transcripts from the real training store — CORD and XFUND only, since SROIE and
    FUNSD are read from their own sources above."""
    from ocrbench.gt import BenchmarkStore

    lines: list[str] = []
    used: list[str] = []
    if not (root / "gt").exists():
        return lines, used
    store = BenchmarkStore(root)
    for sid in store.sample_ids():
        if sid in excluded or not sid.startswith(("cord_", "xfund_")):
            continue
        s = store.read(sid)
        if s.lines:
            texts = [g.text for g in s.lines]
        else:
            # Word-only annotation (XFUND): chunks of ten words in annotation order,
            # rather than one multi-thousand-character "line" per form.
            words = [w.text for w in s.words]
            texts = [" ".join(words[k:k + 10]) for k in range(0, len(words), 10)]
        lines += [normalise(t) for t in texts if t.strip()]
        used.append(sid)
    return lines, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora")
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--funsd", default="bench/raw/funsd/FUNSD")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply per-language budgets")
    ap.add_argument("--real-train", default="data/real/train")
    ap.add_argument("--financial-only", action="store_true", help="rebuild financial.txt only")
    ap.add_argument("--scripts", nargs="*", choices=list(SCRIPT_WIKIS),
                    help="Phase 7: fetch only these scripts' corpora (script_<name>.txt)")
    ap.add_argument("--script-chars", type=int, default=4_000_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.scripts is not None:
        for script in args.scripts or list(SCRIPT_WIKIS):
            lines = script_corpus(SCRIPT_WIKIS[script], args.script_chars)
            (out / f"script_{script}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  {script} ({SCRIPT_WIKIS[script]}): {len(lines)} lines, {sum(map(len, lines)) / 1e6:.1f}M chars", flush=True)
        return 0
    excluded = benchmark_ids(Path(args.bench))
    if Path("bench_real/manifest.json").exists():
        excluded |= benchmark_ids(Path("bench_real"))

    print("financial: SROIE train split", flush=True)
    sroie, sroie_skipped = sroie_train(excluded)
    print(f"  {len(sroie)} lines ({len(sroie_skipped)} benchmark receipts excluded)")
    funsd, funsd_skipped = funsd_forms(Path(args.funsd), excluded)
    print(f"  FUNSD: {len(funsd)} entities ({len(funsd_skipped)} benchmark forms excluded)")
    real, real_ids = real_train_text(Path(args.real_train), excluded)
    print(f"  CORD/XFUND train: {len(real)} lines from {len(real_ids)} documents")
    financial = [t for t in sroie + funsd + real if t]
    # Line-level decontamination. Public datasets' train and test splits share issuers:
    # a SROIE test receipt repeats its shop's header, address and footer from training
    # receipts of the same shop; FUNSD forms share a company's boilerplate. The LM would
    # memorise those exact lines and the test CER would grade recall, not reading. Any
    # line of >= 12 characters that occurs verbatim in a test document is dropped.
    test_lines = benchmark_lines(excluded_roots(args.bench))
    before = len(financial)
    financial = [t for t in financial if not (len(t) >= 12 and t in test_lines)]
    print(f"  decontaminated: dropped {before - len(financial)} lines seen in test documents")
    (out / "financial.txt").write_text("\n".join(financial) + "\n", encoding="utf-8")
    financial_manifest = {
        "sources": ["SROIE train", "FUNSD training_data", "CORD train+validation", "XFUND train (Latin)"],
        "lines": len(financial), "chars": sum(map(len, financial)),
        "excluded_benchmark_ids": sorted(sroie_skipped + funsd_skipped),
        "documents_from_real_train": len(real_ids),
        "licence": "SROIE varies by mirror; FUNSD/XFUND research only; CORD CC BY 4.0",
    }
    if args.financial_only:
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
        old["financial"] = financial_manifest
        (out / "manifest.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in financial_manifest.items() if k != "excluded_benchmark_ids"}, indent=1))
        return 0

    print("public: Wikipedia en/id/vi", flush=True)
    public, per_lang = public_corpus({k: int(v * args.scale) for k, v in PUBLIC_LANGUAGES.items()})
    (out / "public.txt").write_text("\n".join(public) + "\n", encoding="utf-8")

    manifest = {
        "public": {"source": "wikimedia/wikipedia 20231101 (first shard per language)",
                   "chars_per_language": per_lang, "lines": len(public),
                   "chars": sum(map(len, public)), "licence": "CC BY-SA 4.0"},
        "financial": financial_manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v for kk, v in m.items() if kk != "excluded_benchmark_ids"}
                      for k, m in manifest.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
