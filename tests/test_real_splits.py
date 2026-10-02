"""Train/test separation for the real-document data (Phase 3).

A leak here would not crash anything — it would just make every CER we report a
measurement of memorisation. One was caught by hand during development: the LM
fine-tuning corpus briefly included FUNSD's testing_data, which ``bench_real`` uses as
test documents. These tests make the next one fail loudly. They skip when the data has
not been fetched (it is not in the repo).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocrbench.gt import BenchmarkStore

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "real" / "train"
TEST = ROOT / "bench_real"
FINANCIAL = ROOT / "data" / "corpora" / "financial.txt"


def _ids(root: Path) -> set[str]:
    if not (root / "gt").exists():
        pytest.skip(f"{root} not fetched")
    return set(BenchmarkStore(root).sample_ids())


def test_train_and_test_stores_share_no_document():
    assert not (_ids(TRAIN) & _ids(TEST))


def test_published_benchmark_is_not_in_training():
    published = {s["sample_id"] for s in json.loads((ROOT / "bench" / "manifest.json").read_text())["samples"]}
    assert not (_ids(TRAIN) & published)


def test_test_documents_do_not_feed_the_language_model():
    """No benchmark transcript line of real substance may appear verbatim in the LM
    fine-tuning corpus. Short lines ("TOTAL", "CASH") legitimately recur across
    documents, so only distinctive lines (>= 25 characters) are checked."""
    if not FINANCIAL.exists():
        pytest.skip("corpora not built")
    from ocr.recog.charset import normalise

    corpus = set(FINANCIAL.read_text(encoding="utf-8").splitlines())
    store = BenchmarkStore(TEST)
    leaked = []
    for sid in _ids(TEST):
        s = store.read(sid)
        texts = [g.text for g in s.lines] or [g.text for g in s.words]
        for t in texts:
            n = normalise(t)
            if len(n) >= 25 and n in corpus:
                leaked.append((sid, n))
    # Train and test splits of public datasets share issuers (one shop's header and
    # address across many receipts), so fetch_corpora.py decontaminates line by line.
    # After that, nothing may remain.
    assert not leaked, leaked[:5]
