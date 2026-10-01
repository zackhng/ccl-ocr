# ocr-engine — Phases 0–2

An in-house OCR engine for financial documents (NRIC photos, cheques, receipts, forms).
The target architecture is:

```
image → preprocess → binarize → CCL → component filtering → tiny CNN → reconstruction
```

**This repository currently implements Phases 0–2 only**: the benchmark harness, the
connected-component engine, and the geometric filtering that protects the classifier.
There is no neural network and no text output yet. The deliverable is a number:

> Does connected-component labelling reliably isolate character candidates on *our*
> document distribution, and what does it cost in milliseconds?

If isolation recall is poor on photographed cheques and ID cards, every later phase
inherits that ceiling — which is why this is measured before anything is trained.

## Quickstart

```bash
uv sync --group dev
uv run pytest                                   # 87 tests

# Materialise the benchmark: checks what is already on disk, downloads only what is
# missing. Idempotent. --check reports without downloading; --offline skips the network.
uv run python scripts/ensure_data.py

# Look at what CCL actually does -- do this before trusting any number
uv run python -m ocrbench.cli overlays --limit 30 --stages

# Measure it
uv run python -m ocrbench.cli run --repeats 5
```

**[`RESULTS.md`](RESULTS.md) has the current answer.** Short version: 60.9% character
isolation recall overall, but 74.9% on scans against 43.1% on photographs — and that
gap, not the aggregate, is what decides the next step.

## Layout

| Path | What it is |
|---|---|
| `src/ocr/` | The engine. `preprocess` → `binarize` → `ccl` → `filters`, wired by `engine.py`. |
| `src/ocr/visualize.py` | Overlay rendering. The real Phase 1 gate — look before you trust a metric. |
| `src/ocrbench/synth/` | Synthetic document generator. The only source of exact per-glyph ground truth, and the only source of NRIC-shaped documents at all. |
| `src/ocrbench/adapters/` | Normalise each public dataset into one on-disk ground-truth format. |
| `src/ocrbench/metrics.py` | Character isolation recall, over-segmentation, merge, junk, diacritic retention. |
| `src/ocrbench/runner.py` | Per-stage P50/P95/P99 latency + metrics, sliced by capture mode / script / DPI. |
| `scripts/` | Dataset acquisition and the semi-automatic ground-truth review tool. |

## Ground truth format

Every source — real or synthetic — is normalised to one layout so the runner never
knows where an image came from:

```
bench/images/<sample_id>.png
bench/gt/<sample_id>.json
```

```jsonc
{
  "sample_id": "sroie_0042",
  "source": "sroie", "capture": "scan", "script": "latin", "dpi": 300,
  "text": "full reading-order text",
  "lines": [{"bbox": [x, y, w, h], "text": "..."}],
  "words": [{"bbox": [x, y, w, h], "text": "..."}],
  "chars": [{"bbox": [x, y, w, h], "char": "A"}],   // null when unavailable
  "pii":   [{"bbox": [x, y, w, h], "type": "nric|account|name|face"}],
  "regions_nontext": [{"bbox": [x, y, w, h], "type": "photo|logo|signature|rule"}]
}
```

`chars` is populated for synthetic samples and DDI-100, and `null` for SROIE, FUNSD and
the cheque sets. Every metric degrades to a word-level equivalent when it is `null`.

PII regions are **annotated but not acted on** — detection and masking are deliberately
out of scope until OCR quality is known.

## Datasets

See `docs/DATASETS.md` for acquisition commands, licences, and the caveats that matter
(MIDV-2020 is gated and 124 GB; DDI-100 is Cyrillic and must not be folded into
headline accuracy).

## Design notes

`docs/DESIGN.md` records the decisions that are easy to get wrong and expensive to
reverse — polarity handling, why filtering routes rather than deletes, and what the
isolation metrics do and do not prove.
