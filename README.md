# ocr-engine — Phases 0–2 + Phase 6 bake-off

An in-house OCR engine for financial documents (NRIC photos, cheques, receipts, forms).
The target architecture is:

```
image → preprocess → binarize → CCL → component filtering → tiny CNN → reconstruction
```

**This repository currently implements Phases 0–2**: the benchmark harness, the
connected-component engine, and the geometric filtering that protects the classifier.
It also implements **the Phase 6 PaddleOCR bake-off**, which checks whether the
architecture is worth building before anything is trained. There is no neural network
and no text output from the CCL engine yet. The deliverable is a number:

> Does connected-component labelling reliably isolate character candidates on *our*
> document distribution, and what does it cost in milliseconds?

If isolation recall is poor on photographed cheques and ID cards, every later phase
inherits that ceiling — which is why this is measured before anything is trained.

## Quickstart

```bash
uv sync --group dev
uv run pytest                                   # 138 tests

# Materialise the benchmark: checks what is already on disk, downloads only what is
# missing. Idempotent. --check reports without downloading; --offline skips the network.
uv run python scripts/ensure_data.py

# Look at what CCL actually does -- do this before trusting any number
uv run python -m ocrbench.cli overlays --limit 30 --stages

# Measure it
uv run python -m ocrbench.cli run --repeats 5

# Phase 6: CCL against PaddleOCR on the same documents (needs the baselines extra)
uv sync --group dev --extra baselines
uv run python -m ocrbench.cli compare --engines ccl paddle-det paddle --repeats 5 \
    --out bench/results/phase6 --note "idle machine"
```

Each engine's `results.json` and `summary.md` are written as soon as that engine
finishes. Re-running the same command with the same `--out` loads the engines already
saved and runs only the missing ones. Latency is only meaningful on an idle machine, so
record the conditions with `--note`. The report flags runs whose notes, thread counts or
machines differ.

To compare a later pipeline of ours against a baseline measured earlier, without
re-measuring Paddle:

```bash
uv run python -m ocrbench.cli report --candidate ccl-cnn --out bench/results/cnn-vs-paddle \
    --runs bench/results/phase6/paddle bench/results/phase6/paddle-det \
           ccl-cnn=bench/results/<new-run>
```

On Windows machines with Application Control (Smart App Control / WDAC), `uv sync` can
fail building the project itself. Use `uv sync --no-install-project ...` and run with
`PYTHONPATH=src`.

**[`RESULTS.md`](RESULTS.md) has the current answer.** Short version: 65.9% character
isolation recall overall, but 76.3% on scans against 52.7% on photographs, and the
dominant failure is adjacent glyphs *merging* rather than being lost. That gap, not the
aggregate, is what decides the next step.

## Layout

| Path | What it is |
|---|---|
| `src/ocr/` | The engine. `preprocess` → `binarize` → `ccl` → `filters`, wired by `engine.py`. |
| `src/ocr/visualize.py` | Overlay rendering. The real Phase 1 gate — look before you trust a metric. |
| `src/ocrbench/synth/` | Synthetic document generator. The only source of exact per-glyph ground truth, and the only source of NRIC-shaped documents at all. |
| `src/ocrbench/adapters/` | Normalise each public dataset into one on-disk ground-truth format. |
| `src/ocr/paddle_engine.py` | PaddleOCR 3.x baseline behind the same `Engine` protocol (Phase 6). |
| `src/ocrbench/metrics.py` | Character isolation recall, over-segmentation, merge, junk, diacritic retention; plus engine-agnostic line localisation and CER. |
| `src/ocrbench/compare.py` | Phase 6 comparison report and go/no-go verdict. |
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
