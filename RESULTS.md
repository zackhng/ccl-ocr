# Phase 0–2 result

Run `20261001T142503Z` · 245 documents · CCL engine, no neural network
· full report: [`bench/results/20261001T142503Z/summary.md`](bench/results/20261001T142503Z/summary.md)

## The question

> Does connected-component labelling reliably isolate character candidates on our
> document distribution, and what does it cost in milliseconds?

## The answer

**Partially, and it depends almost entirely on capture mode.**

| metric | value |
|---|---|
| character isolation recall | **60.9%** |
| merge rate | 23.7% |
| miss rate | 14.1% |
| over-segmentation rate | 1.3% |
| junk rate (of surviving components) | 36.2% |
| small-mark retention | 75.8% |
| end-to-end P50 / P95 / P99 | **83 / 218 / 231 ms** |

Isolation recall is the ceiling Phase 3 inherits: a character CCL merged, split or
discarded cannot be recovered by any classifier, however good. 60.9% is not a number to
build a 70-class CNN on top of yet.

## The finding that matters

The aggregate hides everything. Split by capture mode:

| capture | samples | isolated | merged | missed | junk |
|---|---|---|---|---|---|
| **scan** | 66 | **74.9%** | 16.1% | 7.6% | 10.7% |
| **photo** | 84 | **43.1%** | 33.2% | 22.4% | 58.1% |

Scans are workable. Photographs are not — 43% isolation with 58% of surviving components
being junk means the classifier would spend more than half its inferences on noise and
still only see fewer than half the characters cleanly.

**This is a preprocessing problem, not a model problem.** Over-segmentation is 1.2% on
photos, so strokes are not breaking; characters are *merging* (33%) and *vanishing*
(22%). Both point at binarisation under uneven illumination and at effective resolution
after the long-side cap — not at anything a bigger network would fix.

## Latency

| stage | P50 | P95 | max |
|---|---|---|---|
| preprocess | 38.0 | 66.6 | 98.8 |
| binarize | 21.7 | 32.3 | 198.4 |
| ccl | 10.2 | 75.5 | **1111.4** |
| filter | 5.1 | 39.0 | 401.8 |
| **total** | **83.1** | **218.3** | **2114.2** |

Two things to note before any comparison with PaddleOCR:

- **`preprocess` dominates the median** at 38 ms of 83 ms, almost all of it the
  morphological background estimate. First thing to attack.
- **The tail is 10× the P99.** One document takes 2.1 s against a 231 ms P99, with CCL
  alone at 1.1 s — a noisy photograph generating a vast number of components. P99 is not
  capturing this; the max is. Worth isolating before quoting any latency figure
  externally.

## What this does not say

- **Character metrics are measured on synthetic data only.** No public dataset of
  financial documents has character-level ground truth. The 95 real documents (SROIE
  receipts, FUNSD forms, cheques) constrain the result at region level — 94.1% region
  hit rate, 64.9% coverage — which is a weaker check. Wiring up DDI-100 would give an
  external character-level check (see [`docs/DATASETS.md`](docs/DATASETS.md)).
- **No text is produced.** Phases 3–5 are not built. `isolation recall` is not OCR
  accuracy.
- **No deskew.** Rotation in the capture profiles is currently absorbed as merges and
  misses rather than corrected.

## Suggested next step

Not Phase 3. The Phase 6 PaddleOCR bake-off is cheap now that the harness exists, and
43% isolation on photographs is the number that decides whether this architecture is
viable for phone-captured NRICs and cheques at all. Run that before training anything.

If the architecture is kept, the preprocessing work is clear and bounded: per-region
binarisation on photos, a resolution floor rather than a fixed long-side cap, and
deskew.

## Reproducing this

The data is not in the repo (249 MB, plus redistribution limits on FUNSD). One command
materialises it — checking what is already on disk and downloading only the rest:

```bash
uv sync --group dev
uv run pytest                                  # 87 tests

uv run python scripts/ensure_data.py           # regenerates/downloads all 245 samples
uv run python -m ocrbench.cli run --repeats 5
uv run python -m ocrbench.cli overlays --limit 30 --stages   # look before you trust
```

`ensure_data.py` is idempotent — a second run costs a directory listing. Add `--check`
to report without downloading, or `--offline` for the synthetic half only.

Reproducibility rests on `bench/manifest.json`, which *is* committed: it pins all 245
sample ids and their sources, so the script can tell you when what is on disk does not
match what the numbers above were measured against. The synthetic half regenerates
byte-identically from seed 7 (asserted in `tests/test_synth.py`).
