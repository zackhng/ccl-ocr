# Phase 2b result — splitting merged glyphs

Published benchmark (seed 7, 245 documents, 57,548 GT characters). Thresholds tuned on
a separate held-out set (seed 11), never on this one. All four runs were made on the
same idle machine, back to back. Runs are in [`bench/results/phase2b/`](bench/results/phase2b).

| | Phase 0–2 | splitter off | **splitter (default)** | + column cut |
|---|---|---|---|---|
| isolation recall | 65.9% | 66.2% | **82.0%** | 84.3% |
| merged | 25.9% | 25.9% | **7.4%** | 4.2% |
| missed | 6.9% | 6.6% | 7.8% | 8.0% |
| over-segmented | 1.3% | 1.3% | 2.8% | 3.5% |
| P50 / P95 ms | 22.6 / 57.7 | 23.9 / 63.4 | **27.0 / 85.7** | 29.2 / 114.9 |

| capture | Phase 0–2 isolated | default isolated | Phase 0–2 merged | default merged |
|---|---|---|---|---|
| photo | 52.7% | **73.0%** | 37.8% | 12.6% |
| scan | 76.3% | **89.1%** | 16.5% | 3.3% |

**Every slice improves:**
- By template: forms 67.5% → 89.9%, plain Latin 67.8% → 87.0%, cheques 62.3% → 77.9%,
  receipts 63.5% → 74.5%, ID cards 59.3% → 68.3%.
- Han 80.2% → 85.0%. The width gate leaves square glyphs alone.
- Junk falls in every slice.

**Small-mark retention** goes from 69–79% to ~99%. That is a filter fix splitting
exposed: baseline punctuation was only ever "retained" by being merged into its
neighbour. See `docs/DESIGN.md` §10.

**Real documents (no character GT) hold:**
- Line recall: SROIE 95.6% → 95.4%, FUNSD 96.0% → 96.6%.
- Word hit rate: SROIE 98.4% → 98.4%, FUNSD 92.2% → 94.6%.

**What it costs:**
- **Latency:** +4.4 ms at P50 and +28 ms at P95. The P95 cost is cheques: 2365 px
  scans with ~400 merge suspects each, at ~18 ms of splitting apiece. The slowest
  document (a pathologically noisy receipt) goes from 472 to 498 ms. The suspect cap
  bounds splitting at 28 ms on any page.
- **Misses rise slightly** on degraded captures: photo 8.2% → 10.2%, hard_photo
  12.1% → 18.4%. An accepted re-threshold applies the finer threshold to the whole
  suspect, which on the worst captures can break a stroke in one of the glyphs it
  separated. Isolation still improves on net in every slice.

**The column cut is implemented but off by default.** It adds 2.3 points here. On
Arabic or Devanagari it would cut through letters that are joined by design, and
script detection (Phase 7) does not exist yet. Phase 7 should enable it per region
once regions are tagged Latin.

---

# Phase 0–2 result

Run `20261001T145727Z` · 245 documents · CCL engine, no neural network
· full report: [`bench/results/20261001T145727Z/summary.md`](bench/results/20261001T145727Z/summary.md)

## The question

> Does connected-component labelling reliably isolate character candidates on our
> document distribution, and what does it cost in milliseconds?

## The answer

**Workable on scans, not yet on photographs.**

| metric | value |
|---|---|
| character isolation recall | **65.9%** |
| merge rate | 25.9% |
| miss rate | 6.9% |
| over-segmentation rate | 1.3% |
| junk rate (of surviving components) | 22.7% |
| small-mark retention | 74.5% |
| end-to-end P50 / P95 / P99 | **87 / 266 / 322 ms** |

Isolation recall is the ceiling Phase 3 inherits: a character CCL merged, split or
discarded cannot be recovered by any classifier, however good.

## The finding that matters

The aggregate hides everything. Split by capture mode:

| capture | samples | isolated | merged | missed | junk |
|---|---|---|---|---|---|
| **scan** | 66 | **76.3%** | 16.5% | 5.9% | 6.0% |
| **photo** | 84 | **52.7%** | 37.8% | 8.2% | 39.8% |

**The dominant failure is merging, not loss.** Over-segmentation is 1.3% and misses are
down to 6.9%, so strokes are not breaking and characters are not vanishing — adjacent
glyphs are fusing into single components, at more than twice the rate on photographs as
on scans. That is a function of effective resolution and of binarisation under uneven
illumination. It is not something a larger classifier can fix, because by the time the
classifier sees the crop the two characters are already one blob.

The 39.8% junk rate on photos is a latency problem rather than an accuracy one: two in
five components reaching the classifier would be speckle.

## A bug worth recording

The first run of this benchmark reported 60.9% overall and **25.1% on ID cards**. That
was not the pipeline being bad at ID cards; it was the scale estimator failing.

Noise specks pass Tier 1 by design — a decimal point is 4 px, and a filter that drops it
turns `1,234.56` into `123456` on a cheque. On a noisy page those specks *outnumber the
glyphs*, so a plain median of component heights collapsed onto the speck population:
**4 px measured where the real glyphs were 17 px**. Every relative gate keys off that
number, so the "4× median" ceiling landed at 16 px and routed the largest and most
important text on the card — the ID number and the name — to `BLOB`. Roughly 130 of 160
characters per card, thrown away by the filter.

Replaced with an ink-weighted median (a speck carries ~4 px of ink, a glyph ~100),
winsorised at the 95th percentile so a single portrait photo cannot drag it the other
way. Estimator error against ground truth fell from 30% to 17%; ID-card isolation went
25.1% → 59.3% and its miss rate 67.5% → 18.5%.

The lesson is in `docs/DESIGN.md`: the aggregate said "photos are hard", which is true
and was not the problem. The per-template slice said "ID cards are broken", which was.

## Latency

| stage | P50 | P95 | max |
|---|---|---|---|
| preprocess | 38.1 | 71.6 | 102.5 |
| binarize | 21.8 | 31.6 | 145.5 |
| ccl | 10.4 | 76.6 | **991.8** |
| filter | 8.3 | 88.6 | 427.4 |
| **total** | **87.3** | **265.5** | **1961.7** |

Two things to settle before any comparison with PaddleOCR:

- **`preprocess` dominates the median** at 38 ms of 87 ms, almost all of it the
  morphological background estimate. First thing to attack.
- **The tail is 6× the P99.** One document takes 2.0 s against a 322 ms P99, with CCL
  alone at 1.0 s — a noisy photograph generating a vast number of components. The P99 is
  not capturing this; the max is. Isolate it before quoting any latency figure
  externally.

## What this does not say

- **Character metrics are measured on synthetic data only.** No public dataset of
  financial documents has character-level ground truth. The 95 real documents (SROIE
  receipts, FUNSD forms, cheques) constrain the result at region level — 97.7% region
  hit rate, 69.0% coverage — which is a weaker check. Wiring up DDI-100 would give an
  external character-level check (see [`docs/DATASETS.md`](docs/DATASETS.md)).
- **No text is produced.** Phases 3–5 are not built. Isolation recall is not OCR
  accuracy.
- **No deskew.** Rotation in the capture profiles is currently absorbed as merges rather
  than corrected, which inflates the merge rate by an unmeasured amount.

## Suggested next step

Not Phase 3. Two things first, in this order:

1. **Run the Phase 6 PaddleOCR bake-off.** The harness already exists and the `Engine`
   protocol is already in place, so this is a day of work, and 52.7% isolation on
   photographs is the number that decides whether this architecture is viable for
   phone-captured NRICs and cheques at all. Training a classifier against a 53% ceiling
   before checking the alternative would be building on an unvalidated premise.
2. **Attack merging on photographs**, which is bounded and well-identified work: a
   resolution floor rather than a fixed long-side cap, per-region binarisation, and
   deskew. Measure with `--captures photo` and watch the merge rate.

## Reproducing this

The data is not in the repo (249 MB, plus redistribution limits on FUNSD). One command
materialises it — checking what is already on disk and downloading only the rest:

```bash
uv sync --group dev
uv run pytest                                  # 90 tests

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
