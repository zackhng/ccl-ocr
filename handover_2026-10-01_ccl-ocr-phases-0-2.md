# Handover — CCL OCR engine, Phases 0–2

**Date:** 2026-10-01
**Objective:** Build the benchmark harness, the connected-component engine, and the
geometric component filtering, and answer with numbers whether CCL isolates character
candidates well enough to build a classifier on.
**Repo:** https://github.com/zackhng/ccl-ocr — default branch **`master`** (not `main`)
**Status:** Complete. 90 tests passing. Measured over 245 documents.

---

## 1. What this is

An in-house OCR engine for financial documents — NRIC photos, cheques, receipts, forms.
The target architecture is:

```
image → preprocess → binarize → CCL → component filtering → tiny CNN → reconstruction
```

with per-script fallbacks for Devanagari/Han/Arabic/Thai later. The motivation is
latency: a general-purpose detector+recognizer like PaddleOCR is heavier than we want
for mostly-printed, mostly-structured documents.

**Only Phases 0–2 are built.** There is no neural network and no text output. The
deliverable is a measurement.

## 2. The answer

> Does CCL reliably isolate character candidates on our document distribution, and what
> does it cost in milliseconds?

**Workable on scans, not yet on photographs.**

| metric | value |
|---|---|
| character isolation recall | **65.9%** |
| merge rate | 25.9% |
| miss rate | 6.9% |
| over-segmentation rate | 1.3% |
| junk (of surviving components) | 22.7% |
| end-to-end P50 / P95 / P99 | **87 / 266 / 322 ms** |

| capture | samples | isolated | merged | missed | junk |
|---|---|---|---|---|---|
| scan | 66 | **76.3%** | 16.5% | 5.9% | 6.0% |
| photo | 84 | **52.7%** | 37.8% | 8.2% | 39.8% |

Isolation recall is the **ceiling Phase 3 inherits**: a character CCL merged, split or
discarded cannot be recovered by any classifier, however good.

**The dominant failure is merging, not loss.** Over-segmentation is 1.3% and misses are
under 7%, so strokes are not breaking and characters are not vanishing — adjacent glyphs
are fusing into single components, at more than twice the rate on photos as on scans.
That is effective resolution and binarisation under uneven illumination. It is not
something a larger classifier can fix, because by the time the classifier sees the crop
the two characters are already one blob.

Full report: [`RESULTS.md`](RESULTS.md) and
[`bench/results/20261001T145727Z/summary.md`](bench/results/20261001T145727Z/summary.md).

## 3. Where things live

| Path | What it is |
|---|---|
| `src/ocr/` | The engine. `preprocess` → `binarize` → `ccl` → `filters`, wired by `engine.py`. |
| `src/ocr/config.py` | Every tunable in one dataclass, so the runner can sweep them. |
| `src/ocr/visualize.py` | Overlay rendering, colour-coded by routing decision. |
| `src/ocrbench/synth/` | Synthetic document generator — the only source of per-glyph ground truth, and of NRIC-shaped documents at all. |
| `src/ocrbench/adapters/` | One per dataset, normalising into the unified GT format. |
| `src/ocrbench/metrics.py` | Isolation metrics + region-level fallback. |
| `src/ocrbench/runner.py` | Per-stage latency percentiles, metric aggregation. |
| `src/ocrbench/report.py` | The sliced markdown report. |
| `scripts/ensure_data.py` | Materialises the benchmark; downloads only what is missing. |
| `docs/DESIGN.md` | **Read this first.** Decisions that are easy to get wrong. |
| `docs/DATASETS.md` | Acquisition, licences, and what each source can/cannot measure. |

## 4. How to run it

```bash
uv sync --group dev
uv run pytest                                  # 90 tests

uv run python scripts/ensure_data.py           # regenerates/downloads all 245 samples
uv run python -m ocrbench.cli run --repeats 5
uv run python -m ocrbench.cli overlays --limit 30 --stages
```

`ensure_data.py` is idempotent; `--check` reports without downloading, `--offline` does
the synthetic half only.

**Benchmark data is not in the repo** — 249 MB, and FUNSD's licence restricts
redistribution. Nothing is lost: the synthetic half regenerates byte-identically from
seed 7 (asserted in `tests/test_synth.py`), the real half re-downloads, and
`bench/manifest.json` *is* committed so the script can tell you when what is on disk
does not match what the published numbers were measured against.

## 5. The benchmark

245 documents: 150 synthetic + 95 real.

| source | n | GT granularity | licence | adapter |
|---|---|---|---|---|
| synthetic | 150 | **characters** + words, lines, PII, non-text | ours | n/a |
| SROIE receipts | 40 | lines | CC-BY-4.0 | verified |
| FUNSD forms | 25 | words + lines | non-commercial research only | verified |
| cheques | 30 | field regions only | Apache-2.0 | verified |
| MIDV-2020 | 0 | document + face quadrangles | access form, 124 GB | **unverified** |
| DDI-100 | 0 | **characters** | see repo | **unverified** |

"Verified" = downloaded and converted end to end. "Unverified" = written against the
documented format, never run on the real archive; raises `AdapterError` with what it
actually found rather than emitting plausible nonsense. **Do not trust a number from an
unverified adapter without checking the conversion first.**

Cheque and MIDV adapters deliberately emit **no** line/word boxes — their annotations are
coarse field regions (a cheque's `name` box spans the full 2365 px width), so scoring
coverage against them would produce a number that looks like a failure and means
nothing. They contribute latency, component routing, and PII/signature regions instead.
`scripts/review_gt.py` is the path to giving them real text GT.

## 6. Design decisions worth not re-litigating

Full reasoning in `docs/DESIGN.md`. The load-bearing ones:

- **Filtering routes, it does not delete.** Only `NOISE` is discarded. Diacritics, rules
  and blobs are labelled and kept — a dropped diacritic is an accuracy loss no later
  stage can undo, and on a cheque it turns `1,234.56` into `123456`.
- **Polarity is handled twice**: globally (Otsu, before illumination correction) and
  locally (`recover_inverted_regions`, for the dark banners on ID cards that otherwise
  binarise into one solid blob).
- **`pixel_area` stays in processing coordinates** while boxes are mapped back to the
  original frame. `fill_ratio` is therefore computed once at labelling time and stored;
  deriving it later mixes coordinate systems and can exceed 1.0.
- **Rates aggregate from counts, not averaged per-sample rates.** Percentiles are taken
  across *documents*, not across timing repeats.
- **Shaped scripts are skipped, not faked.** Without Pillow's Raqm backend, Devanagari/
  Arabic/Thai render without shaping or bidi, so those strata are dropped rather than
  generated wrong. `python -m ocrbench.cli fonts` reports what this machine can render.

## 7. Bugs found and fixed — read before changing the filter

Five, of which three were caught only by tests. All of them produced plausible-looking
output while being wrong, which is the failure mode to watch for here.

1. **`background + 1` wrapped 255→0 in uint8**, zeroing every white page.
2. **`connectivity` was silently ignored.** OpenCV's positional signature is
   `(image, labels, stats, centroids, connectivity, ltype)`, so passing it third landed
   it in an output slot and left connectivity at its default. **Always use keyword args
   with `connectedComponentsWithStats`.**
3. **Otsu's threshold is inclusive.** Comparing with `<` instead of `<=` inverted the
   polarity check on exactly the most strongly bimodal images.
4. **Inverted-region recovery masked out its own result.** In a dark banner the ink *is*
   the background around the letters and the letters are holes, so masking to the blob's
   ink erased the characters recovery had just found. Fixed by filling holes first.
5. **The scale estimator collapsed onto noise specks.** This was the big one — see below.

### The scale-estimator bug, because it will come back if someone "simplifies" it

Noise specks pass Tier 1 by design (a decimal point is 4 px). On a noisy page they
*outnumber the glyphs*, so a plain median of component heights collapsed onto the speck
population: **4 px measured where the real glyphs were 17 px**. Every relative gate keys
off that number, so the "4× median" ceiling landed at 16 px and routed the largest and
most important text on an ID card — the number and the name — to `BLOB`. Roughly 130 of
160 characters per card, discarded by the filter. ID-card isolation recall was 25%.

Replaced with an **ink-weighted median winsorised at the 95th percentile**:

- Ink weighting separates the populations — a speck carries ~4 px of ink, a glyph ~100 —
  so specks can outnumber glyphs ten to one without moving the statistic.
- Winsorising guards the mirror-image failure, where one portrait photo carries more ink
  than every glyph combined.
- **The percentile must be high.** 75th was tried and fails in precisely the case this
  exists for: when specks dominate by count, the 75th percentile of *areas* is itself a
  speck, every weight clips to 4, and it degenerates back to a plain median.
- Dense regions are routed *before* the estimate, not after.

Result: estimator error 30% → 17%; ID cards 25.1% → 59.3% isolated, 67.5% → 18.5%
missed. Three regression tests in `tests/test_filters.py::TestScaleEstimator` pin it at
both ends.

**How it was found matters as much as the fix.** The aggregate said "photos are hard" —
true, and not the problem. The per-template slice said "ID cards are broken" — that was
it. Read the slices, not the headline.

## 8. Ground truth can be wrong too

Three GT bugs were found and fixed, all of which would have silently depressed the
numbers forever: a receipt's `"-" * 42` separator annotated as 42 characters that CCL
correctly merges into one rule; an ID portrait drawn past its own bounding box; and a
receipt canvas computed too short, clipping trailing lines off the image while the GT
still claimed their characters.

Hence `tests/test_synth.py::test_every_char_box_contains_ink` and
`test_degraded_boxes_still_contain_ink`. **Keep these.** A benchmark grading the pipeline
against fiction would grade it *consistently*, so nothing downstream would ever look
wrong.

## 9. What to do next — in this order

**Do not start Phase 3 (the CNN) yet.**

1. **Phase 6 PaddleOCR bake-off.** The harness exists and the `Engine` protocol is
   already in place (`src/ocr/engine.py`), so this is roughly a day. 52.7% isolation on
   photographs is the number that decides whether this architecture is viable for
   phone-captured NRICs and cheques at all. Install with `uv sync --extra baselines`,
   implement a `PaddleEngine` satisfying `Engine`, run both through
   `ocrbench.runner.run`. Training a classifier against a 53% ceiling before checking
   the alternative is building on an unvalidated premise.

2. **Attack merging on photographs.** Bounded, well-identified work:
   - a resolution *floor* rather than a fixed `long_side_cap` — merging tracks effective
     DPI directly
   - per-region binarisation instead of one global adaptive threshold
   - deskew (there is none; rotation currently shows up as merges)

   Measure with `--captures photo` and watch the merge rate, not the headline.

3. **Latency, before quoting any number externally.** `preprocess` is 38 ms of the 87 ms
   median, almost all of it the morphological background estimate. And the tail is 6× the
   P99 — one document takes 2.0 s against a 322 ms P99, with CCL alone at 1.0 s. Find
   that document (`results.json` has per-sample timings) before publishing a latency
   figure.

4. **Get an external character-level check.** See §10.

## 10. Known limitations / open questions

- **Character metrics run on synthetic data only.** At that level the benchmark is
  grading its own homework. The 95 real documents constrain it at region level, which is
  weaker. DDI-100 would give an external check — but it is Cyrillic, so it is a geometry
  stress test, not an accuracy claim, and must be reported separately (the adapter tags
  `script="cyrillic"` so the per-script slice does this automatically).
- **MIDV-2020 is not in the benchmark.** Request access on day one; it is 124 GB behind
  a form and not on the critical path. Pull only the `scan` and `photo` subsets.
- **No deskew**, so the merge rate is inflated by an unmeasured amount.
- **Ground-truth boxes are axis-aligned**, so under perspective warp we store the
  enclosing box, which depresses measured IoU slightly on warped samples.
- **Licensing is unresolved for production.** FUNSD is non-commercial research only;
  IDRBT states no licence; SROIE's terms vary by mirror. Fine for an internal benchmark,
  a different question for training a shipped model. The synthetic generator has none of
  these constraints — another argument for leaning on it.
- **PII is annotated but not acted on.** No detection or masking code exists; that was
  deliberately deferred until OCR quality is known.

## 11. Repo housekeeping

- Default branch is **`master`**, not `main`. To change it:
  `git branch -m master main && git push -u origin main && gh repo edit zackhng/ccl-ocr --default-branch main`
  (breaks existing clones and permalinks — do it now or not at all).
- Public. Nothing third-party is redistributed: the only committed data files are
  `bench/manifest.json` (ids, source, capture, dpi, counts) and the results JSON
  (timings and metric counts). No transcripts, boxes or images.
- Python is pinned to `>=3.11,<3.13`; the system Python 3.14 has no OpenCV wheel.
  OpenCV resolves to 5.0 — note bug #2 above, whose signature behaviour differs from
  what most OpenCV 4 examples online show.
