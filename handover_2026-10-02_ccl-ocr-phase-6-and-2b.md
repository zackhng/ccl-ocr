# Handover: Phase 6 tooling and Phase 2b merge splitting

**Date:** 2026-10-02
**Repo:** https://github.com/zackhng/ccl-ocr. Default branch `master`.
**Branches** (stacked, neither merged nor pushed):
- `phase-6-paddle-bakeoff` cut from `master`: the PaddleOCR baseline and comparison
  tooling.
- `phase-2b-merge-splitting` cut from that: the merge splitter and the punctuation
  fix.

**Status:**
- Phase 2b is complete and measured.
- Phase 6 tooling is complete, but **its full measurement has not been run**. It needs
  an idle machine, and the first attempt was stopped because a GPU training job was
  sharing the CPU.

Read the Phase 0–2 handover (`handover_2026-10-01_ccl-ocr-phases-0-2.md`) first. This
one only covers what changed.

---

## 1. Phase 2b: merge splitting

**Result**, published benchmark, same idle machine, tuned on a held-out set:

| | Phase 0–2 | **now** |
|---|---|---|
| isolation recall | 65.9% | **82.0%** |
| photo / scan isolated | 52.7% / 76.3% | **73.0% / 89.1%** |
| merged | 25.9% | 7.4% |
| missed | 6.9% | 7.8% |
| small-mark retention | ~70–79% | ~99% |
| P50 / P95 ms | 22.6 / 57.7 | 27.0 / 85.7 |

Every template, script and capture slice improves. Full tables are in `RESULTS.md`, and
the runs are in `bench/results/phase2b/`.

**What it does.** `src/ocr/split.py` runs between CCL and filtering:
1. Components wider than 1.2 glyph widths are suspects.
2. Each suspect is re-thresholded inside its own mask with a window of 0.4× the glyph
   height.
3. The result is kept only if it yields two or more whole-height parts.

Reasoning and the rejected alternatives are in `docs/DESIGN.md` §10. The short version:
- not resolution, because photos are already upscaled;
- not deskew, because the merged neighbours' gap is 0 px;
- not a smaller window everywhere, because that plateaus at 68–70% and breaks strokes.

### Things to know before changing it
- **Tune on `bench_tune`, never on `bench`.** Rebuild the tuning set with
  `ocrbench.cli synth --count 150 --seed 11 --out bench_tune`, then run
  `scripts/tune_split.py`, which refuses to run on the published benchmark.
- **The column cut (`split_touching_columns`) is off on purpose.** It adds 2.3 points on
  today's Latin/Han benchmark but would cut through Arabic and Devanagari. Phase 7
  should turn it on per region once regions are tagged Latin.
- **The splitter and the filter share the glyph-scale estimator.** The splitter calls
  `ink_weighted_median_height_arrays` and `tier1_text_mask`. Tests assert these match the
  per-component versions exactly. Keep them as one implementation.
- **Relabelling is local.** Splitting only removes ink inside one component, so the
  pieces are labelled inside the suspect's box and swapped in.
  `test_local_relabel_equals_full_relabel` pins that this is exact. A full-page relabel
  cost +40–60 ms per cheque.

### A filter bug splitting exposed
Baseline punctuation (`.` `,` `-` `:`) sits *beside* its neighbour, so `_find_parent`
(which only looks above and below) never found it. Such marks survived only when merged
into the glyph before them. Once the splitter separated them, they were routed to NOISE.
`_find_line_neighbour` now keeps them as TEXT, reason `t2:punctuation`. The Phase 0–2
small-mark retention figure (74.5%) was inflated by merges all along.

### Known costs
- **Latency:** P95 +28 ms, almost all of it on cheques (~400 suspects each, ~18 ms of
  splitting). The cheapest remaining win is to batch the per-suspect CCL into one
  page-wide pass over the union of suspect masks. That is untried.
- **Misses:** up slightly on degraded photos (hard_photo 12.1% → 18.4%). An accepted
  re-threshold applies the finer threshold to the whole suspect, which can break a
  stroke in one of the separated glyphs.
- **Slowest document:** 472 → 498 ms. This is a pathologically noisy receipt whose
  glyph estimate is 5 px. The suspect cap (1,500, widest first) bounds splitting at
  28 ms.

## 2. Phase 6: PaddleOCR bake-off (tooling done, run pending)

What exists:
- `PaddleEngine` (`src/ocr/paddle_engine.py`) runs PP-OCRv5 mobile on CPU with oneDNN,
  in full and detection-only modes.
- **paddlepaddle is pinned to 3.1.x.** 3.3.x crashes with oneDNN on Windows, and running
  without oneDNN is 3–5× slower, which would flatter CCL. 3.0.0 is blocked by
  Application Control on this machine.
- Engine-agnostic metrics:
  - `region_metrics` scores line localisation on horizontal span, so glyph boxes and
    padded line boxes score alike;
  - `text_metrics` scores CER per connected cluster of lines, whitespace-free and NFKC,
    with case folded for SROIE.
  - Four scoring bugs were found and fixed while building these; see `DESIGN.md` §9.
- `ocrbench.cli compare`:
  - saves each engine's results as soon as it finishes;
  - resumes from a partial `--out` folder;
  - records a `--note` with each run;
  - prints a go/no-go verdict against criteria fixed beforehand: the candidate's
    P50/P95 at most 50% of Paddle's, and line recall within 10 points of
    `paddle-det`, overall and on photos.
- `ocrbench.cli report` rebuilds a comparison from saved runs over their common
  documents. Pass `--candidate` to judge a future pipeline (e.g. `ccl-cnn`) against a
  Paddle baseline measured once.

**To finish Phase 6**, on an idle machine (close GPU jobs, since dataloaders use the
CPU):

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m ocrbench.cli compare \
    --engines ccl paddle-det paddle --repeats 5 --out bench/results/phase6 --note "idle machine"
```

- It takes about 30 minutes, and Paddle recognition dominates (~20 ms per text line on
  this CPU).
- Run it from `phase-2b-merge-splitting`, so `ccl` is the improved engine.
- Then write the Phase 6 section of `RESULTS.md` from `bench/results/phase6/comparison.md`.

Early, indicative numbers from 12-document smoke runs (not results):
- Paddle CER is ~0% on synthetic Latin and Han, 6–10% on SROIE, and 3–19% on FUNSD.
- Paddle drops spaces ("ROCNO:538358-H").
- CCL is far faster at the CCL stages alone, but it does not recognise anything yet.

## 3. Environment notes (this machine)
- Python 3.11 comes from uv's managed interpreters; `py -3.11` points at a missing
  install. Install uv with `python -m pip install --user uv`.
- **Application Control blocks uv's temporary build environments.** Use
  `uv sync --group dev --extra baselines --extra fetch --no-install-project` and run
  everything with `PYTHONPATH=src`.
- Latency published in the Phase 0–2 handover (87 ms P50) was measured under different
  conditions. On this idle machine the same code measures 22.6 ms. Compare only
  same-machine runs; `bench/results/phase2b/phase0-2_same_machine` exists for that.

## 4. Next, in order
1. **Run the Phase 6 bake-off** (above) and record the verdict.
2. If it is GO: start **Phase 5 grouping + script tagging**. That lets the column cut be
   enabled per Latin region, and is the prerequisite for multi-language routing, since
   one document can mix scripts. Then Phase 3, the CNN.
3. Optional latency work: batch the per-suspect CCL in the splitter, and attack
   `preprocess` (still the largest stage at ~11 ms P50).
