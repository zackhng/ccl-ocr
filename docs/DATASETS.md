# Datasets

No public dataset covers Singapore NRIC or cheques with character-level text ground
truth. The benchmark is therefore built from two halves that cover each other's
weakness:

- **Real documents** supply the distribution — actual paper, print, scanner and camera
  artefacts, real layouts. They mostly stop at word or line annotations.
- **Synthetic documents** supply exact per-glyph ground truth, and are the only source
  of NRIC-shaped documents at all.

Character-level metrics are measured on the synthetic half; the real half keeps the
synthetic half honest about latency, component routing, and region coverage.

## Status

| Source | Role | GT granularity | Licence | Adapter |
|---|---|---|---|---|
| synthetic generator | ID cards, cheques, receipts, forms, multi-script pages | **characters**, words, lines, PII, non-text | ours | n/a |
| [ICDAR-2019 SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | real scanned receipts | lines | CC-BY-4.0 | **verified** |
| [FUNSD](https://github.com/crcresearch/FUNSD) | real degraded scanned forms | words + lines | non-commercial research only | **verified** |
| [synthetic cheques](https://huggingface.co/datasets/jaganadhg/cheque-synthetic-images) | real cheque paper, print, signatures | field regions only | Apache-2.0 | **verified** |
| [MIDV-2020](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html) | real ID cards and passports | document + face quadrangles | accept via access form | **unverified** |
| [DDI-100](https://github.com/machine-intelligence-laboratory/DDI-100) | character-box geometry stress test | **characters** | see repository | **unverified** |

"Verified" means downloaded and converted end to end on this machine. "Unverified" means
the adapter is written against the documented format and has never seen the real
archive; it raises `AdapterError` with what it actually found rather than emitting
plausible-looking nonsense. Do not trust a number from an unverified adapter without
checking the conversion first.

## Fetching

```bash
# Everything reachable, stratified (~250 samples)
uv run python scripts/build_benchmark.py --total 250

# Or individually
uv run python -m ocrbench.cli synth --count 150     # no network needed
uv run python scripts/fetch_sroie.py   --limit 40
uv run python scripts/fetch_funsd.py   --limit 25
uv run python scripts/fetch_cheques.py --limit 30
```

SROIE and the cheque set are pulled through the Hugging Face **datasets-server rows
API** rather than the parquet files — the SROIE parquet split is ~500 MB for 987
receipts and we want 40, so the rows endpoint serves annotations as JSON plus a signed
image URL and avoids both the download and a pyarrow dependency.

### MIDV-2020 — gated, 124 GB

Cannot be automated. Accept the licence through the Google Form on the project page to
get sFTP credentials at the University of La Rochelle. **Request access on day one** —
it is not on the critical path, and the synthetic ID-card generator covers the slice
meanwhile. Pull only the `scan` and `photo` subsets; the 1000 video clips are the bulk
of the 124 GB and are useless to a still-image pipeline.

```bash
uv run python scripts/fetch_midv.py --raw /path/to/midv2020 --limit 60
```

### DDI-100 — manual download, and read the caveat

The only large *real* source of character boxes, which makes it the one external check
on an isolation metric that otherwise only ever runs on data we generated ourselves.

Two things to hold onto:

1. **It is Cyrillic.** Russian document pages. They exercise CCL geometry properly —
   touching glyphs, distortion, stamps over text — but they are not our script. The
   adapter tags them `script="cyrillic"` so the report's per-script slice separates them
   automatically. **Never fold them into headline accuracy.**
2. **Annotations are pickles**, and unpickling executes arbitrary code. `--allow-pickle`
   is required and deliberately not the default.

```bash
uv run python scripts/fetch_ddi100.py --raw /path/to/DDI-100 --allow-pickle --limit 40
```

## What each real source can and cannot measure

This is the part that is easy to get wrong, because an adapter that emits empty
annotations looks identical in the final report to a dataset that contributed nothing
interesting.

**SROIE** annotates text *segments* in a column confusingly named `words` — the entries
are lines ("TEL:07-388 2218 FAX:07-388 8218"), not words. They go into `lines`, and the
region metric reports `granularity="line"` for them. Treating them as words would make
SROIE's coverage look mechanically better than FUNSD's for no real reason.

**FUNSD** has genuine word boxes and annotates every word on the page, so `gt_complete`
is true and its junk rate is meaningful.

**Cheques and MIDV** carry only coarse field regions — a cheque's `name` box routinely
spans the full 2365 px width to cover a payee line with a dozen characters in it.
Scoring region coverage against a mostly-empty box produces a number that looks like a
failure and means nothing, so these adapters emit **no** lines or words at all. They are
correctly absent from the region-coverage table and contribute latency, component
routing, and PII/signature region ground truth instead. To give them real text ground
truth, use `scripts/review_gt.py` (bootstrap with an OCR engine, correct by hand,
apply).

## Ground-truth completeness

`Sample.meta["gt_complete"]` governs whether the junk rate is defined for a sample. A
surviving component that sits on real-but-unannotated text is not junk, and counting it
as junk would punish the pipeline for the dataset's gaps. Metrics return `None` rather
than a misleading number, and the report prints `n/a`.

## Licensing

Worth settling before any of this informs a shipped model, not after:

- **FUNSD** is non-commercial, research and educational use only.
- **SROIE**'s terms vary by mirror; the one used here is CC-BY-4.0.
- **IDRBT** (the real cheques behind the synthetic ones) states no licence at all, which
  is why only the Apache-2.0 synthetic derivative is wired up.
- **MIDV-2020** requires accepting its licence through the access form.

Benchmarking an internal experiment is fine under all of these. Training a production
model is a different question. The synthetic generator has none of these constraints,
which is a further argument for leaning on it.
