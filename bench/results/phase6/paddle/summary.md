# Benchmark run `20261002T123717Z`

- engine: **paddle**
- samples: **245** (0 with character-level ground truth)
- repeats: 5 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 0.0% |
| over-segmentation rate | 0.0% |
| merge rate | 0.0% |
| miss rate | 0.0% |
| junk rate (of surviving components) | n/a |
| small-mark retention | n/a |
| region hit rate (none) | 0.0% |
| region coverage (none) | 0.0% |
| line recall (engine-agnostic) | 98.2% |
| CER | 5.7% |
| end-to-end P50 / P95 / P99 ms | 718.7 / 1878.7 / 2047.9 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| paddle_ocr | 718.7 | 1878.7 | 2047.9 | 2185.8 | 861.6 |
| **total** | 718.7 | 1878.7 | 2047.9 | 2185.8 | 861.6 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|

### By source

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|

### By script

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|

### By effective DPI

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|

### By template

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| paddle_ocr | 400.1 | 1431.1 | 1812.5 | 1917.1 | 514.4 |
| **total** | 400.1 | 1431.1 | 1812.5 | 1917.1 | 514.4 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| paddle_ocr | 923.5 | 1935.0 | 2054.1 | 2185.8 | 1042.8 |
| **total** | 923.5 | 1935.0 | 2054.1 | 2185.8 | 1042.8 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 0 | 0 | 0 | 0 | 0 |
| scan | 0 | 0 | 0 | 0 | 0 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
