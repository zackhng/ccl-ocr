# Benchmark run `20261002T123523Z`

- engine: **paddle-det**
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
| line recall (engine-agnostic) | 97.6% |
| CER | n/a |
| end-to-end P50 / P95 / P99 ms | 70.3 / 107.8 / 110.0 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| paddle_det | 70.3 | 107.8 | 110.0 | 119.1 | 70.5 |
| **total** | 70.3 | 107.8 | 110.0 | 119.1 | 70.5 |

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
| paddle_det | 48.5 | 93.4 | 94.0 | 94.1 | 53.3 |
| **total** | 48.5 | 93.4 | 94.0 | 94.1 | 53.3 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| paddle_det | 76.6 | 108.5 | 111.9 | 119.1 | 79.5 |
| **total** | 76.6 | 108.5 | 111.9 | 119.1 | 79.5 |

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
