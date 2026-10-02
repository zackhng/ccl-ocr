# Benchmark run `20261002T202250Z`

- engine: **ccl-cnn-nolm**
- samples: **761** (0 with character-level ground truth)
- repeats: 1 (+0 warm-up), percentiles taken across documents
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
| region hit rate (mixed) | 98.1% |
| region coverage (mixed) | 47.0% |
| line recall (engine-agnostic) | 90.9% |
| CER | 19.2% |
| end-to-end P50 / P95 / P99 ms | 166.5 / 730.7 / 3053.3 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 8.1 | 12.4 | 16.2 | 29.5 | 8.1 |
| rescale_boxes | 1.9 | 10.2 | 24.8 | 85.7 | 3.3 |
| ccl | 4.9 | 58.9 | 115.7 | 301.7 | 12.1 |
| recognize | 104.9 | 571.4 | 2857.8 | 61365.2 | 315.4 |
| split | 8.2 | 20.8 | 35.9 | 61.1 | 9.8 |
| group | 4.2 | 16.3 | 45.9 | 94.8 | 6.6 |
| filter | 8.9 | 47.5 | 96.3 | 177.9 | 14.3 |
| preprocess | 19.5 | 29.1 | 32.6 | 52.7 | 18.7 |
| **total** | 166.5 | 730.7 | 3053.3 | 61581.3 | 387.2 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 100 | word | 2356 | 98.6% | 40.0% | 15.4% | n/a |
| scan | 661 | mixed | 102693 | 98.1% | 47.2% | 5.5% | 13.7% |

### By source

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cord | 100 | word | 2356 | 98.6% | 40.0% | 15.4% | n/a |
| funsd | 50 | word | 8707 | 96.1% | 51.9% | 6.8% | 19.4% |
| sroie | 361 | line | 19385 | 99.6% | 53.4% | 3.0% | 19.9% |
| xfund | 250 | word | 74601 | 97.9% | 45.0% | 5.9% | 9.4% |

### By script

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| latin | 761 | mixed | 105049 | 98.1% | 47.0% | 5.7% | n/a |

### By effective DPI

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 150-249 | 511 | mixed | 30448 | 98.5% | 51.9% | 5.0% | n/a |
| >=250 | 250 | word | 74601 | 97.9% | 45.0% | 5.9% | 9.4% |

### By template

_No character-level ground truth in this slice._



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| form | 300 | word | 83308 | 97.7% | 45.7% | 6.0% | 10.2% |
| receipt | 461 | mixed | 21741 | 99.5% | 51.9% | 4.3% | n/a |

## Latency by capture mode

### photo (100 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 8.5 | 16.2 | 19.1 | 23.2 | 9.0 |
| rescale_boxes | 9.8 | 34.6 | 46.9 | 54.5 | 12.2 |
| ccl | 29.0 | 122.3 | 144.7 | 181.5 | 43.9 |
| recognize | 253.3 | 3097.3 | 10741.0 | 61365.2 | 1303.2 |
| split | 11.4 | 27.1 | 38.3 | 58.5 | 13.0 |
| group | 7.3 | 40.1 | 72.1 | 94.8 | 13.1 |
| filter | 16.9 | 75.6 | 113.3 | 121.9 | 24.7 |
| preprocess | 14.4 | 18.9 | 22.2 | 27.9 | 13.6 |
| **total** | 338.3 | 3352.0 | 10915.7 | 61581.3 | 1426.4 |

### scan (661 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 7.9 | 12.2 | 13.7 | 29.5 | 8.0 |
| ccl | 4.5 | 14.9 | 55.0 | 301.7 | 7.3 |
| recognize | 98.8 | 388.2 | 651.6 | 7817.6 | 165.9 |
| split | 7.9 | 19.1 | 28.4 | 61.1 | 9.3 |
| group | 4.0 | 13.1 | 39.3 | 61.6 | 5.6 |
| filter | 8.2 | 39.1 | 77.7 | 177.9 | 12.7 |
| preprocess | 20.1 | 29.3 | 32.9 | 52.7 | 19.5 |
| rescale_boxes | 1.7 | 5.1 | 8.8 | 85.7 | 2.4 |
| **total** | 157.2 | 488.8 | 753.8 | 7939.2 | 230.0 |

## Grouping (Phase 5)

Line segments and words against the annotated ones, one-to-one at IoU >= 0.5. Line rows exclude FUNSD, whose lines are form entities. n/a precision = incomplete GT.

### By capture

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 993 | 33 |
| scan | 19385 | 64.1% | 59.9% | 60.3% | 61.9% | 83308 | 65.4% | 67.0% | 67.2% | 66.2% | 3521 | 3308 |

### By template

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| form | 0 | n/a | 0.0% | 0.0% | n/a | 83308 | 65.4% | 67.0% | 67.2% | 66.2% | 3521 | 3308 |
| receipt | 20694 | 64.1% | 58.0% | 58.5% | 60.9% | 2356 | n/a | 39.3% | 39.9% | n/a | 993 | 33 |

### By script

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 20694 | 64.1% | 58.0% | 58.5% | 60.9% | 85664 | 65.4% | 66.2% | 66.4% | 65.8% | 4514 | 3341 |

### By source

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cord | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 993 | 33 |
| funsd | 0 | n/a | 0.0% | 0.0% | n/a | 8707 | 45.1% | 54.2% | 54.4% | 49.3% | 335 | 915 |
| sroie | 19385 | 64.1% | 59.9% | 60.3% | 61.9% | 0 | n/a | 0.0% | 0.0% | n/a | 0 | 0 |
| xfund | 0 | n/a | 0.0% | 0.0% | n/a | 74601 | 68.3% | 68.5% | 68.7% | 68.4% | 3186 | 2393 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 221740 | 29692 | 776 | 2410 | 883232 |
| scan | 711031 | 79710 | 6890 | 1315 | 427116 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
