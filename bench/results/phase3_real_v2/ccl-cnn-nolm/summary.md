# Benchmark run `20261002T172801Z`

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
| line recall (engine-agnostic) | 91.0% |
| CER | 38.5% |
| end-to-end P50 / P95 / P99 ms | 160.4 / 800.6 / 3553.8 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 5.0 | 59.7 | 114.3 | 299.8 | 12.6 |
| preprocess | 19.5 | 30.6 | 34.8 | 58.7 | 18.8 |
| group | 4.3 | 18.0 | 53.4 | 77.1 | 6.9 |
| recognize | 97.4 | 659.9 | 3310.1 | 62043.3 | 317.5 |
| split | 8.0 | 20.0 | 27.3 | 54.8 | 9.5 |
| binarize | 7.9 | 12.1 | 16.2 | 32.1 | 8.0 |
| rescale_boxes | 1.9 | 10.1 | 23.9 | 86.7 | 3.3 |
| filter | 9.3 | 49.3 | 100.5 | 192.4 | 14.5 |
| **total** | 160.4 | 800.6 | 3553.8 | 62264.7 | 390.1 |

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
| ccl | 29.9 | 118.2 | 186.6 | 199.6 | 47.0 |
| preprocess | 15.1 | 20.2 | 25.4 | 30.2 | 13.9 |
| group | 6.6 | 32.7 | 56.6 | 76.5 | 11.7 |
| recognize | 237.1 | 3749.9 | 10671.2 | 62043.3 | 1366.3 |
| split | 11.3 | 28.1 | 35.8 | 38.6 | 12.7 |
| binarize | 8.2 | 15.7 | 21.6 | 23.6 | 9.0 |
| rescale_boxes | 9.0 | 34.1 | 47.2 | 54.7 | 12.0 |
| filter | 15.2 | 75.6 | 116.4 | 124.6 | 24.9 |
| **total** | 344.1 | 3926.8 | 10847.1 | 62264.7 | 1491.3 |

### scan (661 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 4.5 | 14.3 | 55.2 | 299.8 | 7.4 |
| preprocess | 20.1 | 31.0 | 34.8 | 58.7 | 19.5 |
| group | 4.0 | 13.5 | 50.6 | 77.1 | 6.2 |
| recognize | 92.6 | 386.1 | 847.5 | 7963.3 | 158.8 |
| split | 7.8 | 18.2 | 24.8 | 54.8 | 9.0 |
| binarize | 7.7 | 11.7 | 14.2 | 32.1 | 7.9 |
| filter | 8.4 | 43.7 | 79.6 | 192.4 | 12.9 |
| rescale_boxes | 1.8 | 5.3 | 9.4 | 86.7 | 2.4 |
| **total** | 151.9 | 483.1 | 973.3 | 8081.1 | 223.5 |

## Grouping (Phase 5)

Line segments and words against the annotated ones, one-to-one at IoU >= 0.5. Line rows exclude FUNSD, whose lines are form entities. n/a precision = incomplete GT.

### By capture

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 1009 | 33 |
| scan | 19385 | 64.1% | 59.9% | 60.3% | 62.0% | 83308 | 65.3% | 67.1% | 67.2% | 66.2% | 3534 | 3328 |

### By template

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| form | 0 | n/a | 0.0% | 0.0% | n/a | 83308 | 65.3% | 67.1% | 67.2% | 66.2% | 3534 | 3328 |
| receipt | 20694 | 64.1% | 58.1% | 58.5% | 61.0% | 2356 | n/a | 39.3% | 39.9% | n/a | 1009 | 33 |

### By script

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 20694 | 64.1% | 58.1% | 58.5% | 61.0% | 85664 | 65.3% | 66.3% | 66.4% | 65.8% | 4543 | 3361 |

### By source

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cord | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 1009 | 33 |
| funsd | 0 | n/a | 0.0% | 0.0% | n/a | 8707 | 45.2% | 54.3% | 54.4% | 49.3% | 334 | 916 |
| sroie | 19385 | 64.1% | 59.9% | 60.3% | 62.0% | 0 | n/a | 0.0% | 0.0% | n/a | 0 | 0 |
| xfund | 0 | n/a | 0.0% | 0.0% | n/a | 74601 | 68.1% | 68.5% | 68.7% | 68.3% | 3200 | 2412 |

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
