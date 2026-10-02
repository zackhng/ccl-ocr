# Benchmark run `20261002T202845Z`

- engine: **ccl-cnn**
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
| CER | 18.9% |
| end-to-end P50 / P95 / P99 ms | 1414.4 / 6515.4 / 12446.7 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 8.2 | 12.6 | 16.0 | 26.9 | 8.0 |
| rescale_boxes | 1.7 | 9.5 | 33.4 | 90.4 | 3.2 |
| ccl | 5.0 | 53.9 | 107.2 | 301.4 | 11.6 |
| recognize | 1338.3 | 6339.7 | 12263.5 | 62261.2 | 2386.8 |
| split | 8.2 | 19.4 | 29.0 | 82.5 | 9.3 |
| group | 4.1 | 17.0 | 54.1 | 115.7 | 6.7 |
| filter | 8.5 | 46.0 | 97.4 | 181.4 | 13.1 |
| preprocess | 19.2 | 33.0 | 49.5 | 58.7 | 19.5 |
| **total** | 1414.4 | 6515.4 | 12446.7 | 62475.4 | 2457.2 |

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
| binarize | 8.0 | 15.3 | 20.0 | 26.9 | 8.6 |
| rescale_boxes | 8.2 | 35.5 | 49.6 | 58.8 | 12.3 |
| ccl | 22.5 | 114.2 | 180.9 | 191.2 | 42.0 |
| recognize | 2190.8 | 14719.3 | 22644.7 | 62261.2 | 4278.6 |
| split | 10.3 | 26.5 | 42.6 | 82.5 | 12.8 |
| group | 6.4 | 32.2 | 106.4 | 115.7 | 12.8 |
| filter | 14.1 | 67.4 | 115.0 | 131.1 | 22.6 |
| preprocess | 13.8 | 25.9 | 28.5 | 36.7 | 14.2 |
| **total** | 2294.1 | 14937.8 | 23050.7 | 62475.4 | 4397.4 |

### scan (661 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 8.3 | 12.5 | 14.4 | 25.7 | 7.9 |
| ccl | 4.7 | 14.5 | 50.8 | 301.4 | 7.1 |
| recognize | 1288.9 | 5575.7 | 8839.7 | 23744.4 | 2100.6 |
| split | 7.9 | 18.2 | 23.2 | 73.2 | 8.7 |
| group | 3.8 | 12.3 | 46.7 | 89.0 | 5.7 |
| filter | 7.9 | 38.6 | 74.7 | 181.4 | 11.7 |
| preprocess | 19.5 | 33.3 | 49.9 | 58.7 | 20.3 |
| rescale_boxes | 1.6 | 4.9 | 8.9 | 90.4 | 2.3 |
| **total** | 1350.7 | 5672.6 | 8945.2 | 24357.9 | 2163.7 |

## Grouping (Phase 5)

Line segments and words against the annotated ones, one-to-one at IoU >= 0.5. Line rows exclude FUNSD, whose lines are form entities. n/a precision = incomplete GT.

### By capture

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 986 | 33 |
| scan | 19385 | 64.3% | 59.9% | 60.3% | 62.0% | 83308 | 65.5% | 67.0% | 67.2% | 66.2% | 3516 | 3307 |

### By template

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| form | 0 | n/a | 0.0% | 0.0% | n/a | 83308 | 65.5% | 67.0% | 67.2% | 66.2% | 3516 | 3307 |
| receipt | 20694 | 64.3% | 58.0% | 58.5% | 61.0% | 2356 | n/a | 39.3% | 39.9% | n/a | 986 | 33 |

### By script

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 20694 | 64.3% | 58.0% | 58.5% | 61.0% | 85664 | 65.5% | 66.2% | 66.4% | 65.8% | 4502 | 3340 |

### By source

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cord | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.3% | 39.9% | n/a | 986 | 33 |
| funsd | 0 | n/a | 0.0% | 0.0% | n/a | 8707 | 45.2% | 54.2% | 54.4% | 49.3% | 334 | 915 |
| sroie | 19385 | 64.3% | 59.9% | 60.3% | 62.0% | 0 | n/a | 0.0% | 0.0% | n/a | 0 | 0 |
| xfund | 0 | n/a | 0.0% | 0.0% | n/a | 74601 | 68.3% | 68.5% | 68.7% | 68.4% | 3182 | 2392 |

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
