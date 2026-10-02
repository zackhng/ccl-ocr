# Benchmark run `20261002T151652Z`

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
| line recall (engine-agnostic) | 90.7% |
| CER | 38.7% |
| end-to-end P50 / P95 / P99 ms | 216.4 / 902.5 / 1918.8 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| rescale_boxes | 3.2 | 18.7 | 47.2 | 164.2 | 5.9 |
| group | 7.0 | 27.3 | 85.8 | 174.6 | 11.1 |
| recognize | 130.9 | 708.6 | 1479.4 | 11349.5 | 247.4 |
| preprocess | 22.1 | 34.5 | 39.4 | 45.4 | 21.6 |
| binarize | 11.1 | 17.2 | 24.7 | 49.8 | 11.3 |
| ccl | 7.4 | 96.5 | 170.6 | 499.5 | 20.1 |
| filter | 14.5 | 78.5 | 191.7 | 332.1 | 23.4 |
| split | 11.6 | 28.4 | 42.3 | 116.7 | 13.5 |
| **total** | 216.4 | 902.5 | 1918.8 | 12128.5 | 352.5 |

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
| rescale_boxes | 14.8 | 59.3 | 97.6 | 125.0 | 22.7 |
| group | 13.7 | 62.7 | 93.8 | 174.6 | 20.8 |
| recognize | 354.1 | 2066.4 | 7576.1 | 11349.5 | 755.5 |
| preprocess | 16.2 | 22.9 | 28.0 | 33.5 | 15.9 |
| binarize | 11.5 | 24.6 | 38.7 | 49.8 | 12.8 |
| ccl | 44.3 | 185.5 | 315.0 | 333.0 | 73.3 |
| filter | 24.8 | 151.4 | 239.9 | 241.6 | 42.7 |
| split | 14.8 | 37.4 | 48.8 | 63.0 | 16.8 |
| **total** | 516.9 | 2278.7 | 8306.3 | 12128.5 | 948.6 |

### scan (661 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| group | 6.5 | 21.5 | 81.4 | 117.3 | 9.6 |
| recognize | 122.2 | 452.2 | 697.9 | 3061.5 | 170.5 |
| preprocess | 22.6 | 34.8 | 39.8 | 45.4 | 22.5 |
| binarize | 11.1 | 16.4 | 20.2 | 48.7 | 11.1 |
| ccl | 6.9 | 27.0 | 93.0 | 499.5 | 12.0 |
| filter | 13.0 | 69.2 | 125.6 | 332.1 | 20.5 |
| split | 11.2 | 26.1 | 40.0 | 116.7 | 13.0 |
| rescale_boxes | 3.0 | 10.1 | 16.1 | 164.2 | 4.3 |
| **total** | 203.7 | 619.7 | 840.5 | 4133.8 | 262.3 |

## Grouping (Phase 5)

Line segments and words against the annotated ones, one-to-one at IoU >= 0.5. Line rows exclude FUNSD, whose lines are form entities. n/a precision = incomplete GT.

### By capture

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.2% | 39.9% | n/a | 993 | 33 |
| scan | 19385 | 63.0% | 59.7% | 60.3% | 61.3% | 83308 | 65.3% | 67.0% | 67.2% | 66.1% | 3557 | 3304 |

### By template

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| form | 0 | n/a | 0.0% | 0.0% | n/a | 83308 | 65.3% | 67.0% | 67.2% | 66.1% | 3557 | 3304 |
| receipt | 20694 | 63.0% | 57.8% | 58.5% | 60.3% | 2356 | n/a | 39.2% | 39.9% | n/a | 993 | 33 |

### By script

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 20694 | 63.0% | 57.8% | 58.5% | 60.3% | 85664 | 65.3% | 66.2% | 66.4% | 65.8% | 4550 | 3337 |

### By source

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cord | 1309 | n/a | 30.6% | 31.2% | n/a | 2356 | n/a | 39.2% | 39.9% | n/a | 993 | 33 |
| funsd | 0 | n/a | 0.0% | 0.0% | n/a | 8707 | 45.2% | 54.2% | 54.4% | 49.3% | 337 | 913 |
| sroie | 19385 | 63.0% | 59.7% | 60.3% | 61.3% | 0 | n/a | 0.0% | 0.0% | n/a | 0 | 0 |
| xfund | 0 | n/a | 0.0% | 0.0% | n/a | 74601 | 68.1% | 68.5% | 68.7% | 68.3% | 3220 | 2391 |

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
