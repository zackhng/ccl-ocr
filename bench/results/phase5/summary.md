# Benchmark run `20261002T141246Z`

- engine: **ccl**
- samples: **245** (150 with character-level ground truth)
- repeats: 3 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 82.0% |
| over-segmentation rate | 2.8% |
| merge rate | 7.4% |
| miss rate | 7.8% |
| junk rate (of surviving components) | 20.1% |
| small-mark retention | 99.1% |
| region hit rate (mixed) | 98.4% |
| region coverage (mixed) | 66.4% |
| line recall (engine-agnostic) | 91.4% |
| CER | n/a |
| end-to-end P50 / P95 / P99 ms | 29.2 / 97.1 / 100.1 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| rescale_boxes | 0.9 | 8.1 | 9.0 | 75.8 | 2.3 |
| group | 2.0 | 6.8 | 8.4 | 56.2 | 3.0 |
| preprocess | 10.7 | 19.9 | 26.8 | 30.5 | 11.8 |
| ccl | 2.5 | 17.7 | 19.0 | 297.9 | 5.3 |
| split | 3.8 | 22.3 | 23.8 | 24.8 | 5.8 |
| binarize | 5.6 | 8.9 | 10.2 | 38.0 | 6.0 |
| filter | 2.6 | 23.5 | 28.9 | 93.4 | 5.4 |
| **total** | 29.2 | 97.1 | 100.1 | 598.1 | 38.9 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| photo | 84 | 25313 | 73.0% | 4.2% | 12.6% | 10.2% | 33.4% | 99.1% |
| scan | 66 | 32235 | 89.1% | 1.8% | 3.3% | 5.8% | 5.4% | 99.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 84 | word | 4836 | 99.8% | 73.2% | 1.8% | 32.6% |
| scan | 161 | mixed | 12286 | 97.8% | 63.7% | 3.4% | n/a |

### By source

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| synth | 150 | 57548 | 82.0% | 2.8% | 7.4% | 7.8% | 20.1% | 99.1% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| funsd | 25 | word | 4169 | 94.6% | 54.5% | 6.8% | 11.2% |
| sroie | 40 | line | 2159 | 98.4% | 52.8% | 4.7% | 43.5% |
| synth | 150 | word | 10794 | 99.8% | 73.7% | 1.1% | 19.7% |

### By script

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| han | 9 | 2211 | 85.0% | 8.9% | 4.6% | 1.5% | 0.0% | 100.0% |
| latin | 141 | 55337 | 81.9% | 2.6% | 7.5% | 8.0% | 20.9% | 99.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| han | 9 | word | 355 | 100.0% | 80.1% | 0.0% | 0.0% |
| latin | 236 | mixed | 16767 | 98.4% | 66.1% | 3.0% | n/a |

### By effective DPI

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| 100-149 | 46 | 28299 | 87.2% | 2.7% | 4.7% | 5.3% | 8.2% | 99.5% |
| 150-249 | 61 | 10661 | 74.1% | 1.9% | 8.1% | 16.0% | 46.0% | 100.0% |
| <100 | 33 | 16644 | 78.3% | 4.0% | 11.6% | 6.1% | 13.6% | 98.0% |
| >=250 | 10 | 1944 | 81.4% | 0.1% | 6.0% | 12.5% | 27.9% | 100.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 100-149 | 46 | word | 5226 | 99.9% | 73.4% | 0.5% | 7.9% |
| 150-249 | 126 | mixed | 8342 | 96.8% | 58.7% | 5.4% | 37.6% |
| <100 | 33 | word | 3197 | 100.0% | 73.7% | 0.3% | 13.0% |
| >=250 | 40 | word | 357 | 99.7% | 78.7% | 3.9% | n/a |

### By template

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| cheque | 29 | 6318 | 77.9% | 2.0% | 10.4% | 9.7% | 3.4% | 100.0% |
| form | 17 | 5937 | 89.9% | 0.5% | 7.2% | 2.4% | 0.3% | 94.1% |
| id_card | 51 | 7993 | 68.3% | 2.9% | 7.8% | 21.0% | 59.1% | 100.0% |
| plain_han | 9 | 2211 | 85.0% | 8.9% | 4.6% | 1.5% | 0.0% | 100.0% |
| plain_latin | 16 | 27736 | 87.0% | 3.0% | 5.4% | 4.6% | 0.0% | 99.8% |
| receipt | 28 | 7353 | 74.5% | 2.8% | 12.9% | 9.8% | 30.4% | 99.1% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cheque | 59 | word | 1133 | 98.8% | 77.5% | 5.3% | n/a |
| form | 42 | word | 5234 | 95.7% | 60.6% | 5.4% | 8.1% |
| id_card | 51 | word | 1558 | 100.0% | 70.8% | 2.8% | 58.3% |
| plain_han | 9 | word | 355 | 100.0% | 80.1% | 0.0% | 0.0% |
| plain_latin | 16 | word | 5063 | 100.0% | 69.8% | 0.0% | 0.0% |
| receipt | 68 | mixed | 3779 | 99.1% | 63.2% | 3.0% | 40.6% |

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| group | 1.4 | 4.5 | 5.9 | 6.1 | 1.8 |
| preprocess | 10.7 | 16.2 | 18.3 | 19.2 | 11.4 |
| ccl | 2.0 | 4.2 | 5.5 | 5.6 | 2.3 |
| split | 3.0 | 6.7 | 11.3 | 13.0 | 3.5 |
| binarize | 5.4 | 6.2 | 6.7 | 7.3 | 5.1 |
| filter | 2.3 | 6.1 | 11.8 | 14.0 | 2.7 |
| rescale_boxes | 0.7 | 1.8 | 2.3 | 2.5 | 0.8 |
| **total** | 27.4 | 41.1 | 47.3 | 48.7 | 27.7 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| rescale_boxes | 1.0 | 8.6 | 16.0 | 75.8 | 3.4 |
| group | 2.7 | 7.4 | 8.8 | 56.2 | 3.7 |
| preprocess | 11.0 | 21.2 | 28.1 | 30.5 | 12.1 |
| ccl | 2.7 | 18.3 | 19.3 | 297.9 | 6.8 |
| split | 5.5 | 22.9 | 24.1 | 24.8 | 7.0 |
| binarize | 5.8 | 9.1 | 10.4 | 38.0 | 6.4 |
| filter | 2.9 | 26.3 | 35.7 | 93.4 | 6.8 |
| **total** | 33.7 | 98.0 | 101.8 | 598.1 | 44.8 |

## Grouping (Phase 5)

Line segments and words against the annotated ones, one-to-one at IoU >= 0.5. Line rows exclude FUNSD, whose lines are form entities. n/a precision = incomplete GT.

### By capture

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photo | 1755 | 74.0% | 81.0% | 83.1% | 77.3% | 4836 | 71.6% | 81.5% | 82.4% | 76.2% | 382 | 135 |
| scan | 3805 | 59.9% | 75.9% | 76.4% | 66.9% | 10127 | 79.0% | 74.6% | 74.6% | 76.7% | 394 | 569 |

### By template

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheque | 377 | 89.0% | 75.1% | 77.2% | 81.4% | 1133 | 68.2% | 77.4% | 77.8% | 72.5% | 144 | 80 |
| form | 612 | 98.7% | 99.5% | 99.5% | 99.1% | 5234 | 69.9% | 57.5% | 57.6% | 63.1% | 206 | 534 |
| id_card | 816 | 56.4% | 76.1% | 76.3% | 64.8% | 1558 | 49.8% | 67.0% | 67.3% | 57.1% | 151 | 62 |
| plain_han | 269 | 63.9% | 63.9% | 63.9% | 63.9% | 355 | 66.9% | 67.6% | 67.6% | 67.2% | 2 | 4 |
| plain_latin | 479 | 100.0% | 100.0% | 100.0% | 100.0% | 5063 | 92.3% | 97.7% | 97.7% | 95.0% | 194 | 6 |
| receipt | 3007 | 54.2% | 71.4% | 72.9% | 61.6% | 1620 | 82.9% | 84.6% | 86.9% | 83.7% | 79 | 18 |

### By script

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| han | 269 | 63.9% | 63.9% | 63.9% | 63.9% | 355 | 66.9% | 67.6% | 67.6% | 67.2% | 2 | 4 |
| latin | 5291 | 63.9% | 78.2% | 79.2% | 70.3% | 14608 | 76.5% | 77.0% | 77.4% | 76.8% | 774 | 700 |

### By source

| slice | GT lines | line P | line R | R ceiling | line F1 | GT words | word P | word R | R ceiling | word F1 | word splits | word merges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| funsd | 0 | n/a | 0.0% | 0.0% | n/a | 4169 | 64.2% | 48.6% | 48.7% | 55.3% | 86 | 505 |
| sroie | 2159 | 44.5% | 64.8% | 65.6% | 52.7% | 0 | n/a | 0.0% | 0.0% | n/a | 0 | 0 |
| synth | 3401 | 80.8% | 85.6% | 86.7% | 83.2% | 10794 | 79.5% | 87.7% | 88.1% | 83.4% | 690 | 199 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 33986 | 5102 | 371 | 115 | 25856 |
| scan | 136928 | 22635 | 7547 | 1108 | 225926 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
