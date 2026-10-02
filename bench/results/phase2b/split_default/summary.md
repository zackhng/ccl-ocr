# Benchmark run `20261002T122617Z`

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
| line recall (engine-agnostic) | 96.5% |
| CER | n/a |
| end-to-end P50 / P95 / P99 ms | 27.0 / 85.7 / 92.0 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| filter | 2.5 | 22.6 | 27.8 | 89.1 | 5.3 |
| split | 3.7 | 22.6 | 24.4 | 25.0 | 5.8 |
| binarize | 5.6 | 9.1 | 10.1 | 36.6 | 6.0 |
| preprocess | 10.7 | 19.8 | 26.6 | 30.0 | 11.8 |
| rescale_boxes | 0.7 | 7.3 | 7.8 | 65.5 | 2.0 |
| ccl | 2.4 | 17.0 | 17.7 | 247.5 | 4.9 |
| **total** | 27.0 | 85.7 | 92.0 | 477.4 | 35.1 |

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
| filter | 2.3 | 6.1 | 11.9 | 13.6 | 2.6 |
| split | 2.9 | 6.7 | 11.1 | 13.1 | 3.5 |
| binarize | 5.5 | 6.1 | 6.8 | 7.3 | 5.1 |
| preprocess | 10.7 | 16.2 | 18.3 | 19.3 | 11.4 |
| ccl | 2.0 | 4.1 | 5.2 | 5.4 | 2.3 |
| rescale_boxes | 0.6 | 1.5 | 2.0 | 2.1 | 0.7 |
| **total** | 25.6 | 35.8 | 43.5 | 43.9 | 25.7 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| filter | 2.9 | 25.4 | 34.1 | 89.1 | 6.6 |
| split | 5.6 | 22.7 | 24.5 | 25.0 | 7.0 |
| binarize | 5.7 | 9.2 | 10.8 | 36.6 | 6.4 |
| preprocess | 10.8 | 21.0 | 27.9 | 30.0 | 12.0 |
| rescale_boxes | 0.8 | 7.7 | 13.7 | 65.5 | 2.9 |
| ccl | 2.7 | 17.2 | 17.9 | 247.5 | 6.3 |
| **total** | 29.8 | 88.7 | 92.3 | 477.4 | 40.0 |

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
