# Benchmark run `20261001T142503Z`

- engine: **ccl**
- samples: **245** (150 with character-level ground truth)
- repeats: 5 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 60.9% |
| over-segmentation rate | 1.3% |
| merge rate | 23.7% |
| miss rate | 14.1% |
| junk rate (of surviving components) | 36.2% |
| small-mark retention | 75.8% |
| region hit rate (mixed) | 94.1% |
| region coverage (mixed) | 64.9% |
| end-to-end P50 / P95 / P99 ms | 83.1 / 218.3 / 230.9 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 10.2 | 75.5 | 79.5 | 1111.4 | 21.8 |
| preprocess | 38.0 | 66.6 | 85.6 | 98.8 | 42.0 |
| binarize | 21.7 | 32.3 | 37.0 | 198.4 | 22.7 |
| filter | 5.1 | 39.0 | 51.4 | 401.8 | 11.4 |
| rescale_boxes | 3.2 | 30.9 | 32.9 | 384.4 | 9.3 |
| **total** | 83.1 | 218.3 | 230.9 | 2114.2 | 104.6 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| photo | 84 | 25313 | 43.1% | 1.2% | 33.2% | 22.4% | 58.1% | 79.6% |
| scan | 66 | 32235 | 74.9% | 1.3% | 16.1% | 7.6% | 10.7% | 71.8% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 84 | word | 4836 | 88.9% | 66.8% | 1.4% | 55.9% |
| scan | 161 | mixed | 12286 | 96.2% | 64.1% | 3.6% | n/a |

### By source

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| synth | 150 | 57548 | 60.9% | 1.3% | 23.7% | 14.1% | 36.2% | 75.8% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| funsd | 25 | word | 4169 | 92.2% | 56.1% | 7.4% | 13.1% |
| sroie | 40 | line | 2159 | 98.5% | 51.1% | 4.8% | 50.1% |
| synth | 150 | word | 10794 | 94.0% | 71.0% | 0.9% | 34.7% |

### By script

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| han | 9 | 2211 | 80.2% | 8.3% | 9.8% | 1.7% | 0.1% | 11.1% |
| latin | 141 | 55337 | 60.2% | 1.0% | 24.2% | 14.6% | 37.7% | 77.2% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| han | 9 | word | 355 | 100.0% | 80.3% | 0.0% | 0.0% |
| latin | 236 | mixed | 16767 | 94.0% | 64.5% | 3.0% | n/a |

### By effective DPI

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| 100-149 | 46 | 28299 | 71.6% | 1.7% | 21.2% | 5.5% | 12.5% | 71.5% |
| 150-249 | 61 | 10661 | 38.6% | 0.4% | 13.8% | 47.2% | 74.1% | 78.9% |
| <100 | 33 | 16644 | 56.5% | 1.2% | 35.7% | 6.6% | 23.0% | 82.6% |
| >=250 | 10 | 1944 | 67.3% | 0.0% | 10.6% | 22.1% | 42.4% | 43.3% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 100-149 | 46 | word | 5226 | 99.3% | 76.1% | 0.4% | 11.0% |
| 150-249 | 126 | mixed | 8342 | 89.4% | 52.8% | 5.6% | 51.9% |
| <100 | 33 | word | 3197 | 98.6% | 77.5% | 0.3% | 19.8% |
| >=250 | 40 | word | 357 | 89.4% | 69.0% | 3.9% | n/a |

### By template

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| cheque | 29 | 6318 | 62.3% | 0.3% | 29.0% | 8.3% | 5.7% | 70.5% |
| form | 17 | 5937 | 67.5% | 0.2% | 29.5% | 2.8% | 2.2% | 42.6% |
| id_card | 51 | 7993 | 25.1% | 0.7% | 6.8% | 67.5% | 85.1% | 78.4% |
| plain_han | 9 | 2211 | 80.2% | 8.3% | 9.8% | 1.7% | 0.1% | 11.1% |
| plain_latin | 16 | 27736 | 67.9% | 1.4% | 26.7% | 4.1% | 2.3% | 74.7% |
| receipt | 28 | 7353 | 61.5% | 1.1% | 25.5% | 11.9% | 39.9% | 96.5% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cheque | 59 | word | 1133 | 96.6% | 79.7% | 5.1% | n/a |
| form | 42 | word | 5234 | 93.7% | 62.5% | 5.9% | 9.2% |
| id_card | 51 | word | 1558 | 64.1% | 33.9% | 1.9% | 84.6% |
| plain_han | 9 | word | 355 | 100.0% | 80.3% | 0.0% | 0.0% |
| plain_latin | 16 | word | 5063 | 100.0% | 74.5% | 0.0% | 0.0% |
| receipt | 68 | mixed | 3779 | 97.9% | 62.1% | 3.0% | 47.7% |

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 8.9 | 18.3 | 22.2 | 23.9 | 10.0 |
| preprocess | 37.6 | 57.8 | 66.3 | 69.3 | 40.3 |
| binarize | 20.9 | 23.9 | 24.7 | 26.0 | 19.9 |
| filter | 3.9 | 11.4 | 17.6 | 19.4 | 4.4 |
| rescale_boxes | 2.5 | 6.1 | 8.8 | 9.2 | 3.0 |
| **total** | 79.7 | 101.6 | 108.6 | 111.9 | 77.3 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 11.7 | 77.7 | 80.3 | 1111.4 | 27.9 |
| preprocess | 38.4 | 74.2 | 90.4 | 98.8 | 42.8 |
| binarize | 22.0 | 34.9 | 38.2 | 198.4 | 24.2 |
| filter | 7.6 | 42.4 | 55.6 | 401.8 | 15.1 |
| rescale_boxes | 3.8 | 31.7 | 68.6 | 384.4 | 13.8 |
| **total** | 87.4 | 223.0 | 234.0 | 2114.2 | 118.8 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 36850 | 981 | 372 | 3956 | 15848 |
| scan | 141296 | 6478 | 6706 | 6157 | 209950 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
