# Benchmark run `20261002T122418Z`

- engine: **ccl**
- samples: **245** (150 with character-level ground truth)
- repeats: 3 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 84.3% |
| over-segmentation rate | 3.5% |
| merge rate | 4.2% |
| miss rate | 8.0% |
| junk rate (of surviving components) | 19.2% |
| small-mark retention | 99.1% |
| region hit rate (mixed) | 99.4% |
| region coverage (mixed) | 66.8% |
| line recall (engine-agnostic) | 96.7% |
| CER | n/a |
| end-to-end P50 / P95 / P99 ms | 29.2 / 114.9 / 130.3 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| filter | 2.9 | 18.9 | 30.3 | 91.1 | 5.7 |
| preprocess | 10.7 | 19.8 | 27.0 | 30.4 | 11.9 |
| binarize | 5.6 | 9.0 | 9.6 | 37.9 | 6.0 |
| rescale_boxes | 0.7 | 7.8 | 8.3 | 75.3 | 2.1 |
| ccl | 2.5 | 17.4 | 19.1 | 257.9 | 5.1 |
| split | 5.5 | 43.8 | 64.0 | 66.6 | 9.6 |
| **total** | 29.2 | 114.9 | 130.3 | 532.7 | 39.7 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| photo | 84 | 25313 | 76.3% | 5.0% | 8.2% | 10.5% | 31.9% | 99.1% |
| scan | 66 | 32235 | 90.6% | 2.3% | 1.2% | 5.9% | 5.1% | 99.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 84 | word | 4836 | 99.9% | 72.8% | 1.8% | 31.3% |
| scan | 161 | mixed | 12286 | 99.2% | 64.5% | 3.8% | n/a |

### By source

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| synth | 150 | 57548 | 84.3% | 3.5% | 4.2% | 8.0% | 19.2% | 99.1% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| funsd | 25 | word | 4169 | 98.5% | 57.3% | 8.0% | 18.0% |
| sroie | 40 | line | 2159 | 98.4% | 52.5% | 4.7% | 43.2% |
| synth | 150 | word | 10794 | 99.9% | 73.3% | 1.1% | 18.9% |

### By script

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| han | 9 | 2211 | 89.0% | 9.3% | 0.1% | 1.5% | 0.0% | 100.0% |
| latin | 141 | 55337 | 84.1% | 3.2% | 4.4% | 8.2% | 20.0% | 99.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| han | 9 | word | 355 | 100.0% | 79.7% | 0.0% | 0.0% |
| latin | 236 | mixed | 16767 | 99.4% | 66.5% | 3.3% | n/a |

### By effective DPI

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| 100-149 | 46 | 28299 | 88.6% | 3.1% | 2.9% | 5.4% | 7.9% | 99.5% |
| 150-249 | 61 | 10661 | 76.5% | 3.1% | 4.2% | 16.3% | 44.1% | 100.0% |
| <100 | 33 | 16644 | 82.1% | 4.6% | 6.8% | 6.5% | 13.0% | 98.0% |
| >=250 | 10 | 1944 | 84.1% | 1.6% | 1.5% | 12.7% | 24.8% | 100.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 100-149 | 46 | word | 5226 | 99.9% | 73.2% | 0.5% | 7.7% |
| 150-249 | 126 | mixed | 8342 | 98.8% | 59.9% | 5.9% | 37.1% |
| <100 | 33 | word | 3197 | 100.0% | 73.1% | 0.4% | 12.7% |
| >=250 | 40 | word | 357 | 99.7% | 78.2% | 4.5% | n/a |

### By template

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| cheque | 29 | 6318 | 81.5% | 2.9% | 5.7% | 9.9% | 3.9% | 100.0% |
| form | 17 | 5937 | 93.2% | 1.7% | 2.5% | 2.6% | 0.2% | 94.1% |
| id_card | 51 | 7993 | 70.2% | 4.2% | 4.0% | 21.5% | 57.1% | 100.0% |
| plain_han | 9 | 2211 | 89.0% | 9.3% | 0.1% | 1.5% | 0.0% | 100.0% |
| plain_latin | 16 | 27736 | 87.5% | 3.3% | 4.5% | 4.7% | 0.0% | 99.8% |
| receipt | 28 | 7353 | 81.4% | 3.5% | 5.0% | 10.1% | 28.9% | 99.1% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cheque | 59 | word | 1133 | 99.2% | 77.6% | 5.3% | n/a |
| form | 42 | word | 5234 | 98.8% | 62.8% | 6.4% | 13.9% |
| id_card | 51 | word | 1558 | 100.0% | 70.3% | 2.7% | 56.4% |
| plain_han | 9 | word | 355 | 100.0% | 79.7% | 0.0% | 0.0% |
| plain_latin | 16 | word | 5063 | 100.0% | 69.7% | 0.0% | 0.0% |
| receipt | 68 | mixed | 3779 | 99.1% | 62.6% | 3.0% | 40.1% |

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| filter | 2.5 | 6.1 | 11.8 | 13.9 | 2.8 |
| preprocess | 10.7 | 16.3 | 18.4 | 19.2 | 11.5 |
| binarize | 5.5 | 6.6 | 6.8 | 7.1 | 5.1 |
| ccl | 2.0 | 4.0 | 5.6 | 5.8 | 2.3 |
| split | 4.3 | 9.2 | 14.6 | 17.7 | 4.9 |
| rescale_boxes | 0.6 | 1.6 | 2.0 | 2.1 | 0.7 |
| **total** | 27.0 | 38.8 | 45.2 | 47.1 | 27.3 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| filter | 3.5 | 28.0 | 35.7 | 91.1 | 7.3 |
| preprocess | 10.8 | 21.2 | 28.3 | 30.4 | 12.1 |
| binarize | 5.8 | 9.2 | 10.2 | 37.9 | 6.4 |
| rescale_boxes | 0.8 | 7.9 | 15.2 | 75.3 | 3.1 |
| ccl | 2.8 | 18.1 | 20.2 | 257.9 | 6.5 |
| split | 7.4 | 54.6 | 64.2 | 66.6 | 12.0 |
| **total** | 32.4 | 117.9 | 135.5 | 532.7 | 46.2 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 35398 | 4415 | 374 | 118 | 27187 |
| scan | 151977 | 22829 | 7155 | 1188 | 229385 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
