# Benchmark run `20261001T145727Z`

- engine: **ccl**
- samples: **245** (150 with character-level ground truth)
- repeats: 5 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 65.9% |
| over-segmentation rate | 1.3% |
| merge rate | 25.9% |
| miss rate | 6.9% |
| junk rate (of surviving components) | 22.7% |
| small-mark retention | 74.5% |
| region hit rate (mixed) | 97.7% |
| region coverage (mixed) | 69.0% |
| end-to-end P50 / P95 / P99 ms | 87.3 / 265.5 / 322.3 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 10.4 | 76.6 | 91.6 | 991.8 | 21.6 |
| preprocess | 38.1 | 71.6 | 88.5 | 102.5 | 42.6 |
| binarize | 21.8 | 31.6 | 38.9 | 145.5 | 22.3 |
| rescale_boxes | 3.2 | 31.4 | 33.5 | 333.5 | 9.0 |
| filter | 8.3 | 88.6 | 110.2 | 427.4 | 20.0 |
| **total** | 87.3 | 265.5 | 322.3 | 1961.7 | 113.3 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| photo | 84 | 25313 | 52.7% | 1.3% | 37.8% | 8.2% | 39.8% | 78.9% |
| scan | 66 | 32235 | 76.3% | 1.3% | 16.5% | 5.9% | 6.0% | 69.9% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 84 | word | 4836 | 99.5% | 78.5% | 1.6% | 36.9% |
| scan | 161 | mixed | 12286 | 96.9% | 65.3% | 3.5% | n/a |

### By source

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| synth | 150 | 57548 | 65.9% | 1.3% | 25.9% | 6.9% | 22.7% | 74.5% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| funsd | 25 | word | 4169 | 92.2% | 56.1% | 7.4% | 12.8% |
| sroie | 40 | line | 2159 | 98.4% | 53.4% | 4.4% | 44.7% |
| synth | 150 | word | 10794 | 99.6% | 77.2% | 1.0% | 20.9% |

### By script

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| han | 9 | 2211 | 80.2% | 8.3% | 9.8% | 1.7% | 0.1% | 11.1% |
| latin | 141 | 55337 | 65.3% | 1.0% | 26.5% | 7.1% | 23.8% | 75.9% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| han | 9 | word | 355 | 100.0% | 80.3% | 0.0% | 0.0% |
| latin | 236 | mixed | 16767 | 97.6% | 68.8% | 3.0% | n/a |

### By effective DPI

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| 100-149 | 46 | 28299 | 71.8% | 1.7% | 22.0% | 4.5% | 9.6% | 68.4% |
| 150-249 | 61 | 10661 | 62.3% | 0.6% | 22.1% | 14.9% | 47.5% | 84.3% |
| <100 | 33 | 16644 | 57.3% | 1.2% | 36.3% | 5.2% | 18.4% | 79.9% |
| >=250 | 10 | 1944 | 73.2% | 0.0% | 13.3% | 13.5% | 28.9% | 60.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 100-149 | 46 | word | 5226 | 99.8% | 76.7% | 0.5% | 8.0% |
| 150-249 | 126 | mixed | 8342 | 95.4% | 60.1% | 5.5% | 39.9% |
| <100 | 33 | word | 3197 | 100.0% | 79.0% | 0.3% | 15.0% |
| >=250 | 40 | word | 357 | 97.5% | 77.5% | 3.9% | n/a |

### By template

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| cheque | 29 | 6318 | 62.3% | 0.3% | 29.0% | 8.3% | 5.6% | 69.5% |
| form | 17 | 5937 | 67.5% | 0.2% | 29.5% | 2.8% | 2.2% | 42.6% |
| id_card | 51 | 7993 | 59.3% | 1.0% | 21.3% | 18.5% | 60.2% | 92.5% |
| plain_han | 9 | 2211 | 80.2% | 8.3% | 9.8% | 1.7% | 0.1% | 11.1% |
| plain_latin | 16 | 27736 | 67.8% | 1.4% | 26.7% | 4.2% | 2.3% | 69.3% |
| receipt | 28 | 7353 | 63.5% | 1.0% | 27.0% | 8.5% | 33.5% | 94.7% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cheque | 59 | word | 1133 | 96.6% | 79.7% | 5.1% | n/a |
| form | 42 | word | 5234 | 93.7% | 62.5% | 5.9% | 9.0% |
| id_card | 51 | word | 1558 | 100.0% | 73.4% | 2.4% | 59.3% |
| plain_han | 9 | word | 355 | 100.0% | 80.3% | 0.0% | 0.0% |
| plain_latin | 16 | word | 5063 | 100.0% | 74.5% | 0.0% | 0.0% |
| receipt | 68 | mixed | 3779 | 99.1% | 64.8% | 2.8% | 42.0% |

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 9.3 | 17.4 | 22.9 | 24.7 | 10.1 |
| preprocess | 38.0 | 60.3 | 68.3 | 70.9 | 41.2 |
| binarize | 21.1 | 23.8 | 24.5 | 24.6 | 20.0 |
| filter | 7.5 | 23.6 | 38.0 | 49.5 | 9.1 |
| rescale_boxes | 2.6 | 6.2 | 9.0 | 9.7 | 3.0 |
| **total** | 83.9 | 104.5 | 142.1 | 146.4 | 83.3 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| ccl | 11.9 | 78.1 | 94.2 | 991.8 | 27.6 |
| preprocess | 39.2 | 74.8 | 94.1 | 102.5 | 43.4 |
| binarize | 22.0 | 35.2 | 40.7 | 145.5 | 23.6 |
| rescale_boxes | 3.9 | 31.9 | 64.8 | 333.5 | 13.3 |
| filter | 9.0 | 92.9 | 119.1 | 427.4 | 25.7 |
| **total** | 91.3 | 282.6 | 329.8 | 1961.7 | 129.0 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 25780 | 5012 | 372 | 91 | 26752 |
| scan | 116378 | 23120 | 6706 | 799 | 223584 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
