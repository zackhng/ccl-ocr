# Benchmark run `20261002T121743Z`

- engine: **ccl**
- samples: **245** (150 with character-level ground truth)
- repeats: 3 (+1 warm-up), percentiles taken across documents
- platform: Windows-10-10.0.26200-SP0 / Python 3.11.15

## Headline

| metric | value |
|---|---|
| character isolation recall | 66.2% |
| over-segmentation rate | 1.3% |
| merge rate | 25.9% |
| miss rate | 6.6% |
| junk rate (of surviving components) | 23.8% |
| small-mark retention | 99.3% |
| region hit rate (mixed) | 97.7% |
| region coverage (mixed) | 69.1% |
| line recall (engine-agnostic) | 96.4% |
| CER | n/a |
| end-to-end P50 / P95 / P99 ms | 23.9 / 63.4 / 67.4 |

Character metrics are computed only over samples that carry character-level ground truth — today that is the synthetic set. Word metrics cover everything. Rates are aggregated from raw counts, so a long document weighs more than a sparse one.

## Latency by stage

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| rescale_boxes | 0.6 | 6.5 | 7.0 | 64.5 | 1.8 |
| binarize | 5.6 | 8.9 | 10.4 | 41.9 | 6.1 |
| ccl | 2.4 | 16.6 | 17.7 | 280.1 | 5.0 |
| filter | 2.3 | 22.0 | 27.6 | 93.8 | 5.1 |
| preprocess | 11.0 | 20.2 | 27.0 | 32.1 | 12.0 |
| **total** | 23.9 | 63.4 | 67.4 | 489.2 | 29.3 |

## Slices

An aggregate hides the failure mode. These are the cuts that matter.

### By capture mode

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| photo | 84 | 25313 | 52.9% | 1.4% | 37.8% | 7.9% | 41.2% | 99.4% |
| scan | 66 | 32235 | 76.6% | 1.3% | 16.5% | 5.6% | 6.3% | 99.2% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| photo | 84 | word | 4836 | 99.5% | 78.6% | 1.7% | 38.4% |
| scan | 161 | mixed | 12286 | 97.0% | 65.4% | 3.6% | n/a |

### By source

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| synth | 150 | 57548 | 66.2% | 1.3% | 25.9% | 6.6% | 23.8% | 99.3% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| funsd | 25 | word | 4169 | 92.3% | 56.2% | 7.4% | 13.1% |
| sroie | 40 | line | 2159 | 98.4% | 53.6% | 4.7% | 44.3% |
| synth | 150 | word | 10794 | 99.6% | 77.2% | 1.0% | 22.1% |

### By script

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| han | 9 | 2211 | 80.3% | 8.3% | 9.8% | 1.6% | 0.1% | 100.0% |
| latin | 141 | 55337 | 65.6% | 1.0% | 26.5% | 6.8% | 24.9% | 99.3% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| han | 9 | word | 355 | 100.0% | 80.6% | 0.0% | 0.0% |
| latin | 236 | mixed | 16767 | 97.6% | 68.9% | 3.1% | n/a |

### By effective DPI

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| 100-149 | 46 | 28299 | 72.2% | 1.7% | 22.0% | 4.1% | 9.9% | 99.5% |
| 150-249 | 61 | 10661 | 62.4% | 0.7% | 22.1% | 14.8% | 49.6% | 100.0% |
| <100 | 33 | 16644 | 57.6% | 1.2% | 36.3% | 4.8% | 19.0% | 98.6% |
| >=250 | 10 | 1944 | 73.3% | 0.0% | 13.3% | 13.4% | 29.6% | 100.0% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| 100-149 | 46 | word | 5226 | 99.8% | 76.7% | 0.5% | 8.4% |
| 150-249 | 126 | mixed | 8342 | 95.5% | 60.2% | 5.7% | 40.4% |
| <100 | 33 | word | 3197 | 100.0% | 79.1% | 0.3% | 15.7% |
| >=250 | 40 | word | 357 | 97.5% | 77.5% | 3.9% | n/a |

### By template

| slice | samples | GT chars | isolated | over-seg | merged | missed | junk | small kept |
|---|---|---|---|---|---|---|---|---|
| cheque | 29 | 6318 | 62.4% | 0.4% | 29.0% | 8.1% | 5.9% | 100.0% |
| form | 17 | 5937 | 68.5% | 0.2% | 29.4% | 1.9% | 2.2% | 96.3% |
| id_card | 51 | 7993 | 59.3% | 1.0% | 21.3% | 18.4% | 62.2% | 100.0% |
| plain_han | 9 | 2211 | 80.3% | 8.3% | 9.8% | 1.6% | 0.1% | 100.0% |
| plain_latin | 16 | 27736 | 68.1% | 1.4% | 26.7% | 3.8% | 2.2% | 99.8% |
| receipt | 28 | 7353 | 63.5% | 1.0% | 27.0% | 8.4% | 34.5% | 99.1% |



| slice | samples | unit | GT regions | hit | coverage | crossing | junk |
|---|---|---|---|---|---|---|---|
| cheque | 59 | word | 1133 | 96.6% | 79.7% | 5.1% | n/a |
| form | 42 | word | 5234 | 93.8% | 62.6% | 6.0% | 9.1% |
| id_card | 51 | word | 1558 | 100.0% | 73.4% | 2.6% | 61.3% |
| plain_han | 9 | word | 355 | 100.0% | 80.6% | 0.0% | 0.0% |
| plain_latin | 16 | word | 5063 | 100.0% | 74.6% | 0.0% | 0.0% |
| receipt | 68 | mixed | 3779 | 99.1% | 64.9% | 3.0% | 42.0% |

## Latency by capture mode

### photo (84 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| binarize | 5.5 | 6.2 | 7.0 | 7.2 | 5.2 |
| ccl | 2.1 | 4.1 | 5.4 | 5.5 | 2.3 |
| filter | 2.1 | 5.3 | 10.7 | 13.9 | 2.5 |
| preprocess | 10.9 | 16.3 | 19.5 | 20.1 | 11.5 |
| rescale_boxes | 0.5 | 1.2 | 1.8 | 2.0 | 0.6 |
| **total** | 22.0 | 28.3 | 37.4 | 38.5 | 22.0 |

### scan (161 samples)

| stage | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|
| rescale_boxes | 0.8 | 6.6 | 12.7 | 64.5 | 2.7 |
| binarize | 5.8 | 9.3 | 10.5 | 41.9 | 6.5 |
| ccl | 2.6 | 17.1 | 18.1 | 280.1 | 6.5 |
| filter | 2.6 | 24.0 | 34.2 | 93.8 | 6.5 |
| preprocess | 11.1 | 21.2 | 28.3 | 32.1 | 12.2 |
| **total** | 25.3 | 64.6 | 72.0 | 489.2 | 33.1 |

## Component routing

| slice | text | diacritic | rule | blob | noise |
|---|---|---|---|---|---|
| photo | 26989 | 5012 | 372 | 91 | 25543 |
| scan | 119159 | 23120 | 6706 | 799 | 220803 |

## How to read this

- **Isolated** is the only outcome Phase 3 can consume as designed. It is the ceiling on end-to-end accuracy, before a single weight is trained.
- **Over-segmentation** points at preprocessing — thresholds too aggressive, resolution too low, strokes breaking.
- **Merging** points at resolution and at glyphs touching rules; it is the failure mode a bigger classifier cannot fix.
- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare it against the filter's routing counts above.
- **Small-mark retention** below ~95% means the size gates are eating punctuation. On a cheque that turns `1,234.56` into `123456`.
