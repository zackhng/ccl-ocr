# Phase 6 bake-off `phase3_real_v3`

- engines: **ccl-cnn-nolm**, **ccl-cnn**
- documents: 761 common to every run
- repeats: 1 (+0 warm-up) per document; percentiles across documents
- platform: Windows-10-10.0.26200-SP0 / AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD / Python 3.11.15
- `ccl-cnn-nolm` run `20261002T202250Z`, 355s wall, config: `{'preprocess': {'long_side_cap': 1600, 'upscale_small': True, 'min_long_side': 1000, 'normalize_illumination': True, 'background_kernel': 31, 'denoise': False}, 'binarize': {'method': 'adaptive_gaussian', 'block_size': 35, 'c': 10.0, 'sauvola_k': 0.2, 'sauvola_window': 25, 'recover_inverted_regions': True, 'inverted_min_area_frac': 0.002, 'inverted_min_fill': 0.55}, 'split': {'enabled': True, 'suspect_width_ratio': 1.2, 'suspect_min_height_ratio': 0.6, 'suspect_max_height_ratio': 2.5, 'max_suspects': 1500, 'small_block_ratio': 0.4, 'small_c': 10.0, 'min_part_height': 0.4, 'split_touching_columns': False, 'cut_max_ink': 0.3, 'min_piece_width': 0.8}, 'filters': {'min_pixel_area': 4, 'min_extent': 2, 'max_width_frac': 0.6, 'max_height_frac': 0.6, 'max_aspect': 12.0, 'min_aspect': 0.03, 'min_fill_ratio': 0.15, 'rule_aspect': 8.0, 'rule_fill_min': 0.4, 'use_relative_gates': True, 'min_height_ratio': 0.3, 'max_height_ratio': 4.0, 'blob_area_frac': 0.02, 'diacritic_max_height_ratio': 0.55, 'diacritic_x_overlap': 0.3, 'diacritic_y_gap_ratio': 1.2, 'punctuation_max_gap_ratio': 0.6, 'punctuation_below_baseline_ratio': 0.35, 'min_components_for_stats': 12}, 'group': {'enabled': True, 'line_gap_ratio': 2.5, 'word_gap_ratio': 0.25, 'word_gap_median_factor': 2.0, 'core_fraction': 0.6, 'core_cap_ratio': 1.0, 'junk_max_height_ratio': 0.7, 'junk_small_max_count': 2, 'junk_single_height_ratio': 0.9, 'junk_median_height_ratio': 0.6, 'junk_barcode_aspect': 0.15}, 'connectivity': 8}` cv2 threads: 8
- `ccl-cnn` run `20261002T202845Z`, 1924s wall, config: `{'preprocess': {'long_side_cap': 1600, 'upscale_small': True, 'min_long_side': 1000, 'normalize_illumination': True, 'background_kernel': 31, 'denoise': False}, 'binarize': {'method': 'adaptive_gaussian', 'block_size': 35, 'c': 10.0, 'sauvola_k': 0.2, 'sauvola_window': 25, 'recover_inverted_regions': True, 'inverted_min_area_frac': 0.002, 'inverted_min_fill': 0.55}, 'split': {'enabled': True, 'suspect_width_ratio': 1.2, 'suspect_min_height_ratio': 0.6, 'suspect_max_height_ratio': 2.5, 'max_suspects': 1500, 'small_block_ratio': 0.4, 'small_c': 10.0, 'min_part_height': 0.4, 'split_touching_columns': False, 'cut_max_ink': 0.3, 'min_piece_width': 0.8}, 'filters': {'min_pixel_area': 4, 'min_extent': 2, 'max_width_frac': 0.6, 'max_height_frac': 0.6, 'max_aspect': 12.0, 'min_aspect': 0.03, 'min_fill_ratio': 0.15, 'rule_aspect': 8.0, 'rule_fill_min': 0.4, 'use_relative_gates': True, 'min_height_ratio': 0.3, 'max_height_ratio': 4.0, 'blob_area_frac': 0.02, 'diacritic_max_height_ratio': 0.55, 'diacritic_x_overlap': 0.3, 'diacritic_y_gap_ratio': 1.2, 'punctuation_max_gap_ratio': 0.6, 'punctuation_below_baseline_ratio': 0.35, 'min_components_for_stats': 12}, 'group': {'enabled': True, 'line_gap_ratio': 2.5, 'word_gap_ratio': 0.25, 'word_gap_median_factor': 2.0, 'core_fraction': 0.6, 'core_cap_ratio': 1.0, 'junk_max_height_ratio': 0.7, 'junk_small_max_count': 2, 'junk_single_height_ratio': 0.9, 'junk_median_height_ratio': 0.6, 'junk_barcode_aspect': 0.15}, 'connectivity': 8}` cv2 threads: 8

**Latency comparability warnings** — read the latency sections with these in mind:

- `ccl-cnn-nolm` note: accuracy run, sequential: CNN v3 (real incl. XFUND) + LM v3 + case pass; latency not measured
- `ccl-cnn` note: accuracy run, sequential: CNN v3 (real incl. XFUND) + LM v3 + case pass; latency not measured

## Verdict

_Not computable: needs both `ccl-cnn` and `paddle` runs._

## Latency

End-to-end ms per document (median of repeats), percentiles across documents.

### By capture mode

| engine | slice | docs | P50 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| ccl-cnn-nolm | all | 761 | 166.5 | 730.7 | 3053.3 | 61581.3 |
| ccl-cnn-nolm | photo | 100 | 338.3 | 3352.0 | 10915.7 | 61581.3 |
| ccl-cnn-nolm | scan | 661 | 157.2 | 488.8 | 753.8 | 7939.2 |
| ccl-cnn | all | 761 | 1414.4 | 6515.4 | 12446.7 | 62475.4 |
| ccl-cnn | photo | 100 | 2294.1 | 14937.8 | 23050.7 | 62475.4 |
| ccl-cnn | scan | 661 | 1350.7 | 5672.6 | 8945.2 | 24357.9 |

### By source

| engine | slice | docs | P50 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| ccl-cnn-nolm | all | 761 | 166.5 | 730.7 | 3053.3 | 61581.3 |
| ccl-cnn-nolm | cord | 100 | 338.3 | 3352.0 | 10915.7 | 61581.3 |
| ccl-cnn-nolm | funsd | 50 | 119.0 | 355.4 | 709.0 | 801.4 |
| ccl-cnn-nolm | sroie | 361 | 132.4 | 210.4 | 307.9 | 7279.3 |
| ccl-cnn-nolm | xfund | 250 | 276.2 | 640.6 | 1015.3 | 7939.2 |
| ccl-cnn | all | 761 | 1414.4 | 6515.4 | 12446.7 | 62475.4 |
| ccl-cnn | cord | 100 | 2294.1 | 14937.8 | 23050.7 | 62475.4 |
| ccl-cnn | funsd | 50 | 1137.7 | 3445.1 | 4432.3 | 5119.6 |
| ccl-cnn | sroie | 361 | 1047.2 | 1885.5 | 3913.5 | 24357.9 |
| ccl-cnn | xfund | 250 | 3283.9 | 7515.4 | 10119.1 | 13510.8 |

## Localisation

Per ground-truth line: the share of its horizontal span covered by the engine's text boxes (gaps under one line-height closed, so character boxes and line boxes score alike). **Found** = span coverage >= 50%. **Spurious** = predicted boxes on no GT line; n/a where GT is incomplete. Cheques carry no line GT and are absent.

### By capture mode

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl-cnn-nolm | all | 23026 | 90.9% | 86.2% | n/a |
| ccl-cnn-nolm | photo | 1309 | 67.8% | 65.7% | n/a |
| ccl-cnn-nolm | scan | 21717 | 92.3% | 87.4% | 17.9% |
| ccl-cnn | all | 23026 | 90.9% | 86.1% | n/a |
| ccl-cnn | photo | 1309 | 67.8% | 65.6% | n/a |
| ccl-cnn | scan | 21717 | 92.3% | 87.4% | 17.9% |

### By source

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl-cnn-nolm | all | 23026 | 90.9% | 86.2% | n/a |
| ccl-cnn-nolm | cord | 1309 | 67.8% | 65.7% | n/a |
| ccl-cnn-nolm | funsd | 2332 | 93.7% | 88.8% | 48.2% |
| ccl-cnn-nolm | sroie | 19385 | 92.1% | 87.2% | 9.1% |
| ccl-cnn | all | 23026 | 90.9% | 86.1% | n/a |
| ccl-cnn | cord | 1309 | 67.8% | 65.6% | n/a |
| ccl-cnn | funsd | 2332 | 93.5% | 88.7% | 48.3% |
| ccl-cnn | sroie | 19385 | 92.1% | 87.2% | 9.0% |

### By script

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl-cnn-nolm | all | 23026 | 90.9% | 86.2% | n/a |
| ccl-cnn-nolm | latin | 23026 | 90.9% | 86.2% | n/a |
| ccl-cnn | all | 23026 | 90.9% | 86.1% | n/a |
| ccl-cnn | latin | 23026 | 90.9% | 86.1% | n/a |

## Recognition accuracy (target for Phase 3+)

### By capture mode

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| ccl-cnn-nolm | all | 661113 | 19.2% | 52.7% |
| ccl-cnn-nolm | photo | 11324 | 50.8% | 26.9% |
| ccl-cnn-nolm | scan | 649789 | 18.7% | 53.3% |
| ccl-cnn | all | 661113 | 18.9% | 56.2% |
| ccl-cnn | photo | 11324 | 48.2% | 30.0% |
| ccl-cnn | scan | 649789 | 18.4% | 56.8% |

### By source

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| ccl-cnn-nolm | all | 661113 | 19.2% | 52.7% |
| ccl-cnn-nolm | cord | 11324 | 50.8% | 26.9% |
| ccl-cnn-nolm | funsd | 44064 | 44.3% | 25.5% |
| ccl-cnn-nolm | sroie | 185486 | 25.2% | 42.4% |
| ccl-cnn-nolm | xfund | 420239 | 13.1% | 62.5% |
| ccl-cnn | all | 661113 | 18.9% | 56.2% |
| ccl-cnn | cord | 11324 | 48.2% | 30.0% |
| ccl-cnn | funsd | 44064 | 47.8% | 27.7% |
| ccl-cnn | sroie | 185486 | 23.1% | 47.8% |
| ccl-cnn | xfund | 420239 | 13.2% | 65.0% |

### By script

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| ccl-cnn-nolm | all | 661113 | 19.2% | 52.7% |
| ccl-cnn-nolm | latin | 661113 | 19.2% | 52.7% |
| ccl-cnn | all | 661113 | 18.9% | 56.2% |
| ccl-cnn | latin | 661113 | 18.9% | 56.2% |

### By template

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| ccl-cnn-nolm | all | 661113 | 19.2% | 52.7% |
| ccl-cnn-nolm | form | 464303 | 16.1% | 59.0% |
| ccl-cnn-nolm | receipt | 196810 | 26.7% | 41.4% |
| ccl-cnn | all | 661113 | 18.9% | 56.2% |
| ccl-cnn | form | 464303 | 16.5% | 61.5% |
| ccl-cnn | receipt | 196810 | 24.5% | 46.7% |

CER = (edits on GT lines + characters predicted on no GT line) / GT characters, with whitespace removed, and case folded for SROIE, whose transcripts are all upper case. Word F1 is bag-of-words and space-sensitive, so it is where dropped spaces ("ROCNO:538358-H") are charged; Han is tokenised by character. SROIE transcripts also omit some printed text on annotated lines, which inflates CER for every engine alike.

## Slowest documents

The tail is where a latency claim breaks. Per-stage ms for each engine's 5 slowest documents.

### ccl-cnn-nolm

| sample | capture | total ms | stages (ms) |
|---|---|---|---|
| cord_test_74 | photo | 61581 | recognize 61365, filter 122, ccl 30, split 19, preprocess 17, binarize 11, rescale_boxes 11, group 6 |
| cord_test_76 | photo | 10404 | recognize 10230, ccl 84, filter 30, split 20, preprocess 17, binarize 12, group 11 |
| xfund_pt_val_29 | scan | 7939 | recognize 7818, split 27, ccl 25, preprocess 21, filter 16, binarize 13, group 12, rescale_boxes 8 |
| sroie_X51005361908 | scan | 7279 | recognize 6667, ccl 302, filter 92, rescale_boxes 86, group 62, binarize 29, split 23, preprocess 19 |
| cord_test_42 | photo | 6917 | recognize 6729, filter 59, split 36, ccl 36, preprocess 16, group 16, rescale_boxes 15, binarize 10 |

### ccl-cnn

| sample | capture | total ms | stages (ms) |
|---|---|---|---|
| cord_test_74 | photo | 62475 | recognize 62261, filter 131, ccl 23, split 19, preprocess 16, rescale_boxes 11, binarize 9, group 6 |
| sroie_X51005361908 | scan | 24358 | recognize 23744, ccl 301, filter 92, rescale_boxes 90, group 64, binarize 26, split 22, preprocess 18 |
| cord_test_38 | photo | 22653 | recognize 22245, ccl 191, rescale_boxes 59, group 56, filter 47, split 23, binarize 16, preprocess 16 |
| cord_test_59 | photo | 20302 | recognize 19811, ccl 125, filter 115, group 106, split 42, rescale_boxes 39, preprocess 37, binarize 27 |
| cord_test_70 | photo | 15828 | recognize 15534, filter 107, ccl 91, split 35, group 27, preprocess 19, binarize 15 |

## Caveats

- CCL's latency covers preprocess → binarise → CCL → filter only. It has no classifier and no reconstruction yet; full Paddle includes recognition. The budget columns are the honest reading, not the raw ratio.
- Paddle runs PP-OCRv5 *mobile* on CPU with oneDNN, orientation and unwarping off, and its own resize defaults (no long-side cap; CCL caps at 1600 px).
- Character-level and word-level isolation metrics are CCL-only and live in each engine's own `summary.md`.
