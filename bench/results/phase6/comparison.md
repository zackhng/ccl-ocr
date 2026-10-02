# Phase 6 bake-off `phase6`

- engines: **ccl**, **paddle-det**, **paddle**
- documents: 245 common to every run
- repeats: 5 (+1 warm-up) per document; percentiles across documents
- platform: Windows-10-10.0.26200-SP0 / AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD / Python 3.11.15
- `ccl` run `20261002T123421Z`, 60s wall, config: `{'preprocess': {'long_side_cap': 1600, 'upscale_small': True, 'min_long_side': 1000, 'normalize_illumination': True, 'background_kernel': 31, 'denoise': False}, 'binarize': {'method': 'adaptive_gaussian', 'block_size': 35, 'c': 10.0, 'sauvola_k': 0.2, 'sauvola_window': 25, 'recover_inverted_regions': True, 'inverted_min_area_frac': 0.002, 'inverted_min_fill': 0.55}, 'split': {'enabled': True, 'suspect_width_ratio': 1.2, 'suspect_min_height_ratio': 0.6, 'suspect_max_height_ratio': 2.5, 'max_suspects': 1500, 'small_block_ratio': 0.4, 'small_c': 10.0, 'min_part_height': 0.4, 'split_touching_columns': False, 'cut_max_ink': 0.3, 'min_piece_width': 0.8}, 'filters': {'min_pixel_area': 4, 'min_extent': 2, 'max_width_frac': 0.6, 'max_height_frac': 0.6, 'max_aspect': 12.0, 'min_aspect': 0.03, 'min_fill_ratio': 0.15, 'rule_aspect': 8.0, 'rule_fill_min': 0.4, 'use_relative_gates': True, 'min_height_ratio': 0.3, 'max_height_ratio': 4.0, 'blob_area_frac': 0.02, 'diacritic_max_height_ratio': 0.55, 'diacritic_x_overlap': 0.3, 'diacritic_y_gap_ratio': 1.2, 'punctuation_max_gap_ratio': 0.6, 'punctuation_below_baseline_ratio': 0.35, 'min_components_for_stats': 12}, 'connectivity': 8}` cv2 threads: 8
- `paddle-det` run `20261002T123523Z`, 114s wall, config: `{'mode': 'det', 'det_model': 'PP-OCRv5_mobile_det', 'rec_model': 'PP-OCRv5_mobile_rec', 'cpu_threads': 8, 'enable_mkldnn': True}` versions: `{'paddleocr': '3.4.1', 'paddlepaddle': '3.1.1'}` cv2 threads: 8
- `paddle` run `20261002T123717Z`, 1328s wall, config: `{'mode': 'full', 'det_model': 'PP-OCRv5_mobile_det', 'rec_model': 'PP-OCRv5_mobile_rec', 'cpu_threads': 8, 'enable_mkldnn': True}` versions: `{'paddleocr': '3.4.1', 'paddlepaddle': '3.1.1'}` cv2 threads: 8

**Latency comparability warnings** — read the latency sections with these in mind:

- `ccl` note: idle machine; ccl = phase 2b default
- `paddle-det` note: idle machine; ccl = phase 2b default
- `paddle` note: idle machine; ccl = phase 2b default

## Verdict

**GO** — `ccl` meets every criterion.

| slice | criterion | value | limit | result | detail |
|---|---|---|---|---|---|
| overall | latency P50 ratio (ccl / paddle) | 0.04 | <= 0.50 | pass | ccl 27.8 ms vs paddle 718.7 ms over 245 docs |
| overall | latency P95 ratio (ccl / paddle) | 0.05 | <= 0.50 | pass | ccl 86.7 ms vs paddle 1878.7 ms over 245 docs |
| overall | line recall gap (paddle-det - ccl) | 0.01 | <= 0.10 | pass | ccl 96.5% vs paddle-det 97.6% over 6605 GT lines |
| photo | latency P50 ratio (ccl / paddle) | 0.06 | <= 0.50 | pass | ccl 25.9 ms vs paddle 400.1 ms over 84 docs |
| photo | latency P95 ratio (ccl / paddle) | 0.03 | <= 0.50 | pass | ccl 36.2 ms vs paddle 1431.1 ms over 84 docs |
| photo | line recall gap (paddle-det - ccl) | 0.02 | <= 0.10 | pass | ccl 96.6% vs paddle-det 98.6% over 1755 GT lines |

Criteria fixed before the run: `ccl` P50 and P95 each <= 50% of `paddle`'s (paired per document), and line recall within 10 points of `paddle-det` — overall and on photographs.

## Latency

End-to-end ms per document (median of repeats), percentiles across documents.

### By capture mode

| engine | slice | docs | P50 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| ccl | all | 245 | 27.8 | 86.7 | 95.3 | 495.7 |
| ccl | photo | 84 | 25.9 | 36.2 | 43.6 | 44.2 |
| ccl | scan | 161 | 31.7 | 91.9 | 96.5 | 495.7 |
| paddle-det | all | 245 | 70.3 | 107.8 | 110.0 | 119.1 |
| paddle-det | photo | 84 | 48.5 | 93.4 | 94.0 | 94.1 |
| paddle-det | scan | 161 | 76.6 | 108.5 | 111.9 | 119.1 |
| paddle | all | 245 | 718.7 | 1878.7 | 2047.9 | 2185.8 |
| paddle | photo | 84 | 400.1 | 1431.1 | 1812.5 | 1917.1 |
| paddle | scan | 161 | 923.5 | 1935.0 | 2054.1 | 2185.8 |

### By source

| engine | slice | docs | P50 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| ccl | all | 245 | 27.8 | 86.7 | 95.3 | 495.7 |
| ccl | cheque | 30 | 67.8 | 95.3 | 97.1 | 97.7 |
| ccl | funsd | 25 | 24.8 | 29.7 | 30.9 | 31.2 |
| ccl | sroie | 40 | 39.4 | 73.8 | 334.3 | 495.7 |
| ccl | synth | 150 | 25.4 | 43.4 | 49.8 | 51.3 |
| paddle-det | all | 245 | 70.3 | 107.8 | 110.0 | 119.1 |
| paddle-det | cheque | 30 | 70.4 | 74.8 | 76.7 | 77.3 |
| paddle-det | funsd | 25 | 107.8 | 113.8 | 118.0 | 119.1 |
| paddle-det | sroie | 40 | 79.2 | 96.2 | 98.6 | 99.9 |
| paddle-det | synth | 150 | 64.7 | 94.7 | 97.5 | 102.2 |
| paddle | all | 245 | 718.7 | 1878.7 | 2047.9 | 2185.8 |
| paddle | cheque | 30 | 1011.7 | 1970.5 | 2028.9 | 2052.0 |
| paddle | funsd | 25 | 1264.0 | 1927.1 | 2032.7 | 2057.3 |
| paddle | sroie | 40 | 1014.7 | 1713.3 | 1908.2 | 1925.8 |
| paddle | synth | 150 | 512.3 | 1741.2 | 1981.2 | 2185.8 |

### Paired: `ccl` against `paddle` on the same document

| slice | docs | ratio P50 | ratio P95 | budget P50 ms | budget P5 ms |
|---|---|---|---|---|---|
| all | 245 | 0.04 | 0.09 | 690.8 | 286.9 |
| scan | 161 | 0.04 | 0.08 | 875.8 | 327.7 |
| photo | 84 | 0.06 | 0.09 | 381.0 | 265.4 |

**Budget** is `paddle`'s time minus `ccl`'s on the same document: what remains for the stages `ccl` has not built yet before it stops being faster. P5 is the tight end — the documents with least room.

`paddle` recognition cost (full minus detection-only, paired over 245 docs): P50 625.0 ms, P95 1790.1 ms — 91% of its time at the median.

## Localisation

Per ground-truth line: the share of its horizontal span covered by the engine's text boxes (gaps under one line-height closed, so character boxes and line boxes score alike). **Found** = span coverage >= 50%. **Spurious** = predicted boxes on no GT line; n/a where GT is incomplete. Cheques carry no line GT and are absent.

### By capture mode

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl | all | 6605 | 96.5% | 93.1% | 25.6% |
| ccl | photo | 1755 | 96.6% | 95.2% | 31.9% |
| ccl | scan | 4850 | 96.5% | 92.3% | 23.0% |
| paddle-det | all | 6605 | 97.6% | 97.2% | 1.9% |
| paddle-det | photo | 1755 | 98.6% | 98.5% | 1.9% |
| paddle-det | scan | 4850 | 97.2% | 96.7% | 1.8% |
| paddle | all | 6605 | 98.2% | 97.7% | 1.8% |
| paddle | photo | 1755 | 98.6% | 98.5% | 1.9% |
| paddle | scan | 4850 | 98.0% | 97.4% | 1.8% |

### By source

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl | all | 6605 | 96.5% | 93.1% | 25.6% |
| ccl | funsd | 1045 | 96.6% | 90.4% | 7.8% |
| ccl | sroie | 2159 | 95.4% | 89.6% | 43.5% |
| ccl | synth | 3401 | 97.2% | 96.0% | 19.3% |
| paddle-det | all | 6605 | 97.6% | 97.2% | 1.9% |
| paddle-det | funsd | 1045 | 98.8% | 97.8% | 3.6% |
| paddle-det | sroie | 2159 | 95.0% | 94.3% | 1.7% |
| paddle-det | synth | 3401 | 98.9% | 98.9% | 1.4% |
| paddle | all | 6605 | 98.2% | 97.7% | 1.8% |
| paddle | funsd | 1045 | 98.7% | 97.6% | 3.4% |
| paddle | sroie | 2159 | 96.6% | 95.9% | 1.7% |
| paddle | synth | 3401 | 99.0% | 98.9% | 1.4% |

### By script

| engine | slice | GT lines | found | coverage | spurious |
|---|---|---|---|---|---|
| ccl | all | 6605 | 96.5% | 93.1% | 25.6% |
| ccl | han | 269 | 100.0% | 98.8% | 0.0% |
| ccl | latin | 6336 | 96.4% | 92.8% | 26.2% |
| paddle-det | all | 6605 | 97.6% | 97.2% | 1.9% |
| paddle-det | han | 269 | 100.0% | 99.9% | 0.0% |
| paddle-det | latin | 6336 | 97.5% | 97.1% | 1.9% |
| paddle | all | 6605 | 98.2% | 97.7% | 1.8% |
| paddle | han | 269 | 100.0% | 99.9% | 0.0% |
| paddle | latin | 6336 | 98.1% | 97.6% | 1.9% |

## Recognition accuracy (target for Phase 3+)

### By capture mode

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| paddle | all | 98469 | 5.7% | 78.1% |
| paddle | photo | 25322 | 5.0% | 79.1% |
| paddle | scan | 73147 | 6.0% | 77.8% |

### By source

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| paddle | all | 98469 | 5.7% | 78.1% |
| paddle | funsd | 20657 | 9.8% | 65.4% |
| paddle | sroie | 20206 | 10.4% | 61.7% |
| paddle | synth | 57606 | 2.6% | 87.5% |

### By script

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| paddle | all | 98469 | 5.7% | 78.1% |
| paddle | han | 2211 | 0.7% | 99.5% |
| paddle | latin | 96258 | 5.9% | 75.4% |

### By template

| engine | slice | GT chars | CER | word F1 |
|---|---|---|---|---|
| paddle | all | 98469 | 5.7% | 78.1% |
| paddle | cheque | 6318 | 6.4% | 70.6% |
| paddle | form | 26594 | 7.6% | 70.5% |
| paddle | id_card | 8002 | 2.5% | 71.2% |
| paddle | plain_han | 2211 | 0.7% | 99.5% |
| paddle | plain_latin | 27785 | 0.4% | 94.1% |
| paddle | receipt | 27559 | 10.5% | 64.9% |

CER = (edits on GT lines + characters predicted on no GT line) / GT characters, with whitespace removed, and case folded for SROIE, whose transcripts are all upper case. Word F1 is bag-of-words and space-sensitive, so it is where dropped spaces ("ROCNO:538358-H") are charged; Han is tokenised by character. SROIE transcripts also omit some printed text on annotated lines, which inflates CER for every engine alike.

## Slowest documents

The tail is where a latency claim breaks. Per-stage ms for each engine's 5 slowest documents.

### ccl

| sample | capture | total ms | stages (ms) |
|---|---|---|---|
| sroie_X51005361908 | scan | 496 | ccl 249, filter 89, rescale_boxes 68, binarize 28, split 24, preprocess 22 |
| cheque_icici_syn_0000 | scan | 98 | filter 25, split 25, ccl 17, preprocess 10, binarize 8, rescale_boxes 7 |
| cheque_icici_syn_0021 | scan | 96 | filter 26, split 26, ccl 17, preprocess 10, binarize 8, rescale_boxes 7 |
| cheque_icici_syn_0041 | scan | 95 | filter 26, split 25, ccl 17, preprocess 10, binarize 8, rescale_boxes 7 |
| cheque_icici_syn_0074 | scan | 95 | filter 26, split 24, ccl 16, preprocess 10, binarize 8, rescale_boxes 7 |

### paddle-det

| sample | capture | total ms | stages (ms) |
|---|---|---|---|
| funsd_83641919_1921 | scan | 119 | paddle_det 119 |
| funsd_82253362_3364 | scan | 115 | paddle_det 115 |
| funsd_82253245_3247 | scan | 110 | paddle_det 110 |
| funsd_83624198 | scan | 110 | paddle_det 110 |
| funsd_83573282 | scan | 110 | paddle_det 110 |

### paddle

| sample | capture | total ms | stages (ms) |
|---|---|---|---|
| synth_plain_latin_0015 | scan | 2186 | paddle_ocr 2186 |
| funsd_82253362_3364 | scan | 2057 | paddle_ocr 2057 |
| cheque_canara_syn_0096 | scan | 2052 | paddle_ocr 2052 |
| synth_plain_latin_0001 | scan | 2043 | paddle_ocr 2043 |
| cheque_canara_syn_0026 | scan | 1972 | paddle_ocr 1972 |

## Caveats

- CCL's latency covers preprocess → binarise → CCL → filter only. It has no classifier and no reconstruction yet; full Paddle includes recognition. The budget columns are the honest reading, not the raw ratio.
- Paddle runs PP-OCRv5 *mobile* on CPU with oneDNN, orientation and unwarping off, and its own resize defaults (no long-side cap; CCL caps at 1600 px).
- Character-level and word-level isolation metrics are CCL-only and live in each engine's own `summary.md`.
