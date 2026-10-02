# Design notes — Phases 0–2

Decisions that are easy to get wrong and expensive to reverse, and the reasoning behind
them. Everything here is about the classical-CV half of the pipeline; no network has
been trained.

## 1. Filtering routes; it does not delete

Only `ComponentKind.NOISE` is discarded. Everything else is labelled and kept:

| kind | why it survives |
|---|---|
| `TEXT` | character candidate — Phase 3's input |
| `DIACRITIC` | the dot of an `i`, a comma, a Thai vowel sign. Phase 5 merges it back into its parent glyph. **Dropping it is an accuracy loss no later stage can undo.** |
| `RULE` | table borders, underlines, the line a cheque payee is written on. Phase 5 needs them for layout. |
| `BLOB` | portrait photos, logos, signatures, barcodes — and oversized text. Phase 11 needs them for masking; oversized text can be re-segmented at a different scale. |

A heading that is four times the median glyph height becomes `BLOB`, not `NOISE`,
precisely because it *is* text we are declining to classify as a single character.

Every routed component records which gate fired (`Component.reason`). That is what makes
the overlay diagnostic rather than decorative, and what lets thresholds be set from
evidence instead of intuition.

## 2. Polarity is handled twice, at two scales

**Globally**, before anything else: Otsu splits the page, and if the majority of pixels
fall on the dark side then dark is the *background*, so the image is inverted. Documents
are overwhelmingly mostly background, which is what makes this cheap test reliable.

The comparison is `<=`, not `<`. OpenCV's Otsu threshold is inclusive and lands *on* the
dark peak for a strongly bimodal image; with `<`, a page that is 96% dark measures as 0%
dark and the check inverts its answer exactly when the signal is clearest. This was a
real bug, caught by a test, and it is the kind that produces plausible output forever.

**Locally**, after binarisation: ID cards and cheques carry dark banners with light text
(an NRIC header strip, a reversed table header). Global normalisation keeps the page
right-way-up, which means those regions threshold into one solid rectangle and every
character inside is gone before filtering even looks at them. `recover_inverted_regions`
finds large solid components, re-thresholds them with the opposite polarity, and keeps
the result if it looks like text.

The subtlety that makes this work: the patch is masked to the blob's **hole-filled**
extent, not its ink. In a dark banner the ink *is* the background around the letters and
the letters are holes, so masking to ink erases exactly the characters recovery just
found. Also a real bug, also caught by a test.

## 3. Mixed coordinate systems, deliberately and narrowly

`Component.bbox` and `.centroid` are mapped back to original-image coordinates before
the engine returns. `Component.pixel_area` is **not** — an ink count has no meaningful
interpretation after resampling.

`fill_ratio` is therefore computed once at labelling time, where both quantities are
exact, and stored. Deriving it afterwards divides two differently-rounded values from
two different coordinate systems and can exceed 1.0, which is geometrically impossible
and was happening.

## 4. Thresholds set from the benchmark, not guessed

`min_fill_ratio` is the worked example. Components sitting on a ground-truth character
have a 1st-percentile fill ratio of **0.34** over the synthetic set; a 2 px frame around
a 190×50 form field sits near **0.10**. The gate is set to 0.15 — clear of the frames,
well below the glyphs.

The measurement is censored: it only sees components that already survived the current
gate, so re-measure after any large change. The command is in the git history of this
file's sibling; the shape is "for each surviving component, is it on a GT character, and
what is its fill ratio".

`max_height_ratio` was left at 4.0 deliberately. On-text components reach 3.0 at p99 and
junk reaches 3.5 — the height gate is simply not the discriminator here, and tightening
it would cost real characters for very little junk.

## 5. What the isolation metrics do and do not prove

They measure the **ceiling Phase 3 inherits**. A character CCL split into three
fragments, merged into its neighbour, or filtered away cannot be recovered by any
classifier downstream, however good. They are not OCR accuracy — nothing is recognised
yet.

Four outcomes per ground-truth character, in priority order: `isolated` (the only one
Phase 3 can consume as designed), `merged`, `over_segmented`, `missed`. Plus `junk` over
components rather than characters, reported separately because it costs latency — one
wasted inference each — and not accuracy.

Three things that keep the numbers honest:

- **Rates aggregate from counts, not averaged per-sample rates.** A per-sample mean
  weights a 12-character ID card the same as a 900-character form, which inflates the
  headline whenever the sparse pages are the easy ones.
- **Percentiles are taken across documents, not across repeats.** Each sample is timed
  several times and its *median* becomes that document's cost. The product question is
  "how slow is the 95th-percentile document", not "how slow was the 95th percentile of a
  timing loop on one easy page".
- **`junk_rate` is `None` where ground truth is incomplete.** A component on real but
  unannotated text is not junk.

**The headline is close to useless on its own.** Photographed ID cards and clean scanned
forms fail in different ways at different rates, and the fix differs accordingly — which
is why the report slices by capture mode, source, script, DPI and template, and why
30 overlays are a better first look than any table.

## 6. Ground truth is a thing that can be wrong

Three ground-truth bugs were found and fixed while building this, all of which would
have silently depressed the measured numbers forever:

- A receipt's `"-" * 42` separator was annotated as 42 characters that CCL correctly
  merges into one rule. Decorative text now renders (the pipeline must cope with it) but
  is recorded as a non-text region, not as characters.
- The ID-card portrait's shoulders were drawn past the photo's bounding box, so the
  pixels and the region annotation disagreed.
- The receipt canvas was computed too short, clipping trailing lines off the image while
  the ground truth still claimed their characters.

Hence `tests/test_synth.py::test_every_char_box_contains_ink` and
`test_degraded_boxes_still_contain_ink`: every box claiming a character must have darker
pixels inside it than the surrounding page, before *and* after degradation. A benchmark
grading the pipeline against fiction would grade it *consistently*, so nothing
downstream would ever look wrong.

## 7. Shaped scripts are skipped, not faked

Devanagari, Arabic and Thai need HarfBuzz (Pillow's Raqm backend). Without it Pillow
maps codepoints straight to glyphs: no contextual forms, no reordering, no bidi. Arabic
comes out in isolated forms and left-to-right.

The generator detects this and **drops those strata** rather than producing samples
labelled `script="arabic"` that contain mis-shaped glyphs. A per-script slice built on
those would be fiction. `python -m ocrbench.cli fonts` reports what this machine can
actually render.

For the same reason, character-level ground truth is emitted only for Latin and Han. In
a shaped script "the box of character *i*" is not a well-defined object, and one shaped
line poisons the whole sample's character truth — a partially-populated `chars` list
reads as "these are all the characters", and every unlisted glyph scores as junk.

## 8. The `Engine` protocol exists before its second implementation

Phase 6 compares this pipeline against PaddleOCR end to end, and that comparison is only
honest if both sides are driven through one interface by one runner — otherwise the
measured difference includes whatever each harness happens to do around the call. The
protocol costs nothing now and avoids a rewrite then.

## 9. Phase 6: scoring engines that disagree about what a box is

CCL answers with one box per glyph; PaddleOCR answers with one padded box per line.
Every metric that compares them has to be indifferent to that, or it measures
granularity instead of quality. Three choices follow, each from a bug that produced
plausible numbers first.

**Localisation is scored on a GT line's horizontal span**, not its area, and not by
matching boxes (`region_metrics`). A predicted box contributes to a line if its
vertical centre is inside the line and it overlaps at least half the shorter height.
Gaps under one line-height are closed before measuring. Without the gap closing,
forty tight character boxes could never reach the coverage one padded line box gets
for free. Without the centre test, Paddle's padding (a 12 px line gets a 26–29 px
box) lets a box localise the line above it too.
`test_line_box_and_character_boxes_score_alike` pins the invariant.

**Text is scored per connected cluster**, not per matched pair (`text_metrics`). GT
lines and predictions are linked where they overlap by half the *smaller* box. Each
connected group is compared as one string, GT in reading order against predictions in
reading order. One-to-one matching charges a correct read twice (one deletion, one
insertion) whenever the two sides split text differently. They often do:

- Paddle splits one line into two boxes.
- FUNSD annotates "TO:" and its value as two entities on one printed line.
- A FUNSD entity spans three printed lines.

The first version linked on the prediction's area alone. Paddle's padded boxes then
fell below 50% on their own line, and a perfectly-read form scored **CER 185% with
word F1 92%**. Two metrics disagreeing that badly is the signal to look.

**CER compares characters, not layout.** Whitespace is removed and NFKC applied on
both sides. Case is folded only for sources whose transcripts are case-normalised
(SROIE is all upper case). Dropped spaces are a real Paddle defect ("ROCNO:538358-H"),
and they are charged in the space-sensitive word F1. Folding them into CER would make
them indistinguishable from misreads. NFKC is needed because the multilingual
dictionary emits a full-width colon (U+FF1A) for a printed ASCII one.

**Paddle's version is pinned for latency fairness.** paddlepaddle 3.3.x crashes in
the PP-OCRv5 detector with oneDNN enabled on Windows CPU. Running it without oneDNN is
~3–5x slower, which would hand CCL a win it did not earn. 3.1.1 runs with oneDNN; 3.0.0
is blocked by Application Control on the reference machine. See `pyproject.toml`.

## 10. Phase 2b: splitting merged glyphs

Merging was the dominant isolation failure: 37.8% of glyphs on photographs reached the
classifier fused with a neighbour, against 8.2% lost. Three fixes looked plausible and
two were wrong, which the measurements showed before any code was written.

- **Not resolution.** Photos are already *upscaled* before thresholding (×1.35 for the
  photo profile, ×2.4 for hard_photo). Glyphs reach CCL at ~16 px, the same as scans.
  A resolution floor would add pixels without adding information; that was lost at
  capture.
- **Not rotation.** The merged neighbours' ground-truth boxes have a median gap of
  **0 px**. They touch because of downscale and blur, and rotating a line does not
  close horizontal gaps within it. Deskew belongs to Phase 5, where lines are built.
- **The threshold window.** `adaptive_gaussian` uses a 35 px window on ~16 px glyphs.
  That window averages across neighbours, so the one-pixel grey seam between two blurred
  glyphs falls below the threshold and the glyphs join.

**Why not just shrink the window.** It opens the seams but breaks strokes *inside*
glyphs. Isolation on photos plateaus at 68–70% for every window from 9 to 19 px. Merges
fall, but misses rise to match, and the extra misses are almost all normal-height glyphs
broken into fragments.

**What was built** (`src/ocr/split.py`, between CCL and filtering):

1. Measure the page's glyph height and width with the filter's own ink-weighted
   estimator, so the splitter and the filter agree on what one glyph is.
2. Select **suspects**, components at least 1.2 glyph widths wide and 0.6–2.5 glyph
   heights tall, with array operations on CCL's `stats` (a noisy photo carries tens of
   thousands of components).
3. **Re-threshold** each suspect inside its own mask, with a window of 0.4× the glyph
   height. Accept the result only if it yields ≥2 parts of at least 0.4× the suspect's
   height. A re-threshold that merely fragments strokes is rejected, so this step
   cannot make a glyph worse than it was.
4. Optionally **cut at low-ink columns** whatever still touches. This is *off by
   default*: it assumes glyphs are separate shapes that meet at thin points, which is
   false for Arabic (joined within a word) and Devanagari (a continuous headline). It
   should be enabled per region once Phase 7 tags regions by script.

The thresholds were tuned on a **held-out** synthetic set (seed 11) with
`scripts/tune_split.py`, which refuses to run on the published benchmark. Every
re-threshold setting in the grid landed within about two points of the best.

**What splitting exposed in the filter.** Small-mark retention appeared to drop when
splitting went in. In fact the drop exposed an existing bug: a full stop, comma,
decimal point or hyphen sits *beside* its neighbour, never above or below it, so
`_find_parent` cannot find it. Before splitting, such marks survived only by being
fused to the glyph before them, which the metric counted as "retained". Separated, they
were routed to NOISE as orphans, and `1,234.56` became `123456`.
`_find_line_neighbour` now keeps a small mark that sits beside a normal glyph, from
mid-height down to just below the baseline, as TEXT with reason `t2:punctuation`. With
the splitter off, retention on the held-out set rose from 81.6% to **98.1%**. The old
figure had been inflated by merges all along.

## Known limitations

- **Ground-truth boxes are axis-aligned**, so under perspective warp we store the
  enclosing box of the warped glyph, which is slightly larger than its true ink extent
  and depresses measured IoU on warped samples. Warps are kept mild in the standard
  profiles; the aggressive ones live in the `hard_photo` profile and should be read per-
  slice.
- **Character metrics run on synthetic data only** until DDI-100 is wired up. The
  benchmark is currently grading its own homework at character level; the real datasets
  constrain it at region level, which is weaker.
- **`preprocess` is the largest latency stage** (~28 ms P50 of ~61 ms total), dominated
  by the morphological background estimate. That is the first thing to attack if the
  Phase 6 bake-off comes out unfavourable.
- **No deskew.** Phase 5 will need it; the rotation in the degradation profiles is
  currently absorbed as merge/miss rather than corrected.
