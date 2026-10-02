"""Labelling glyph clusters from ground truth, for training the Phase 3 CNN.

Each cluster (:func:`ocr.recog.crops.clusters_of`) gets one label:

character      one-to-one IoU >= 0.5 with a GT character's box (greedy, best first).
``<MULTI>``    the cluster covers >= half of two or more GT characters: a merge.
``<PART>``     >= 60% of the cluster lies inside one GT character, but it is not a match:
               a fragment or broken stroke.
``<NONTEXT>``  none of the above: speckle, texture, a rule fragment.

Same thresholds as the isolation metric (``ocrbench.metrics``), so "the CNN was trained
to call this a merge" and "the benchmark counts this as a merge" mean the same thing.
GT characters outside the charset produce label -1 (dropped), not NONTEXT: a real
character the model cannot name must not be taught as junk.
"""

from __future__ import annotations

import unicodedata

from ocr.recog.charset import GLYPH_INDEX, MULTI, NONTEXT, PART
from ocr.recog.crops import Cluster
from ocr.types import BBox

from .gt import Sample
from .metrics import BoxIndex

def is_tune_doc(sample_id: str) -> bool:
    """A fixed 10% of real training documents, reserved for tuning the decoder
    (LM weight, format bonus, beam). The glyph builder never trains the CNN on them,
    and the test benchmark never sees them: a third set, by stable hash of the id."""
    import hashlib

    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 10 == 0


def doc_seed(sample_id: str) -> int:
    """Stable per-document id for crops from real documents (they have no page seed),
    so CNN validation can hold out whole documents."""
    import hashlib

    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12], 16) or 1


IOU_MATCH = 0.5
COVER = 0.5
INSIDE = 0.6
ALNUM_MIN_HEIGHT = 0.35
"""In alignment labelling, a letter or digit cluster shorter than this fraction of its
word box's height rejects the word (a dot standing in for a fused letter)."""


def label_by_alignment(sample: Sample, clusters: list[Cluster]) -> list[int]:
    """Character labels for *real* documents, which annotate words (or lines), not
    characters.

    For each annotated word, the clusters whose centre lies inside its box are taken in
    x order. If their count equals the transcript's non-space characters, the k-th
    cluster *is* the k-th character. Any mismatch — fused glyphs, a ':' that is two
    dots, a '%' in three pieces — and the whole word is skipped (-1) rather than
    guessed: a wrong label teaches the CNN a confusion it then has with confidence.

    Clusters inside no annotated word are ``<NONTEXT>`` where the dataset's annotation is
    complete, and -1 where it is not (CORD's "dontcare" regions hold real, untranscribed
    text that must not be taught as junk).
    """
    regions = sample.words or sample.lines
    labels = [-1] * len(clusters)
    if not regions:
        return labels
    complete = bool(sample.meta.get("gt_complete", False))
    index = BoxIndex([r.bbox for r in regions])
    owner: list[int] = [-1] * len(clusters)
    for j, cl in enumerate(clusters):
        cx, cy = cl.bbox.cx, cl.bbox.cy
        for i in index.query(cl.bbox):
            b = regions[i].bbox
            if b.x <= cx <= b.x2 and b.y <= cy <= b.y2:
                owner[j] = i
                break
        if owner[j] < 0 and complete:
            labels[j] = GLYPH_INDEX[NONTEXT]

    members: dict[int, list[int]] = {}
    for j, i in enumerate(owner):
        if i >= 0:
            members.setdefault(i, []).append(j)
    for i, js in members.items():
        chars = [c for c in unicodedata.normalize("NFC", regions[i].text) if not c.isspace()]
        if len(chars) != len(js) or any(c not in GLYPH_INDEX for c in chars):
            continue
        ordered = sorted(js, key=lambda j: clusters[j].bbox.x)
        # Coincidental count matches are this method's failure mode: a smudge makes up
        # the count while a real letter fused with its neighbour. A letter or digit is
        # never a small fraction of its word's height, so such a pairing rejects the
        # whole word. (Punctuation is exempt — a full stop *is* small.)
        word_h = max(1, regions[i].bbox.h)
        if any(c.isalnum() and clusters[j].bbox.h < ALNUM_MIN_HEIGHT * word_h
               for j, c in zip(ordered, chars)):
            continue
        for j, c in zip(ordered, chars):
            labels[j] = GLYPH_INDEX[c]
    return labels


def label_clusters(sample: Sample, clusters: list[Cluster]) -> list[int]:
    gt = sample.chars or []
    gt_boxes: list[BBox] = [c.bbox for c in gt]
    labels = [GLYPH_INDEX[NONTEXT]] * len(clusters)
    if not gt_boxes:
        return labels
    index = BoxIndex(gt_boxes)

    # A cluster covering two or more characters is a merge, whatever its IoU with
    # either: two equal fused glyphs have IoU exactly 0.5 with each, which would
    # otherwise "match" and teach the CNN that an AB crop is an A. Same precedence as
    # the isolation metric.
    multi = set()
    for j, cl in enumerate(clusters):
        b = cl.bbox
        if sum(1 for i in index.query(b) if b.intersection_area(gt_boxes[i]) >= COVER * gt_boxes[i].area) >= 2:
            multi.add(j)
            labels[j] = GLYPH_INDEX[MULTI]

    pairs = []
    for j, cl in enumerate(clusters):
        if j in multi:
            continue
        for i in index.query(cl.bbox):
            v = cl.bbox.iou(gt_boxes[i])
            if v >= IOU_MATCH:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    matched: dict[int, int] = {}
    for _, i, j in pairs:
        if i not in used_g and j not in matched:
            used_g.add(i)
            matched[j] = i

    for j, cl in enumerate(clusters):
        b = cl.bbox
        if j in matched:
            ch = unicodedata.normalize("NFC", gt[matched[j]].char)
            labels[j] = GLYPH_INDEX.get(ch, -1)
            continue
        near = list(index.query(b))
        covered = sum(1 for i in near if b.intersection_area(gt_boxes[i]) >= COVER * gt_boxes[i].area)
        if covered >= 2:
            labels[j] = GLYPH_INDEX[MULTI]
        elif any(b.intersection_area(gt_boxes[i]) >= INSIDE * b.area for i in near):
            labels[j] = GLYPH_INDEX[PART]
    return labels
