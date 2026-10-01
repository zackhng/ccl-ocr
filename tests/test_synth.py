"""Synthetic-generator tests, centred on one question: is the ground truth true?

Every isolation number in this project is measured against boxes this generator
emitted. If a box does not actually contain the glyph it claims, the benchmark grades
the pipeline against fiction — and it would grade it *consistently*, so nothing
downstream would ever look wrong.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, Sample
from ocrbench.synth import fonts
from ocrbench.synth.degrade import PROFILES, degrade, transform_boxes
from ocrbench.synth.generate import generate_sample, plan
from ocrbench.synth.render import render
from ocrbench.synth.templates import TEMPLATES

SIMPLE_TEMPLATES = ["id_card", "cheque", "receipt", "form", "plain_latin"]


@pytest.mark.parametrize("template", SIMPLE_TEMPLATES)
def test_every_char_box_contains_ink(template):
    """The core ground-truth guarantee, checked on the undegraded render.

    A box claiming a character must have darker pixels inside it than the page around
    it. Checked before degradation so a failure points at the renderer rather than at
    blur.
    """
    spec = TEMPLATES[template](random.Random(3))
    result = render(spec)
    assert result.chars, f"{template} should emit character ground truth"

    gray = result.image.mean(axis=2)
    page_level = float(np.median(gray))

    empty = []
    for c in result.chars:
        b = c.bbox
        patch = gray[b.y : b.y2, b.x : b.x2]
        if patch.size == 0 or float(patch.min()) > page_level - 20:
            empty.append(c.char)

    # A handful of light-on-dark glyphs legitimately invert this test's assumption;
    # anything beyond a trickle means the box geometry is wrong.
    assert len(empty) <= 0.05 * len(result.chars), f"{len(empty)} empty boxes: {empty[:10]}"


@pytest.mark.parametrize("template", SIMPLE_TEMPLATES)
def test_boxes_stay_inside_the_image(template):
    spec = TEMPLATES[template](random.Random(5))
    result = render(spec)
    h, w = result.image.shape[:2]
    for c in result.chars or []:
        assert 0 <= c.bbox.x and c.bbox.x2 <= w + 1, c.char
        assert 0 <= c.bbox.y and c.bbox.y2 <= h + 1, c.char


def test_word_boxes_contain_their_characters():
    spec = TEMPLATES["plain_latin"](random.Random(11))
    result = render(spec)
    # Every character must fall inside some word box: the two are derived from the same
    # advances, so a mismatch means one of the two derivations is wrong.
    for c in result.chars[:200]:
        assert any(w.bbox.intersection_area(c.bbox) >= 0.5 * c.bbox.area for w in result.words), (
            f"character {c.char!r} at {c.bbox} lies in no word box"
        )


def test_shaped_scripts_emit_no_character_ground_truth():
    """Devanagari, Arabic and Thai reorder and join glyphs during shaping, so
    'the box of character i' is not a well-defined object. Claiming one would be
    fabricated ground truth that silently inflates every metric."""
    if "devanagari" not in fonts.available_fonts():
        pytest.skip("no Devanagari font installed")
    spec = TEMPLATES["plain_devanagari"](random.Random(2))
    result = render(spec)
    assert result.chars is None
    assert result.words, "word-level ground truth should still be emitted"


def test_unshapeable_scripts_are_skipped_not_faked():
    """supported_scripts() must exclude shaped scripts when Pillow has no Raqm."""
    supported = set(fonts.supported_scripts())
    if not fonts.has_shaping():
        assert not (supported & fonts.SHAPED_SCRIPTS)
    assert "latin" in supported


class TestDegradation:
    def test_identity_profile_leaves_boxes_untouched(self):
        boxes = [BBox(10, 10, 20, 20), BBox(50, 50, 30, 10)]
        image = np.full((200, 200, 3), 255, np.uint8)
        out = degrade(image, boxes, PROFILES["digital"], random.Random(0))
        assert out.boxes == boxes

    def test_scaling_scales_boxes(self):
        boxes = [BBox(100, 100, 40, 20)]
        image = np.full((400, 400, 3), 255, np.uint8)
        profile = PROFILES["digital"].__class__(
            name="half", capture="scan", scale_range=(0.5, 0.5)
        )
        out = degrade(image, boxes, profile, random.Random(0))
        assert out.image.shape[:2] == (200, 200)
        assert out.boxes[0] == BBox(50, 50, 20, 10)

    def test_boxes_leaving_the_frame_become_none(self):
        identity = np.eye(3, dtype=np.float32)
        shifted = identity.copy()
        shifted[0, 2] = 10_000  # push everything far off the right edge
        assert transform_boxes([BBox(10, 10, 20, 20)], shifted, 200, 200) == [None]

    def test_degraded_boxes_still_contain_ink(self):
        """Pixels and ground truth must go through the same geometry. A box that
        survives the warp but no longer sits on its glyph is worse than a dropped one."""
        spec = TEMPLATES["plain_latin"](random.Random(7))
        rendered = render(spec)
        boxes = [c.bbox for c in rendered.chars]
        out = degrade(rendered.image, boxes, PROFILES["photo"], random.Random(7))

        gray = out.image.mean(axis=2)
        page_level = float(np.median(gray))
        kept = [b for b in out.boxes if b is not None]
        assert len(kept) > 0.8 * len(boxes), "a mild photo profile should keep most boxes"

        inked = sum(
            1 for b in kept
            if gray[b.y : b.y2, b.x : b.x2].size
            and float(gray[b.y : b.y2, b.x : b.x2].min()) < page_level - 15
        )
        assert inked > 0.85 * len(kept), f"only {inked}/{len(kept)} degraded boxes contain ink"


class TestGenerate:
    def test_is_reproducible_from_its_seed(self):
        a_sample, a_image = generate_sample("x", "id_card", "photo", 42)
        b_sample, b_image = generate_sample("x", "id_card", "photo", 42)
        assert np.array_equal(a_image, b_image)
        assert a_sample.to_json() == b_sample.to_json()

    def test_different_seeds_differ(self):
        a, _ = generate_sample("x", "cheque", "scan", 1)
        b, _ = generate_sample("x", "cheque", "scan", 2)
        assert a.text != b.text

    def test_plan_is_deterministic_and_sized(self):
        first = list(plan(30, seed=3))
        assert len(first) == 30
        assert first == list(plan(30, seed=3))

    def test_pii_regions_are_annotated_but_not_acted_on(self):
        s, _ = generate_sample("x", "id_card", "clean_scan", 9)
        kinds = {p.type for p in s.pii}
        assert "nric" in kinds
        assert "face" in kinds

    def test_capture_mode_follows_the_profile(self):
        s, _ = generate_sample("x", "form", "photo", 4)
        assert s.capture == "photo"
        s, _ = generate_sample("x", "form", "clean_scan", 4)
        assert s.capture == "scan"


def test_store_round_trip(tmp_path):
    store = BenchmarkStore(tmp_path)
    sample, image = generate_sample("synth_form_0000", "form", "scan", 21)
    store.write(sample, image)

    loaded = store.read("synth_form_0000")
    assert loaded.to_json() == sample.to_json()
    assert store.sample_ids() == ["synth_form_0000"]

    manifest = store.write_manifest()
    assert manifest.exists()


def test_store_rejects_nothing_silently(tmp_path):
    """A Sample must survive JSON round-tripping exactly, including chars=None."""
    store = BenchmarkStore(tmp_path)
    s = Sample(sample_id="a", source="cheque", capture="scan", script="latin", dpi=300)
    s.regions_nontext = [__import__("ocrbench.gt", fromlist=["GTBox"]).GTBox(BBox(0, 0, 5, 5), type="signature")]
    store.write(s)
    assert store.read("a").chars is None
    assert store.read("a").regions_nontext[0].type == "signature"
