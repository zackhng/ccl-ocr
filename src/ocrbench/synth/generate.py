"""Compose templates, rendering and degradation into benchmark samples.

Every sample is a pure function of its seed, so a run is reproducible from the manifest
alone — which matters because the images are gitignored and must be regenerable rather
than archived.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from ocr.types import BBox
from ocrbench.gt import BenchmarkStore, GTBox, GTChar, Sample

from . import fonts
from .degrade import PROFILES, degrade
from .render import render
from .templates import TEMPLATES


@dataclass(frozen=True, slots=True)
class Stratum:
    template: str
    weight: float
    profiles: tuple[tuple[str, float], ...]
    """(profile name, weight). Capture mode is taken from the chosen profile."""


# Weighted so the mix matches what the product actually sees: ID cards and cheques are
# mostly phone captures, receipts and forms mostly scans. The hard_photo share is what
# makes P95/P99 latency meaningful rather than a restatement of the median.
DEFAULT_STRATA: tuple[Stratum, ...] = (
    Stratum("id_card", 0.24, (("photo", 0.5), ("scan", 0.25), ("hard_photo", 0.15), ("clean_scan", 0.10))),
    Stratum("cheque", 0.18, (("photo", 0.4), ("scan", 0.35), ("hard_photo", 0.15), ("clean_scan", 0.10))),
    Stratum("receipt", 0.16, (("photo", 0.45), ("scan", 0.35), ("hard_photo", 0.20))),
    Stratum("form", 0.14, (("scan", 0.55), ("clean_scan", 0.2), ("photo", 0.25))),
    Stratum("plain_latin", 0.10, (("scan", 0.5), ("clean_scan", 0.3), ("photo", 0.2))),
    Stratum("plain_han", 0.05, (("scan", 0.6), ("photo", 0.4))),
    Stratum("plain_devanagari", 0.05, (("scan", 0.6), ("photo", 0.4))),
    Stratum("plain_thai", 0.04, (("scan", 0.6), ("photo", 0.4))),
    Stratum("plain_arabic", 0.04, (("scan", 0.6), ("photo", 0.4))),
)


def _weighted_choice(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    names = [n for n, _ in options]
    weights = [w for _, w in options]
    return rng.choices(names, weights=weights, k=1)[0]


def _available_strata(strata: tuple[Stratum, ...], warn: bool = True) -> list[Stratum]:
    """Drop strata this machine cannot render correctly.

    A script is dropped when its font is missing, or when it needs shaping and Pillow
    has no Raqm backend. Falling back to Latin instead would put samples labelled
    ``script=thai`` in the benchmark that contain no Thai at all, and the per-script
    slice would be fiction.
    """
    have = set(fonts.supported_scripts())
    out, skipped = [], []
    for s in strata:
        if s.template.startswith("plain_"):
            script = s.template.removeprefix("plain_")
            if script != "latin" and script not in have:
                skipped.append(script)
                continue
        out.append(s)
    if warn and skipped:
        reasons = []
        installed = set(fonts.available_fonts())
        no_font = [s for s in skipped if s not in installed]
        no_shaping = [s for s in skipped if s in installed]
        if no_font:
            reasons.append(f"{', '.join(no_font)}: no font installed")
        if no_shaping:
            reasons.append(
                f"{', '.join(no_shaping)}: Pillow has no Raqm backend, so these scripts "
                "cannot be shaped correctly"
            )
        print(f"  note: skipping strata -- {'; '.join(reasons)}")
    return out


def generate_sample(sample_id: str, template: str, profile_name: str, seed: int) -> tuple[Sample, "object"]:
    """Build one sample. Returns ``(sample, bgr_image)``."""
    rng = random.Random(seed)
    spec = TEMPLATES[template](rng)
    rendered = render(spec)
    profile = PROFILES[profile_name]

    # One flat list so every annotation goes through identical geometry. Order is
    # fixed on the way out and sliced back apart on the way in.
    n_lines, n_words = len(rendered.lines), len(rendered.words)
    chars = rendered.chars or []
    n_chars, n_pii = len(chars), len(rendered.pii)
    flat: list[BBox] = (
        [b.bbox for b in rendered.lines]
        + [b.bbox for b in rendered.words]
        + [c.bbox for c in chars]
        + [b.bbox for b in rendered.pii]
        + [b.bbox for b in rendered.regions_nontext]
    )

    result = degrade(rendered.image, flat, profile, rng)
    boxes = result.boxes

    i = 0
    out_lines = [GTBox(b, src.text, src.type) for src, b in zip(rendered.lines, boxes[i : i + n_lines]) if b]
    i += n_lines
    out_words = [GTBox(b, src.text, src.type) for src, b in zip(rendered.words, boxes[i : i + n_words]) if b]
    i += n_words
    out_chars = [GTChar(b, src.char) for src, b in zip(chars, boxes[i : i + n_chars]) if b]
    i += n_chars
    out_pii = [GTBox(b, src.text, src.type) for src, b in zip(rendered.pii, boxes[i : i + n_pii]) if b]
    i += n_pii
    out_nontext = [GTBox(b, src.text, src.type) for src, b in zip(rendered.regions_nontext, boxes[i:]) if b]

    effective_dpi = round(spec.dpi * result.meta.get("scale", 1.0))
    sample = Sample(
        sample_id=sample_id,
        source="synth",
        capture=profile.capture,
        script=spec.script,
        dpi=effective_dpi,
        text=rendered.text,
        lines=out_lines,
        words=out_words,
        chars=out_chars if rendered.chars is not None else None,
        pii=out_pii,
        regions_nontext=out_nontext,
        meta={
            "template": spec.meta.get("template", template),
            "seed": seed,
            "render_size": [spec.width, spec.height],
            **result.meta,
        },
    )
    return sample, result.image


def plan(count: int, seed: int, strata: tuple[Stratum, ...] = DEFAULT_STRATA) -> Iterator[tuple[str, str, str, int]]:
    """Yield ``(sample_id, template, profile, seed)`` for ``count`` samples."""
    rng = random.Random(seed)
    usable = _available_strata(strata)
    names = [s.template for s in usable]
    weights = [s.weight for s in usable]
    by_name = {s.template: s for s in usable}

    counters: dict[str, int] = {}
    for i in range(count):
        template = rng.choices(names, weights=weights, k=1)[0]
        profile = _weighted_choice(rng, by_name[template].profiles)
        n = counters.get(template, 0)
        counters[template] = n + 1
        yield f"synth_{template}_{n:04d}", template, profile, seed * 100003 + i


def generate(
    store: BenchmarkStore,
    count: int,
    seed: int = 7,
    strata: tuple[Stratum, ...] = DEFAULT_STRATA,
    progress: bool = True,
) -> list[Sample]:
    """Generate ``count`` samples into ``store``."""
    store.ensure_dirs()
    samples: list[Sample] = []
    for n, (sample_id, template, profile, sample_seed) in enumerate(plan(count, seed, strata), 1):
        sample, image = generate_sample(sample_id, template, profile, sample_seed)
        store.write(sample, image)
        samples.append(sample)
        if progress and (n % 20 == 0 or n == count):
            print(f"  generated {n}/{count}")
    return samples
