"""Why are characters not cleanly isolated? A visual dossier of failures.

For every synthetic sample (the only source with character ground truth), each
character that CCL did not isolate is classified by *cause*, and for a stratified
subset of ~30 images, zoomed crops are written showing what CCL saw.

    uv run python scripts/diagnose_isolation.py                 # -> bench/diagnostics/isolation/
    uv run python scripts/diagnose_isolation.py --images 30 --per-image 6

Output:
    index.html      every selected image's failures, crops side by side, with causes
    summary.md      cause breakdown over ALL synthetic samples, by capture profile
    failures.csv    one row per failure (all samples), for your own slicing

Each crop has three panels, enlarged:
    1. the original pixels around the character
    2. the binary image CCL actually labelled (black = ink)
    3. the original with boxes: green = this GT character, red = the component(s)
       responsible, grey = other surviving components, blue = annotated non-text
       (rule, frame, photo, barcode...)

Causes are assigned by geometry, in this order, and are hints, not proof:

merged / ambiguous (the character reached the classifier inside a bigger component)
    joined to <type>      the component runs into an annotated non-text region —
                          a rule, a frame's border, a photo, a barcode
    joined to background  the component is far bigger than the characters it covers
                          (>= 1.8x their union) with no non-text region to blame —
                          ink grown from background texture, shadow or speckle
    glyph touch           two or more characters fused, and nothing else
    ambiguous match       one component matches two characters about equally
missed (nothing usable reached the classifier)
    filtered as <kind>    ink was there but the filter routed it away (with reason)
    faded                 < 5% ink inside the character's box after thresholding
    fragment              ink present, but only in pieces too small to count
over_segmented
    broken into N         strokes split into N components

Per failure the dossier also records local **contrast** (median background grey minus
median ink grey, 0-255), **background noise** (grey std in the ring around the box,
ink excluded) and **speckle density** (NOISE components within three glyph heights,
per glyph-height squared) — the numbers that say "low contrast" or "speckled page"
rather than leaving it to the eye.
"""

from __future__ import annotations

import argparse
import csv
import html
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.engine import CCLEngine, load_image  # noqa: E402
from ocr.types import BBox, ComponentKind, PageResult  # noqa: E402
from ocrbench.gt import BenchmarkStore, Sample  # noqa: E402
from ocrbench.metrics import CharOutcome, char_metrics, surviving_components  # noqa: E402

FAILURES = ("merged", "ambiguous", "over_segmented", "missed")
STRATA = (("hard_photo", 7), ("photo", 11), ("scan", 8), ("clean_scan", 4))


@dataclass
class Failure:
    sample_id: str
    profile: str
    template: str
    char: str
    outcome: str
    cause: str
    bbox: BBox
    glyph_h: float
    contrast: float | None
    bg_noise: float | None
    speckle_density: float
    gap_to_neighbour: float | None = None
    """For glyph touches: GT gap to the nearest fused neighbour, in glyph heights. Weak
    evidence only — GT boxes are enclosing boxes at capture resolution, so most
    neighbours measure <= 0 regardless (see the summary's comparison)."""
    blame: list[BBox] = field(default_factory=list)
    crop: str = ""


# --------------------------------------------------------------------------- causes


def _frame_band(b: BBox, img_w: int, img_h: int) -> list[BBox]:
    t = max(3, int(0.01 * min(img_w, img_h)))
    return [BBox(b.x, b.y, b.w, t), BBox(b.x, b.y2 - t, b.w, t),
            BBox(b.x, b.y, t, b.h), BBox(b.x2 - t, b.y, t, b.h)]


def _nontext_hit(comp: BBox, sample: Sample, img_w: int, img_h: int) -> str | None:
    best, best_frac = None, 0.0
    for r in sample.regions_nontext:
        parts = _frame_band(r.bbox, img_w, img_h) if r.type == "frame" else [r.bbox]
        inter = sum(comp.intersection_area(p) for p in parts)
        frac = inter / comp.area if comp.area else 0.0
        if frac > best_frac:
            best, best_frac = r.type or "non-text", frac
    return best if best_frac >= 0.15 else None


def _union(boxes: list[BBox]) -> BBox:
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


def classify(o: CharOutcome, sample: Sample, result: PageResult, surv, binary_orig: np.ndarray,
             img_w: int, img_h: int) -> tuple[str, list[BBox], float | None]:
    gb = sample.chars[o.index].bbox
    gt = [c.bbox for c in sample.chars]
    blame = [surv[j].bbox for j in o.components]

    if o.outcome in ("merged", "ambiguous"):
        if not blame:
            return "glyph touch", blame, None
        comp = max(blame, key=lambda b: b.area)
        hit = _nontext_hit(comp, sample, img_w, img_h)
        if hit:
            return f"joined to {hit}", blame, None
        if o.outcome == "ambiguous":
            return "ambiguous match", blame, None
        covered = [b for b in gt if comp.intersection_area(b) >= 0.5 * b.area]
        if covered and comp.area >= 1.8 * _union(covered).area:
            return "joined to background", blame, None
        others = [b for b in covered if b != gb and abs(b.cy - gb.cy) < gb.h / 2]
        gap = None
        if others:
            g = min(max(b.x - gb.x2, gb.x - b.x2) for b in others)
            gap = g / max(1, gb.h)
        return "glyph touch", blame, gap

    if o.outcome == "over_segmented":
        return f"broken into {len(o.components)}", blame, None

    # missed
    overlapping = [c for c in result.components
                   if c.bbox.intersection_area(gb) >= 0.3 * gb.area
                   and c.kind not in (ComponentKind.TEXT, ComponentKind.DIACRITIC)]
    if overlapping:
        c = max(overlapping, key=lambda c: c.bbox.intersection_area(gb))
        return f"filtered as {c.kind.value} ({c.reason})", [c.bbox], None
    # A metric artefact, not a pipeline failure: exactly one surviving component sits
    # inside this character's box, spans most of its height and covers no other
    # character, yet IoU < 0.5 because the GT box is wider than the ink. Thin glyphs
    # (I, l, 1, :) are 2-3 px wide, so a 1 px side bearing or the enclosing box of a
    # rotated glyph is enough to fail the match.
    inside = [c for c in surv
              if c.bbox.intersection_area(gb) >= 0.8 * c.bbox.area
              and c.bbox.h >= 0.7 * gb.h
              and not any(c.bbox.intersection_area(o) >= 0.5 * o.area for o in gt if o != gb)]
    if len(inside) == 1:
        return "isolated, box mismatch (metric)", [inside[0].bbox], None
    crop = binary_orig[gb.y : gb.y2, gb.x : gb.x2]
    ink = float((crop > 0).mean()) if crop.size else 0.0
    if ink < 0.05:
        return "faded", [], None
    return "fragment", [c.bbox for c in result.components if c.bbox.intersection_area(gb) > 0], None


def indicators(gray: np.ndarray, binary_orig: np.ndarray, gb: BBox, noise_centres: np.ndarray,
               glyph_h: float) -> tuple[float | None, float | None, float]:
    h, w = gray.shape
    box = gray[gb.y : gb.y2, gb.x : gb.x2]
    ink = binary_orig[gb.y : gb.y2, gb.x : gb.x2] > 0
    pad = max(2, gb.h // 3)
    x0, y0, x1, y1 = max(0, gb.x - pad), max(0, gb.y - pad), min(w, gb.x2 + pad), min(h, gb.y2 + pad)
    ring = gray[y0:y1, x0:x1].astype(np.float32)
    ring_ink = binary_orig[y0:y1, x0:x1] > 0
    bg = ring[~ring_ink]
    contrast = None
    if ink.any() and bg.size:
        contrast = float(np.median(bg) - np.median(box[ink]))
    elif box.size:
        contrast = float(np.percentile(box, 90) - np.percentile(box, 10))
    noise = float(bg.std()) if bg.size > 8 else None

    r = 3 * glyph_h
    density = 0.0
    if len(noise_centres):
        d = np.abs(noise_centres - np.array([gb.cx, gb.cy]))
        n = int(((d[:, 0] <= r) & (d[:, 1] <= r)).sum())
        density = n / ((2 * r / glyph_h) ** 2)
    return contrast, noise, density


# ---------------------------------------------------------------------------- crops


def render_crop(image: np.ndarray, binary_orig: np.ndarray, f: Failure, surv_boxes: list[BBox],
                nontext: list[BBox], out: Path) -> None:
    h, w = image.shape[:2]
    focus = _union([f.bbox] + f.blame) if f.blame else f.bbox
    m = int(1.5 * f.glyph_h)
    x0, y0 = max(0, focus.x - m), max(0, focus.y - m)
    x1, y1 = min(w, focus.x2 + m), min(h, focus.y2 + m)
    # Never wider than ~14 glyphs: a component joined to a page-wide rule would
    # otherwise produce an unreadable sliver.
    max_w = int(14 * f.glyph_h)
    if x1 - x0 > max_w:
        cx = f.bbox.cx
        x0, x1 = max(0, int(cx - max_w / 2)), min(w, int(cx + max_w / 2))

    orig = image[y0:y1, x0:x1].copy()
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    binv = cv2.cvtColor(255 - binary_orig[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    over = orig.copy()

    def rect(b: BBox, color, t):
        cv2.rectangle(over, (b.x - x0, b.y - y0), (b.x2 - x0 - 1, b.y2 - y0 - 1), color, t)

    for b in nontext:
        rect(b, (220, 120, 40), 1)
    for b in surv_boxes:
        if b.intersection_area(BBox(x0, y0, x1 - x0, y1 - y0)):
            rect(b, (150, 150, 150), 1)
    for b in f.blame:
        rect(b, (40, 40, 230), 1)
    rect(f.bbox, (40, 190, 40), 1)

    scale = max(1.0, 150.0 / max(1, y1 - y0))
    panels = [cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
              for p in (orig, binv, over)]
    sep = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.hconcat([panels[0], sep, panels[1], sep, panels[2]]))


# ---------------------------------------------------------------------------- main


def analyse(sample: Sample, engine: CCLEngine, store: BenchmarkStore):
    image = load_image(str(store.image_path(sample.sample_id)))
    result, debug = engine.run_with_debug(image)
    h, w = image.shape[:2]
    binary_orig = cv2.resize(debug.binary, (w, h), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    surv = surviving_components(result)
    detail: list[CharOutcome] = []
    char_metrics(sample, result, detail)

    glyph_h = float(np.median([c.bbox.h for c in sample.chars]))
    noise_centres = np.array([[c.bbox.cx, c.bbox.cy] for c in result.components
                              if c.kind is ComponentKind.NOISE] or np.zeros((0, 2)))
    failures = []
    for o in detail:
        if o.outcome not in FAILURES:
            continue
        gb = sample.chars[o.index].bbox
        cause, blame, gap = classify(o, sample, result, surv, binary_orig, w, h)
        contrast, noise, density = indicators(gray, binary_orig, gb, noise_centres, glyph_h)
        failures.append(Failure(
            sample.sample_id, str(sample.meta.get("profile")), str(sample.meta.get("template")),
            sample.chars[o.index].char, o.outcome, cause, gb, glyph_h,
            contrast, noise, density, gap, blame,
        ))
    return image, binary_orig, result, failures, len(detail)


def cause_family(cause: str) -> str:
    if cause.startswith("filtered as"):
        return "filtered (" + cause.split()[2] + ")"
    if cause.startswith("broken into"):
        return "broken strokes"
    if cause.startswith("joined to") and cause != "joined to background":
        return "joined to non-text: " + cause.split("joined to ")[1]
    return cause


def select_samples(samples: list[Sample], n_images: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    by_profile: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_profile[str(s.meta.get("profile"))].append(s)
    total = sum(k for _, k in STRATA)
    chosen: list[Sample] = []
    for profile, k in STRATA:
        pool = by_profile.get(profile, [])
        # Round-robin over templates so one stratum is not all ID cards.
        by_t: dict[str, list[Sample]] = defaultdict(list)
        for s in pool:
            by_t[str(s.meta.get("template"))].append(s)
        for lst in by_t.values():
            rng.shuffle(lst)
        want = max(1, round(k * n_images / total))
        order = sorted(by_t)
        while want and any(by_t.values()):
            for t in order:
                if by_t[t] and want:
                    chosen.append(by_t[t].pop())
                    want -= 1
    return chosen


def pick_failures(failures: list[Failure], k: int) -> list[Failure]:
    """Up to k per image, diverse in cause: one of each cause first, then the rest."""
    by_cause: dict[str, list[Failure]] = defaultdict(list)
    for f in failures:
        by_cause[cause_family(f.cause)].append(f)
    out: list[Failure] = []
    while len(out) < k and any(by_cause.values()):
        for c in sorted(by_cause, key=lambda c: -len(by_cause[c])):
            if by_cause[c] and len(out) < k:
                out.append(by_cause[c].pop(0))
    return out


def fmt(v, spec=".0f"):
    return "–" if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--out", default="bench/diagnostics/isolation")
    ap.add_argument("--images", type=int, default=30)
    ap.add_argument("--per-image", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    store = BenchmarkStore(args.bench)
    samples = [s for s in (store.read(i) for i in store.sample_ids())
               if s.has_char_gt and store.image_path(s.sample_id).exists()]
    engine = CCLEngine()
    out = Path(args.out)
    chosen = {s.sample_id for s in select_samples(samples, args.images, args.seed)}

    all_failures: list[Failure] = []
    all_pair_gaps: list[int] = []
    totals: Counter = Counter()
    sections: list[str] = []
    for s in samples:
        image, binary_orig, result, failures, n_chars = analyse(s, engine, store)
        profile = str(s.meta.get("profile"))
        totals[(profile, "chars")] += n_chars
        rows_sorted = sorted(s.chars, key=lambda c: (round(c.bbox.cy / max(1, c.bbox.h)), c.bbox.x))
        for c1, c2 in zip(rows_sorted, rows_sorted[1:]):
            if abs(c1.bbox.cy - c2.bbox.cy) < c1.bbox.h / 2 and 0 <= c2.bbox.x - c1.bbox.x < 3 * c1.bbox.h:
                all_pair_gaps.append(c2.bbox.x - c1.bbox.x2)
        all_failures += failures
        if s.sample_id not in chosen or not failures:
            continue

        surv_boxes = [c.bbox for c in surviving_components(result)]
        nontext = [r.bbox for r in s.regions_nontext if r.type != "frame"]
        rows = []
        for k, f in enumerate(pick_failures(failures, args.per_image)):
            f.crop = f"crops/{s.sample_id}_{k}.png"
            render_crop(image, binary_orig, f, surv_boxes, nontext, out / f.crop)
            rows.append(
                f"<tr><td><img src='{f.crop}'></td><td><b>{html.escape(f.char)}</b><br>"
                f"{f.outcome}<br><b>{html.escape(f.cause)}</b>"
                + (f"<br>GT gap {f.gap_to_neighbour:+.2f} h" if f.gap_to_neighbour is not None else "")
                + f"</td><td>contrast {fmt(f.contrast)}<br>bg noise {fmt(f.bg_noise, '.1f')}"
                f"<br>speckle {f.speckle_density:.2f}<br>glyph {f.glyph_h:.0f} px</td></tr>"
            )
        n_fail = len(failures)
        causes = Counter(cause_family(f.cause) for f in failures).most_common()
        meta = s.meta
        sections.append(
            f"<h2>{s.sample_id}</h2><p>{profile} · {meta.get('template')} · "
            f"blur σ {meta.get('blur_sigma', '–')} · scale {meta.get('scale', '–')} · "
            f"JPEG q{meta.get('jpeg_quality', '–')} · <b>{n_fail}/{n_chars}</b> characters not isolated: "
            + ", ".join(f"{html.escape(c)} {n}" for c, n in causes)
            + f"</p><table>{''.join(rows)}</table>"
        )

    # --- CSV over everything
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "failures.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["sample", "profile", "template", "char", "outcome", "cause", "family",
                     "x", "y", "w", "h", "glyph_h", "contrast", "bg_noise", "speckle_density",
                     "gt_gap_h", "crop"])
        for f in all_failures:
            wr.writerow([f.sample_id, f.profile, f.template, f.char, f.outcome, f.cause,
                         cause_family(f.cause), f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h,
                         round(f.glyph_h, 1), fmt(f.contrast, ".1f"), fmt(f.bg_noise, ".1f"),
                         round(f.speckle_density, 3),
                         "" if f.gap_to_neighbour is None else round(f.gap_to_neighbour, 3), f.crop])

    # --- summary over everything
    profiles = [p for p, _ in STRATA]
    fam = Counter((f.profile, cause_family(f.cause)) for f in all_failures)
    families = sorted({cause_family(f.cause) for f in all_failures},
                      key=lambda c: -sum(fam[(p, c)] for p in profiles))
    lines = ["# Isolation failures: causes", "",
             f"All {len(samples)} synthetic samples, current default engine. A rate is the "
             "share of *all* characters in that profile lost to that cause.", "",
             "| cause | " + " | ".join(profiles) + " | all |", "|---" * (len(profiles) + 2) + "|"]
    for c in families:
        cells = []
        for p in profiles:
            n = totals[(p, "chars")]
            cells.append(f"{fam[(p, c)] / n:.1%}" if n else "–")
        total_n = sum(totals[(p, 'chars')] for p in profiles)
        cells.append(f"{sum(fam[(p, c)] for p in profiles) / total_n:.1%}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("| **all failures** | " + " | ".join(
        f"**{sum(fam[(p, c)] for c in families) / totals[(p, 'chars')]:.1%}**" if totals[(p, 'chars')] else "–"
        for p in profiles) + f" | **{len(all_failures) / sum(totals[(p, 'chars')] for p in profiles):.1%}** |")

    def stat(fs, attr):
        v = [getattr(f, attr) for f in fs if getattr(f, attr) is not None]
        return f"{np.median(v):.1f}" if v else "–"

    lines += ["", "## Local conditions by cause (medians)", "",
              "| cause | n | contrast | bg noise | speckle density | GT gap (h) |", "|---|---|---|---|---|---|"]
    by_fam: dict[str, list[Failure]] = defaultdict(list)
    for f in all_failures:
        by_fam[cause_family(f.cause)].append(f)
    for c in families:
        fs = by_fam[c]
        lines.append(f"| {c} | {len(fs)} | {stat(fs, 'contrast')} | {stat(fs, 'bg_noise')} | "
                     f"{stat(fs, 'speckle_density')} | {stat(fs, 'gap_to_neighbour')} |")
    touches = [f.gap_to_neighbour for f in all_failures if f.gap_to_neighbour is not None]
    if touches:
        t = np.array(touches)
        base = np.array(all_pair_gaps) if all_pair_gaps else np.zeros(1)
        lines += ["", f"GT gap between fused neighbours: {np.mean(t <= 0):.0%} are <= 0 px, against "
                  f"{np.mean(base <= 0):.0%} of *all* adjacent character pairs. Ground-truth boxes "
                  "are measured after downscale, rounding and rotation (enclosing boxes), so most "
                  "neighbours 'touch' on paper whether or not their ink does: this gap cannot tell "
                  "font kerning from capture blur, and is reported only as that comparison."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    page = (
        "<!doctype html><meta charset='utf-8'><title>Isolation failures</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
        "table{border-collapse:collapse;margin-bottom:24px}td{border:1px solid #ddd;padding:6px;"
        "vertical-align:top}img{image-rendering:pixelated;max-width:900px}h2{margin-top:32px}"
        ".legend span{display:inline-block;margin-right:14px}</style>"
        "<h1>Characters not cleanly isolated</h1>"
        "<p class='legend'>Panels: original · binary CCL saw · boxes. "
        "<span style='color:#28be28'>■ GT character</span><span style='color:#e62828'>■ responsible component</span>"
        "<span style='color:#999'>■ other components</span><span style='color:#2878dc'>■ annotated non-text</span></p>"
        f"<p>Cause breakdown over all samples: <a href='summary.md'>summary.md</a> · "
        f"every failure: <a href='failures.csv'>failures.csv</a></p>"
        + "".join(sections)
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(all_failures)} failures over {len(samples)} samples; "
          f"{len(sections)} images in the dossier -> {out / 'index.html'}")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
