"""Degradations, applied to pixels and ground truth together.

A benchmark built only from crisp renders measures nothing: CCL is near-perfect on a
born-digital page and the interesting failures all live in capture. So each sample is
pushed through a profile that simulates a real acquisition path, and every ground-truth
box is carried through the same geometry.

One honest limitation, stated here because it bounds what the metrics mean: ground-truth
boxes are axis-aligned, so under perspective warp we store the *enclosing* box of the
warped glyph. That box is slightly larger than the glyph's true ink extent, which
depresses measured IoU a little on warped samples. Warps are therefore kept mild in the
standard profiles, and the aggressive ones are confined to the ``adversarial`` stratum
and reported separately rather than averaged in.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr.types import BBox


@dataclass(frozen=True, slots=True)
class DegradeProfile:
    name: str
    capture: str

    rotation_deg: float = 0.0
    """Max absolute skew. Flatbed scans land within a degree or two; phones do worse."""

    perspective: float = 0.0
    """Max corner displacement as a fraction of the image dimension."""

    blur_sigma: float = 0.0
    illumination: float = 0.0
    """Strength of the shadow/hotspot field, 0-1."""

    noise_sigma: float = 0.0
    jpeg_quality: int = 0
    """0 disables JPEG round-tripping."""

    scale_range: tuple[float, float] = (1.0, 1.0)
    """Resolution ladder. Applied last so boxes scale exactly."""

    moire: float = 0.0


PROFILES: dict[str, DegradeProfile] = {
    "digital": DegradeProfile("digital", "digital"),
    "clean_scan": DegradeProfile(
        "clean_scan", "scan", rotation_deg=0.6, blur_sigma=0.4, noise_sigma=2.0,
        jpeg_quality=92, scale_range=(0.9, 1.0),
    ),
    "scan": DegradeProfile(
        "scan", "scan", rotation_deg=1.8, blur_sigma=0.8, noise_sigma=4.0,
        illumination=0.12, jpeg_quality=80, scale_range=(0.55, 0.9),
    ),
    "photo": DegradeProfile(
        "photo", "photo", rotation_deg=2.5, perspective=0.025, blur_sigma=1.1,
        illumination=0.38, noise_sigma=6.0, jpeg_quality=72, scale_range=(0.5, 0.85),
    ),
    "hard_photo": DegradeProfile(
        "hard_photo", "photo", rotation_deg=5.0, perspective=0.055, blur_sigma=2.0,
        illumination=0.55, noise_sigma=10.0, jpeg_quality=48, scale_range=(0.3, 0.55),
        moire=0.10,
    ),
}


@dataclass(slots=True)
class DegradeResult:
    image: np.ndarray
    boxes: list[BBox | None]
    """Same length and order as the input. ``None`` means the box left the frame."""

    meta: dict = field(default_factory=dict)


def _homography(w: int, h: int, profile: DegradeProfile, rng: random.Random) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()

    if profile.perspective > 0:
        jitter = profile.perspective
        for i in range(4):
            dst[i, 0] += rng.uniform(-jitter, jitter) * w
            dst[i, 1] += rng.uniform(-jitter, jitter) * h

    H = cv2.getPerspectiveTransform(src, dst)

    if profile.rotation_deg > 0:
        angle = rng.uniform(-profile.rotation_deg, profile.rotation_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        R = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        H = R @ H

    return H.astype(np.float32)


def transform_boxes(boxes: list[BBox], H: np.ndarray, w: int, h: int) -> list[BBox | None]:
    """Map boxes through a homography, taking the enclosing axis-aligned box."""
    if not boxes:
        return []
    corners = np.empty((len(boxes) * 4, 1, 2), dtype=np.float32)
    for i, b in enumerate(boxes):
        corners[i * 4 : i * 4 + 4, 0] = [(b.x, b.y), (b.x2, b.y), (b.x2, b.y2), (b.x, b.y2)]

    warped = cv2.perspectiveTransform(corners, H).reshape(len(boxes), 4, 2)

    out: list[BBox | None] = []
    for i, b in enumerate(boxes):
        xs, ys = warped[i, :, 0], warped[i, :, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(w), x1), min(float(h), y1)
        if cx1 - cx0 < 1 or cy1 - cy0 < 1:
            out.append(None)
            continue
        # Mostly out of frame: a sliver of a glyph is not the glyph, and keeping it
        # would make a legitimate miss look like a detection failure.
        if (cx1 - cx0) * (cy1 - cy0) < 0.4 * max(1.0, (x1 - x0) * (y1 - y0)):
            out.append(None)
            continue
        out.append(BBox(round(cx0), round(cy0), round(cx1 - cx0), round(cy1 - cy0)))
    return out


def _illumination_field(w: int, h: int, strength: float, rng: random.Random) -> np.ndarray:
    """A smooth multiplicative field: one bright hotspot, one dark corner.

    This is what separates a phone capture from a scan, and it is the reason the
    pipeline normalises illumination before thresholding at all.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    radius = max(w, h) * rng.uniform(0.45, 0.9)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    hotspot = 1.0 + strength * (1.0 - np.clip(dist, 0, 1.5))

    angle = rng.uniform(0, 2 * math.pi)
    ramp = (math.cos(angle) * xx / w + math.sin(angle) * yy / h)
    gradient = 1.0 - strength * 0.6 * (ramp - ramp.min()) / max(1e-6, float(np.ptp(ramp)))

    return (hotspot * gradient).astype(np.float32)


def _add_moire(image: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]
    period = rng.uniform(3.0, 7.0)
    angle = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pattern = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * 2 * math.pi / period)
    delta = (pattern * strength * 255).astype(np.float32)[:, :, None]
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def degrade(
    image: np.ndarray,
    boxes: list[BBox],
    profile: DegradeProfile,
    rng: random.Random,
) -> DegradeResult:
    """Apply a capture profile to an image and its ground-truth boxes."""
    h, w = image.shape[:2]
    meta: dict = {"profile": profile.name, "capture": profile.capture}
    out_boxes: list[BBox | None]

    if profile.perspective > 0 or profile.rotation_deg > 0:
        H = _homography(w, h, profile, rng)
        image = cv2.warpPerspective(
            image, H, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_boxes = transform_boxes(boxes, H, w, h)
    else:
        out_boxes = list(boxes)

    if profile.illumination > 0:
        field = _illumination_field(w, h, profile.illumination, rng)
        image = np.clip(image.astype(np.float32) * field[:, :, None], 0, 255).astype(np.uint8)

    if profile.moire > 0:
        image = _add_moire(image, profile.moire, rng)

    if profile.blur_sigma > 0:
        sigma = rng.uniform(profile.blur_sigma * 0.5, profile.blur_sigma)
        if sigma > 0.05:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
            meta["blur_sigma"] = round(sigma, 2)

    if profile.noise_sigma > 0:
        noise = np.random.default_rng(rng.randrange(1 << 32)).normal(
            0, profile.noise_sigma, image.shape
        )
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    lo, hi = profile.scale_range
    if lo != 1.0 or hi != 1.0:
        scale = rng.uniform(lo, hi)
        new_w, new_h = max(32, round(w * scale)), max(32, round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out_boxes = [b.scaled(scale) if b is not None else None for b in out_boxes]
        meta["scale"] = round(scale, 3)
        w, h = new_w, new_h

    if profile.jpeg_quality:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, profile.jpeg_quality])
        if ok:
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        meta["jpeg_quality"] = profile.jpeg_quality

    # Final clip: rounding during scaling can push a box one pixel past the edge.
    clipped: list[BBox | None] = []
    for b in out_boxes:
        if b is None:
            clipped.append(None)
            continue
        x0, y0 = max(0, b.x), max(0, b.y)
        x1, y1 = min(w, b.x2), min(h, b.y2)
        clipped.append(BBox(x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None)

    meta["output_size"] = [w, h]
    return DegradeResult(image=image, boxes=clipped, meta=meta)
