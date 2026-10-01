"""Overlay rendering — the actual Phase 1 deliverable.

The milestone question ("does CCL isolate character candidates on our documents?") is
not answered by a number alone. A 0.82 isolation recall could be uniform mild
over-segmentation or a clean 0.95 everywhere except the MICR band; those have
completely different fixes. Colour-coding by routing decision, and labelling each box
with the gate that fired, is what turns the metric into a diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import ComponentKind, PageResult

# BGR. Chosen to stay distinguishable on both white scans and dark photographs.
KIND_COLORS: dict[ComponentKind, tuple[int, int, int]] = {
    ComponentKind.TEXT: (60, 200, 60),
    ComponentKind.DIACRITIC: (230, 200, 60),
    ComponentKind.RULE: (230, 120, 40),
    ComponentKind.BLOB: (200, 60, 220),
    ComponentKind.NOISE: (60, 60, 230),
}

KIND_THICKNESS: dict[ComponentKind, int] = {
    ComponentKind.TEXT: 2,
    ComponentKind.DIACRITIC: 2,
    ComponentKind.RULE: 1,
    ComponentKind.BLOB: 2,
    ComponentKind.NOISE: 1,
}


def draw_components(
    image: np.ndarray,
    result: PageResult,
    kinds: set[ComponentKind] | None = None,
    draw_legend: bool = True,
    label_reasons: bool = False,
) -> np.ndarray:
    """Return a copy of ``image`` with component boxes drawn.

    ``kinds`` restricts what is drawn — pass ``{ComponentKind.TEXT}`` to see what the
    classifier would actually receive, or leave it ``None`` to audit the routing.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for c in result.components:
        if kinds is not None and c.kind not in kinds:
            continue
        color = KIND_COLORS[c.kind]
        b = c.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x2, b.y2), color, KIND_THICKNESS[c.kind])
        if label_reasons and c.reason:
            cv2.putText(
                canvas, c.reason, (b.x, max(10, b.y - 2)),
                cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1, cv2.LINE_AA,
            )

    if draw_legend:
        canvas = _draw_legend(canvas, result)
    return canvas


def _draw_legend(canvas: np.ndarray, result: PageResult) -> np.ndarray:
    counts = result.kind_counts()
    lines = [f"{k.value}: {counts[k.value]}" for k in ComponentKind]
    lines.append(f"total: {len(result.components)}")
    median = result.meta.get("filter", {}).get("median_height")
    if median:
        lines.append(f"median h: {median}")
    lines.append(f"{result.timings_ms.get('total', 0):.1f} ms")

    pad, line_h = 8, 18
    box_w = 190
    box_h = pad * 2 + line_h * len(lines)

    # Semi-transparent panel so it never hides the document underneath it.
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), (40, 40, 40), 1)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - 5
        color = KIND_COLORS.get(_kind_for_line(i), (30, 30, 30))
        cv2.putText(canvas, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def _kind_for_line(index: int) -> ComponentKind | None:
    kinds = list(ComponentKind)
    return kinds[index] if index < len(kinds) else None


def binary_preview(binary: np.ndarray) -> np.ndarray:
    """Ink=255 binary rendered as black-on-white, which is how a human expects to read
    it — the raw array looks like a photo negative."""
    return cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)


def side_by_side(*panels: np.ndarray) -> np.ndarray:
    """Stack panels horizontally, padding to a common height.

    Useful for the one comparison that matters when debugging a bad page: source,
    binary, and routed components next to each other.
    """
    height = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        if p.shape[0] < height:
            pad = np.full((height - p.shape[0], p.shape[1], 3), 230, dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)
    return np.hstack(padded)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write via imencode so non-ASCII paths work on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"could not encode image for {path}")
    path.write_bytes(buf.tobytes())
