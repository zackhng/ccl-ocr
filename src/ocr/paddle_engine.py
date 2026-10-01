"""PaddleOCR baseline, behind the same :class:`~ocr.engine.Engine` protocol as CCL.

Phase 6 exists to answer one question with numbers: is the CCL pipeline fast enough,
with enough headroom left for a classifier and reconstruction, to be worth building
instead of using an off-the-shelf detector + recogniser? That comparison is only
honest if both sides are driven by one runner over the same images, so Paddle is
wrapped here rather than benchmarked by a separate script.

Two modes, exposed as two engine names:

``paddle``
    Full detection + recognition — what we would ship instead of CCL + CNN.
``paddle-det``
    Detection only. The like-for-like comparison with today's CCL engine, which also
    stops at "where is the text". Recognition cost is the per-document difference
    between the two.

Choices that move the latency numbers, all recorded in :class:`PaddleConfig`:

- **PP-OCRv5 mobile** models. The server models are more accurate and several times
  slower; mobile is the latency-comparable choice, and covers Latin and Han in one model.
- **Document orientation, unwarping and text-line orientation are off.** The CCL
  pipeline does none of those yet, and each is an extra model pass.
- **oneDNN on, paddlepaddle pinned to 3.1.x.** On 3.3.x the detector crashes with oneDNN
  enabled, and running without it is ~5x slower — a latency win for CCL that would say
  nothing about the architecture.
- Paddle's own detector resizing defaults are kept (``limit_type=min``), i.e. what a
  user of the library gets. CCL caps the long side at 1600 px; Paddle does not cap.

Paddle is imported lazily, so nothing in :mod:`ocr` depends on it.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from .timing import StageTimer
from .types import BBox, Line, PageResult, Word

INSTALL_HINT = "PaddleOCR is not installed. Run: uv sync --extra baselines"


@dataclass(frozen=True, slots=True)
class PaddleConfig:
    mode: Literal["full", "det"] = "full"
    det_model: str = "PP-OCRv5_mobile_det"
    rec_model: str = "PP-OCRv5_mobile_rec"
    cpu_threads: int = 8
    enable_mkldnn: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bgr(image: np.ndarray) -> np.ndarray:
    """Paddle reads ndarrays as 3-channel BGR, the same convention as ``cv2``."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def page_from_paddle(
    raw: Mapping[str, Any],
    width: int,
    height: int,
    timings_ms: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
) -> PageResult:
    """Convert one Paddle result into a :class:`PageResult`.

    Pure function, so the conversion is testable without Paddle installed. A full-
    pipeline result carries ``rec_polys``/``rec_texts``; a detection-only result
    carries ``dt_polys``. Each polygon becomes one :class:`Line` holding one
    :class:`Word` whose text is the whole recognised line — Paddle does not segment
    words, and inventing a split here would be guessing.

    Boxes are clipped to the image: Paddle's detector expands polygons by its unclip
    ratio and can return points a few pixels outside the frame.
    """
    # Paddle returns lists on some paths and ndarrays on others; `x or []` raises on an
    # ndarray, so test for None explicitly.
    def _seq(key: str) -> list:
        v = raw.get(key)
        return [] if v is None else list(v)

    recognised = "rec_polys" in raw
    if recognised:
        polys, texts = _seq("rec_polys"), _seq("rec_texts")
        if len(texts) != len(polys):
            raise ValueError(f"paddle returned {len(polys)} polygons but {len(texts)} texts")
    else:
        polys, texts = _seq("dt_polys"), []

    lines: list[Line] = []
    for i, poly in enumerate(polys):
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        pts[:, 0] = pts[:, 0].clip(0, width)
        pts[:, 1] = pts[:, 1].clip(0, height)
        bbox = BBox.from_points(pts)
        text = texts[i] if recognised else ""
        lines.append(Line(bbox=bbox, words=[Word(bbox=bbox, text=text)]))

    # Reading order, roughly: top-to-bottom then left-to-right. Paddle already sorts,
    # but the metrics must not depend on that.
    lines.sort(key=lambda ln: (ln.bbox.y, ln.bbox.x))

    timings = dict(timings_ms or {})
    if "total" not in timings:
        timings["total"] = round(sum(timings.values()), 3)
    return PageResult(width=width, height=height, lines=lines, timings_ms=timings, meta=meta or {})


class PaddleEngine:
    """PaddleOCR 3.x behind the :class:`~ocr.engine.Engine` protocol."""

    def __init__(self, config: PaddleConfig | None = None) -> None:
        self.config = config or PaddleConfig()
        self.name = "paddle" if self.config.mode == "full" else "paddle-det"

        # The connectivity check pings model hosts on every construction; models are
        # cached after the first download, and the check adds seconds of noise.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import paddleocr
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(INSTALL_HINT) from exc

        cfg = self.config
        common = {"device": "cpu", "cpu_threads": cfg.cpu_threads, "enable_mkldnn": cfg.enable_mkldnn}
        if cfg.mode == "full":
            self._predictor = paddleocr.PaddleOCR(
                text_detection_model_name=cfg.det_model,
                text_recognition_model_name=cfg.rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                **common,
            )
            self._stage = "paddle_ocr"
        else:
            self._predictor = paddleocr.TextDetection(model_name=cfg.det_model, **common)
            self._stage = "paddle_det"
        self.versions = {"paddleocr": getattr(paddleocr, "__version__", "?")}
        try:
            import paddle

            self.versions["paddlepaddle"] = paddle.__version__
        except ImportError:  # pragma: no cover
            pass

    def run(self, image: np.ndarray) -> PageResult:
        timer = StageTimer()
        with timer.stage(self._stage):
            raw = next(iter(self._predictor.predict(_bgr(image))))
        h, w = image.shape[:2]
        return page_from_paddle(raw, w, h, timer.as_dict(), meta={"versions": self.versions})
