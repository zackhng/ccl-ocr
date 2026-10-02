"""The glyph classifier: a small CNN over a 32x32 crop plus line-relative geometry.

Inputs come from :func:`ocr.recog.crops.extract` — the same function at training and
inference. The crop alone is scale-free, so it cannot separate ``o``/``O``, ``c``/``C``,
``,``/``'``; the five geometry features (size and position relative to the line) are
concatenated before the head to restore that.

Outputs are logits over :data:`ocr.recog.charset.GLYPH_CLASSES` — the printable Latin
set plus ``<NONTEXT>``, ``<MULTI>``, ``<PART>``. A scalar temperature, fitted on
held-out documents after training, is stored with the weights: the decoder weighs these
probabilities against a language model, so they must mean the same thing on a clean
scan and a blurred photo.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .charset import GLYPH_CLASSES, N_GLYPH_CLASSES
from .crops import CROP, N_GEOMETRY

CHARSET_HASH = hashlib.sha1("\x00".join(GLYPH_CLASSES).encode("utf-8")).hexdigest()[:12]
"""Stored with every checkpoint: a model trained on another class list must not load."""


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class GlyphCNN(nn.Module):
    def __init__(self, n_classes: int = N_GLYPH_CLASSES, width: int = 48) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(1, width), nn.MaxPool2d(2),               # 16x16
            _block(width, 2 * width), nn.MaxPool2d(2),       # 8x8
            _block(2 * width, 4 * width),                    # 8x8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.geometry = nn.Sequential(nn.Linear(N_GEOMETRY, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(4 * width + 32, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, crops: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """``crops`` float [N,1,32,32] in 0..1 with ink high; ``geom`` float [N,5]."""
        return self.head(torch.cat([self.features(crops), self.geometry(geom)], dim=1))

    def log_probs(self, crops: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self(crops, geom) / self.temperature, dim=1)


def prepare(crops: np.ndarray, geom: np.ndarray, device: torch.device | str = "cpu"
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 crops (paper white) and raw geometry -> model inputs (ink high, clipped)."""
    x = torch.as_tensor(crops, device=device).float().div_(255.0).neg_().add_(1.0)
    x = x.reshape(-1, 1, CROP, CROP)
    g = torch.as_tensor(np.asarray(geom, dtype=np.float32), device=device).clamp_(-4.0, 8.0)
    return x, g


def save(model: GlyphCNN, path: str | Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "charset": CHARSET_HASH, "meta": meta}, path)


def load(path: str | Path, device: torch.device | str = "cpu") -> GlyphCNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("charset") != CHARSET_HASH:
        raise ValueError(f"{path}: trained on a different charset ({ckpt.get('charset')} != {CHARSET_HASH})")
    width = ckpt["meta"].get("width", 48)
    model = GlyphCNN(width=width).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval()
