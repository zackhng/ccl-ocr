"""Which script is this line in? Routes each line to its recogniser (Phase 7).

One document can mix scripts — an Indian bank statement in Hindi and English, a UAE
invoice in Arabic and English, a Chinese form with Latin codes — so the decision is
per *line*, never per page. The classifier only has to choose a reading path, not a
language: Latin (which covers English, Indonesian, Vietnamese, ...), Han, Devanagari,
Thai or Arabic.

A small CNN over the line crop resized to 40 px high, pooled across its width, so lines
of any length classify alike. PyTorch only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .seq import LINE_HEIGHT, batch, line_tensor

SCRIPTS = ("latin", "han", "devanagari", "thai", "arabic")


class ScriptCNN(nn.Module):
    def __init__(self, n: int = len(SCRIPTS)) -> None:
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]

        self.body = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128), *block(128, 128))
        self.head = nn.Linear(256, n)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``x`` [B,1,40,W]; pools only over each line's real width (not padding)."""
        f = self.body(x).mean(2)  # [B, 128, W/16]
        t = f.shape[-1]
        valid = (torch.arange(t, device=x.device)[None, :] < (lengths.to(x.device) // 4 + 1)[:, None]).float()
        mean = (f * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True).clamp(min=1)
        mx = f.masked_fill(valid[:, None] == 0, -1e4).amax(-1)
        return self.head(torch.cat([mean, mx], dim=1))


class ScriptIdentifier:
    def __init__(self, path: str | Path, device="cpu") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if tuple(ckpt["scripts"]) != SCRIPTS:
            raise ValueError(f"{path}: trained on scripts {ckpt['scripts']}")
        self.model = ScriptCNN().to(device).eval()
        self.model.load_state_dict(ckpt["state"])
        self.device = device

    @torch.no_grad()
    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        x, lengths = batch([line_tensor(c) for c in crops], self.device)
        p = torch.softmax(self.model(x, lengths), dim=1).cpu()
        conf, idx = p.max(1)
        return [(SCRIPTS[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]


def save(model: ScriptCNN, path: str | Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "scripts": SCRIPTS, "meta": meta,
                "line_height": LINE_HEIGHT}, path)
