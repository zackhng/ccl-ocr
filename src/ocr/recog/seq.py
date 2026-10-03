"""Line-level sequence recogniser (CRNN + CTC) for scripts CCL cannot segment (Phase 7).

The Latin path classifies one glyph cluster at a time. That does not hold for the
other scripts in scope: a Devanagari conjunct is several characters in one shape,
Arabic letters join within a word, Thai stacks vowel and tone marks over consonants,
and Han documents are dense enough that touching characters are the norm. Following
the original architecture, these lines go to a recogniser that reads the whole line
crop and emits characters with CTC — no character segmentation needed.

One model per script, each with its own character set (built from that script's
corpus and stored in the checkpoint). Every charset also carries ASCII digits, Latin
letters and punctuation, because real documents mix them into every script (amounts,
codes, English labels).

**Arabic is right-to-left.** The model reads pixels left to right, so it is trained on
the *visual* order (the logical string reversed) and its output is reversed back.
Embedded left-to-right runs (digits, Latin) are therefore reversed twice in a pure
reversal; :func:`visual_to_logical` re-reverses digit and Latin runs, a simple bidi
approximation sufficient for amounts and codes.

PyTorch only; CUDA when available.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

LINE_HEIGHT = 40
"""Crops are resized to this height, width scaled to keep the aspect ratio."""
MAX_WIDTH = 1024

BLANK = "<blank>"
ALWAYS = string.ascii_letters + string.digits + " .,:;/-()%&*#@+'\"!?$"
RTL_SCRIPTS = frozenset({"arabic"})


def build_charset(lines: list[str], coverage: float = 0.9995, max_size: int = 8000) -> list[str]:
    """Characters covering ``coverage`` of the corpus, plus :data:`ALWAYS`; index 0 is
    the CTC blank."""
    from collections import Counter

    counts = Counter(ch for line in lines for ch in line)
    total = sum(counts.values())
    chars, acc = [], 0
    for ch, n in counts.most_common(max_size):
        chars.append(ch)
        acc += n
        if acc / total >= coverage:
            break
    return [BLANK] + sorted(set(chars) | set(ALWAYS))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 256) -> None:
        super().__init__()

        def conv(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers

        self.cnn = nn.Sequential(
            *conv(1, 64, (2, 2)),        # 20 x W/2
            *conv(64, 128, (2, 2)),      # 10 x W/4
            *conv(128, 256, None),
            *conv(256, 256, (2, 1)),     # 5 x W/4
            *conv(256, 384, None),
            *conv(384, 384, (5, 1)),     # 1 x W/4
        )
        self.rnn = nn.LSTM(384, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [B,1,40,W] ink-high -> log-probs [B, T=W/4, C]."""
        f = self.cnn(x).squeeze(2).transpose(1, 2)  # [B, T, 384]
        y, _ = self.rnn(f)
        return torch.log_softmax(self.head(y), dim=-1)


def line_tensor(gray_crop: np.ndarray) -> np.ndarray:
    """A grey line crop (paper white) -> float32 [40, W] ink-high, W <= MAX_WIDTH."""
    h, w = gray_crop.shape[:2]
    new_w = max(8, min(MAX_WIDTH, int(round(w * LINE_HEIGHT / max(1, h)))))
    img = cv2.resize(gray_crop, (new_w, LINE_HEIGHT), interpolation=cv2.INTER_AREA if h > LINE_HEIGHT else cv2.INTER_CUBIC)
    return 1.0 - img.astype(np.float32) / 255.0


def batch(images: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [40, W] arrays to one tensor; also returns output lengths (W/4)."""
    w = max(i.shape[1] for i in images)
    w = (w + 3) // 4 * 4
    out = np.zeros((len(images), 1, LINE_HEIGHT, w), np.float32)
    for k, im in enumerate(images):
        out[k, 0, :, : im.shape[1]] = im
    lengths = torch.tensor([max(1, i.shape[1] // 4) for i in images], dtype=torch.long)
    return torch.as_tensor(out, device=device), lengths


def greedy_decode(logp: torch.Tensor, lengths: torch.Tensor, charset: list[str]) -> list[str]:
    best = logp.argmax(-1).cpu().numpy()
    out = []
    for row, n in zip(best, lengths.tolist()):
        chars, prev = [], 0
        for k in row[:n]:
            if k != prev and k != 0:
                chars.append(charset[k])
            prev = k
        out.append("".join(chars))
    return out


_LTR_RUN = re.compile(r"[0-9A-Za-z.,:/%\-]+")


def logical_to_visual(text: str, script: str) -> str:
    """Training label in pixel order: RTL text reversed, its LTR runs kept LTR."""
    if script not in RTL_SCRIPTS:
        return text
    rev = text[::-1]
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], rev)


def visual_to_logical(text: str, script: str) -> str:
    """Inverse of :func:`logical_to_visual` (the mapping is its own inverse)."""
    return logical_to_visual(text, script)


class SequenceRecognizer:
    """A trained CRNN for one script, ready to read line crops."""

    def __init__(self, path: str | Path, device="cpu") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.script = ckpt["script"]
        self.charset = ckpt["charset"]
        self.model = CRNN(len(self.charset), ckpt.get("hidden", 256)).to(device).eval()
        self.model.load_state_dict(ckpt["state"])
        self.device = device

    @torch.no_grad()
    def read(self, crops: list[np.ndarray]) -> list[str]:
        return [t for t, _ in self.read_scored(crops)]

    @torch.no_grad()
    def read_scored(self, crops: list[np.ndarray], max_batch: int = 64) -> list[tuple[str, float]]:
        """(text, confidence) per crop. Confidence is the mean top probability over the
        frames that emit a character — low when any glyph is ambiguous; 0 for empty."""
        out: list[tuple[str, float]] = []
        # Batches of similar width, so one long line does not pad every short one.
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
        res: dict[int, tuple[str, float]] = {}
        for s in range(0, len(order), max_batch):
            idx = order[s:s + max_batch]
            x, lengths = batch([line_tensor(crops[i]) for i in idx], self.device)
            logp = self.model(x).float()
            texts = greedy_decode(logp, lengths, self.charset)
            top, arg = logp.max(-1)
            for k, i in enumerate(idx):
                n = int(lengths[k])
                emit = arg[k, :n] != 0
                conf = float(top[k, :n][emit].exp().mean()) if bool(emit.any()) else 0.0
                res[i] = (visual_to_logical(texts[k], self.script), conf)
        out = [res[i] for i in range(len(crops))]
        return out


def save(model: CRNN, path: str | Path, script: str, charset: list[str], meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "script": script, "charset": charset,
                "hidden": model.rnn.hidden_size, "meta": meta}, path)
