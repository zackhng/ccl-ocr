"""Character language model: P(next character | previous characters).

A two-layer GRU over :data:`ocr.recog.charset.LM_TOKENS`. A recurrent model suits the
decoder: beam search extends each hypothesis one character at a time, and a GRU carries
that prefix as a fixed-size hidden state, so scoring the next character costs one step,
not a re-read of the prefix.

Pretrained on public text (Wikipedia en/id/vi), then fine-tuned on financial-document
text (receipts, forms, statements). The model only ever *weights* candidates; the
decoder never lets it override confident visual evidence or touch structured fields
(``ocr.recog.formats``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .charset import BOS, LM_INDEX, N_LM_TOKENS
from .glyph_cnn import CHARSET_HASH


class CharLM(nn.Module):
    def __init__(self, embed: int = 128, hidden: int = 512, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_LM_TOKENS, embed)
        self.rnn = nn.GRU(embed, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, N_LM_TOKENS)
        self.hidden_size, self.layers = hidden, layers

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens`` long [B, T] -> (logits [B, T, V], state [layers, B, H])."""
        y, state = self.rnn(self.embed(tokens), state)
        return self.out(y), state

    @torch.no_grad()
    def start(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(log-probs of the first character [B, V], state) after BOS."""
        bos = torch.full((batch, 1), LM_INDEX[BOS], dtype=torch.long, device=device)
        logits, state = self(bos)
        return torch.log_softmax(logits[:, -1], dim=-1), state

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance each hypothesis by one token: ``tokens`` long [B]."""
        logits, state = self(tokens[:, None], state)
        return torch.log_softmax(logits[:, -1], dim=-1), state


def save(model: CharLM, path: str | Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "charset": CHARSET_HASH, "meta": meta}, path)


def load(path: str | Path, device="cpu") -> CharLM:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("charset") != CHARSET_HASH:
        raise ValueError(f"{path}: trained on a different charset")
    m = ckpt["meta"]
    model = CharLM(embed=m.get("embed", 128), hidden=m.get("hidden", 512), layers=m.get("layers", 2)).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval()
