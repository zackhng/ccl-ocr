"""Joint decoding of a line: glyph CNN evidence + character LM context + field formats.

This is the "use the neighbours" step agreed with the user. Each character's CNN
probabilities are not read in isolation: a whole line is decoded by beam search, and
every hypothesis is scored

    sum over clusters   log P_cnn(class | crop)                    (calibrated)
  + lambda * sum        log P_lm(char | every character before it)  (words only)
  + bonus               if a structured field satisfies its grammar (checksum included)

so in ``C ? E Q`` the middle can lean towards ``H`` when the crop is ambiguous, while a
confident crop still wins: the LM only re-weights, it never vetoes ("unlikely, never
impossible"). Decoding the line, not the word, lets context cross word boundaries; a
space is emitted between words.

**Fields the LM must not touch.** For each word, the CNN's own best reading decides its
kind (:func:`ocr.recog.formats.token_kind`, which also looks at the preceding label).
For IDs, amounts, dates and numbers the LM weight is 0 — the LM state still advances so
later words keep their context — and a hypothesis whose finished token satisfies the
field's grammar earns ``format_bonus``. A checksum-valid candidate is *preferred*, never
forced: if no candidate in the beam validates, visual evidence decides.

**Structural classes.** ``<NONTEXT>`` and ``<PART>`` emit nothing (junk; a fragment whose
glyph is read through its neighbour). ``<MULTI>`` — fused characters — is expanded by an
optional callback into column-cut pieces re-read by the CNN, each piece then decoded
like any cluster; without a callback, or when it cannot cut, it falls back to its best
single character at a penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from . import formats
from .charset import GLYPH_CLASSES, GLYPH_INDEX, LM_INDEX, MULTI, NONTEXT, PART, SPACE, STRUCTURAL

_NONTEXT, _MULTI, _PART = GLYPH_INDEX[NONTEXT], GLYPH_INDEX[MULTI], GLYPH_INDEX[PART]
_SPACE = LM_INDEX[SPACE]
_STRUCT_IDS = [GLYPH_INDEX[c] for c in STRUCTURAL]


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    lm_weight: float = 0.5
    """lambda. 0 = pure CNN argmax per character (the ablation baseline)."""
    beam: int = 8
    top_k: int = 6
    """CNN candidates considered per cluster."""
    format_bonus: float = 4.0
    """Log-score bonus for a structured token that satisfies its grammar."""
    multi_penalty: float = 2.0
    """Charged when a <MULTI> cluster falls back to a single character."""


Expander = Callable[[int, int], "list[torch.Tensor] | None"]
"""(word index, cluster index) -> log-prob rows [C] for the pieces of a <MULTI> cluster,
left to right, or None if it cannot be cut."""


@dataclass(slots=True)
class _Hyp:
    score: float
    text: str
    token: str
    """Characters of the current word (for the format check)."""
    state: int
    """Row of the LM state/log-prob tensors holding this hypothesis's context."""


def _best_char(row: torch.Tensor) -> str:
    """Best emitting (non-structural) character of a cluster."""
    masked = row.clone()
    masked[_STRUCT_IDS] = float("-inf")
    return GLYPH_CLASSES[int(masked.argmax())]


def _greedy(row: torch.Tensor) -> str:
    i = int(row.argmax())
    return "" if i in _STRUCT_IDS else GLYPH_CLASSES[i]


class _LMState:
    """LM context for every live hypothesis, as one batched tensor pair."""

    def __init__(self, lm) -> None:
        self.lm = lm
        device = next(lm.parameters()).device
        self.logp, self.state = lm.start(1, device)  # [S, V], [L, S, H]

    def score(self, state_row: int, token: int) -> float:
        return float(self.logp[state_row, token])

    def score_many(self, rows: list[int], token: int) -> list[float]:
        return self.logp[rows, token].tolist()

    def advance(self, parents: list[int], tokens: list[int | None]) -> list[int]:
        """New rows: hypothesis i continues from row ``parents[i]``, stepping by
        ``tokens[i]`` if it is not None, else keeping the parent's context."""
        idx = torch.tensor(parents, device=self.state.device)
        logp, state = self.logp[idx], self.state[:, idx]
        step = [i for i, t in enumerate(tokens) if t is not None]
        if step:
            s_idx = torch.tensor(step, device=self.state.device)
            toks = torch.tensor([tokens[i] for i in step], device=self.state.device)
            lp, st = self.lm.step(toks, state[:, s_idx])
            logp = logp.clone()
            state = state.clone()
            logp[s_idx] = lp
            state[:, s_idx] = st
        self.logp, self.state = logp, state
        return list(range(len(parents)))


def decode_line(words: list[torch.Tensor], lm, cfg: DecodeConfig,
                expand: Expander | None = None) -> list[str]:
    """Decode one line. ``words``: per word, CNN log-probs [n_clusters, C], x order.

    ``lm`` is a :class:`~ocr.recog.char_lm.CharLM`, or ``None`` for the CNN-only
    baseline. Returns one string per word ('' for a word read as all junk).
    """
    if not words:
        return []
    kinds, prev = [], ""
    for w in words:
        best = "".join(_greedy(r) for r in w)
        kinds.append(formats.token_kind(best, prev))
        prev = best

    use_lm = lm is not None and cfg.lm_weight > 0
    ctx = _LMState(lm) if use_lm else None
    beams = [_Hyp(0.0, "", "", 0)]

    for wi, (w, kind) in enumerate(zip(words, kinds)):
        lam = cfg.lm_weight if (use_lm and formats.uses_language_model(kind)) else 0.0
        if wi > 0:
            if ctx is not None:
                if lam > 0:
                    for h, s in zip(beams, ctx.score_many([h.state for h in beams], _SPACE)):
                        h.score += lam * s
                rows = ctx.advance([h.state for h in beams], [_SPACE] * len(beams))
                for h, r in zip(beams, rows):
                    h.state = r
            for h in beams:
                h.text += "\x1f"  # word separator, split on at the end
                h.token = ""

        for ci in range(w.shape[0]):
            pieces = [w[ci]]
            if int(w[ci].argmax()) == _MULTI and expand is not None:
                cut = expand(wi, ci)
                if cut:
                    pieces = cut
            for row in pieces:
                top = torch.topk(row, min(cfg.top_k, row.shape[0]))
                cands: list[tuple[float, _Hyp, str]] = []
                for h in beams:
                    for lp, cls in zip(top.values.tolist(), top.indices.tolist()):
                        if cls in (_NONTEXT, _PART):
                            cands.append((h.score + lp, h, ""))
                            continue
                        ch = _best_char(row) if cls == _MULTI else GLYPH_CLASSES[cls]
                        s = h.score + lp - (cfg.multi_penalty if cls == _MULTI else 0.0)
                        if lam > 0:
                            tok = LM_INDEX.get(ch)
                            if tok is not None:
                                s += lam * ctx.score(h.state, tok)
                        cands.append((s, h, ch))
                cands.sort(key=lambda c: -c[0])
                chosen: list[tuple[float, _Hyp, str]] = []
                seen: set[str] = set()
                for s, h, ch in cands:
                    key = h.text + ch
                    if key in seen:
                        continue  # the same string by another path: keep the best
                    seen.add(key)
                    chosen.append((s, h, ch))
                    if len(chosen) >= cfg.beam:
                        break
                parents = [h.state for _, h, _ in chosen]
                if ctx is not None:
                    tokens = [LM_INDEX.get(ch) if ch else None for _, _, ch in chosen]
                    rows = ctx.advance(parents, tokens)
                else:
                    rows = parents
                beams = [_Hyp(s, h.text + ch, h.token + ch, r) for (s, h, ch), r in zip(chosen, rows)]

        if kind != "word" and cfg.format_bonus:
            for h in beams:
                if h.token and formats.satisfies(h.token, kind):
                    h.score += cfg.format_bonus

    best = max(beams, key=lambda h: h.score)
    return best.text.split("\x1f")
