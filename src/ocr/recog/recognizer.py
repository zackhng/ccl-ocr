"""The Phase 3 recogniser: fills ``Word.text`` on a :class:`PageResult` from CCL.

Pipeline per page:

1. Glyph clusters and their crops for every word of every line, emitted *and* rejected
   (:func:`ocr.recog.crops.extract`) — one batched CNN forward pass on the GPU.
2. Per line, joint decoding (:func:`ocr.recog.decode.decode_line`) with the character
   LM and field formats.
3. ``<MULTI>`` clusters are cut at low-ink columns (:func:`ocr.split.cut_columns`, the
   Phase 2b splitter's cutter) and the pieces re-read by the CNN, inside the decoder.
4. Rejected lines (junk-gated by Phase 5 geometry) are promoted back into
   ``PageResult.lines`` when the CNN reads them as mostly real characters — the
   recogniser, not geometry, has the final say on small print.

PyTorch only; runs on CUDA when available (the deployment target), CPU otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import SplitConfig
from ..group import reading_order
from ..split import cut_columns
from ..timing import StageTimer
from ..types import BBox, PageResult, Word
from . import char_lm, glyph_cnn
from .charset import GLYPH_INDEX, NONTEXT, PART
from .crops import Cluster, extract
from .decode import DecodeConfig, decode_line

_JUNK = (GLYPH_INDEX[NONTEXT], GLYPH_INDEX[PART])


@dataclass(frozen=True, slots=True)
class RecogConfig:
    cnn_path: str = "models/glyph_cnn.pt"
    lm_path: str | None = "models/char_lm.pt"
    """None = CNN-only decoding (the ablation baseline)."""
    decode: DecodeConfig = DecodeConfig()
    expand_multi: bool = True
    promote_rejected_below: float = 0.5
    """A rejected line is promoted when fewer than this share of its clusters read as
    <NONTEXT>/<PART>."""
    device: str = "cuda"
    script_id_path: str | None = None
    """Phase 7. When set, every line is classified by script first; non-Latin lines are
    read whole by that script's sequence recogniser instead of the glyph CNN."""
    seq_paths: dict | None = None
    """Phase 7: script name -> CRNN checkpoint (``ocr.recog.seq``)."""
    script_min_confidence: float = 0.6
    """Below this, a line stays on the Latin path (the default reader)."""


class Recognizer:
    def __init__(self, cfg: RecogConfig = RecogConfig()) -> None:
        self.cfg = cfg
        dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(dev)
        self.cnn = glyph_cnn.load(cfg.cnn_path, self.device)
        self.lm = char_lm.load(cfg.lm_path, self.device) if cfg.lm_path else None
        self._cut_cfg = SplitConfig(split_touching_columns=True)
        self.script_id = None
        self.readers: dict = {}
        if cfg.script_id_path:
            from .script_id import ScriptIdentifier
            from .seq import SequenceRecognizer

            self.script_id = ScriptIdentifier(cfg.script_id_path, self.device)
            for script, path in (cfg.seq_paths or {}).items():
                self.readers[script] = SequenceRecognizer(path, self.device)

    def _route(self, gray: np.ndarray, lines: list) -> dict[int, str]:
        """Line index -> script, for lines a sequence recogniser should read."""
        if self.script_id is None or not lines:
            return {}
        crops = [_line_crop(gray, ln.bbox) for ln in lines]
        routed = {}
        for i, (script, conf) in enumerate(self.script_id.classify(crops)):
            if script != "latin" and script in self.readers and conf >= self.cfg.script_min_confidence:
                routed[i] = script
        return routed

    @torch.no_grad()
    def _classify(self, crops: np.ndarray, geom: np.ndarray) -> torch.Tensor:
        if not len(crops):
            return torch.zeros((0, glyph_cnn.N_GLYPH_CLASSES))
        x, g = glyph_cnn.prepare(crops, geom, self.device)
        return self.cnn.log_probs(x, g).float().cpu()

    def _expander(self, gray: np.ndarray, page: PageResult, words_clusters, word_geoms):
        """Callback for the decoder: cut a <MULTI> cluster into pieces and re-read."""
        def expand(wi: int, ci: int):
            cl: Cluster = words_clusters[wi][ci]
            b = cl.bbox
            patch = gray[b.y:b.y2, b.x:b.x2]
            if patch.size == 0 or b.w < 4:
                return None
            _, ink = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            region = ink.copy()
            glyph_w = max(2.0, 0.6 * b.h)
            if not cut_columns(region, region > 0, glyph_w, self._cut_cfg):
                return None
            n, _, st, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            boxes = sorted((BBox(b.x + int(s[0]), b.y, int(s[2]), b.h) for s in st[1:n]
                            if s[3] >= 0.4 * b.h), key=lambda bb: bb.x)
            if len(boxes) < 2:
                return None
            pieces = [Cluster(bb, cl.members, cl.line_index, cl.word_index, cl.rejected) for bb in boxes]
            crops, geom, _ = extract(gray, page, pieces)
            return list(self._classify(crops, geom))
        return expand

    def prepare(self, image: np.ndarray, page: PageResult) -> "Prepared":
        """The expensive, config-independent half: clusters, crops, one CNN pass."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crops, geom, clusters = extract(gray, page)
        logp = self._classify(crops, geom)
        by_word: dict[tuple[int, int], list[int]] = {}
        for i, cl in enumerate(clusters):
            by_word.setdefault((cl.line_index, cl.word_index), []).append(i)
        for k in by_word:
            by_word[k].sort(key=lambda i: clusters[i].bbox.x)
        lines = page.lines + page.rejected_lines
        return Prepared(gray, clusters, logp, by_word, lines, self._route(gray, lines))

    def decode(self, prep: "Prepared", page: PageResult, cfg: DecodeConfig | None = None,
               lm=...) -> list:
        """Fill word texts for every line (emitted and rejected) under ``cfg``; return
        the rejected lines that read as text. Does not reorder ``page`` — repeatable,
        so a tuner can decode one prepared page under many configs."""
        cfg = cfg or self.cfg.decode
        lm = self.lm if lm is ... else lm
        logp, clusters = prep.logp, prep.clusters
        promoted = []
        # Phase 7: non-Latin lines are read whole by their script's recogniser. The
        # line's words collapse into one span: a CRNN reads a line, not CCL's words.
        by_script: dict[str, list[int]] = {}
        for li, script in prep.routes.items():
            by_script.setdefault(script, []).append(li)
        for script, idxs in by_script.items():
            texts = self.readers[script].read([_line_crop(prep.gray, prep.lines[i].bbox) for i in idxs])
            for li, text in zip(idxs, texts):
                line = prep.lines[li]
                line.words = [Word(bbox=line.bbox, components=[c for w in line.words for c in w.components],
                                   text=text)]
                line.script = script
                if line.rejected and text.strip():
                    promoted.append(line)
        for li, line in enumerate(prep.lines):
            if li in prep.routes:
                continue
            rows, wclusters, keys = [], [], []
            for wi in range(len(line.words)):
                idx = prep.by_word.get((li, wi), [])
                rows.append(logp[idx] if idx else torch.zeros((0, logp.shape[1])))
                wclusters.append([clusters[i] for i in idx])
                keys.append(idx)
            expand = self._expander(prep.gray, page, wclusters, None) if self.cfg.expand_multi else None
            for word, text in zip(line.words, decode_line(rows, lm, cfg, expand)):
                word.text = text
            if line.rejected:
                flat = [i for k in keys for i in k]
                junk = sum(1 for i in flat if int(logp[i].argmax()) in _JUNK)
                if flat and junk / len(flat) < self.cfg.promote_rejected_below and line.text.strip():
                    promoted.append(line)
        return promoted

    def recognise(self, image: np.ndarray, page: PageResult, timer: StageTimer | None = None) -> None:
        """Fill word texts in place; promote rejected lines that read as text."""
        timer = timer or StageTimer()
        with timer.stage("recognize"):
            prep = self.prepare(image, page)
            promoted = self.decode(prep, page)
            if promoted:
                for line in promoted:
                    page.rejected_lines.remove(line)
                    line.rejected = ""
                lines = page.lines + promoted
                page.lines = [lines[i] for i in reading_order([ln.bbox for ln in lines])]
            page.meta["recognize"] = {"clusters": len(prep.clusters), "promoted_lines": len(promoted)}


@dataclass(slots=True)
class Prepared:
    gray: np.ndarray
    clusters: list
    logp: torch.Tensor
    by_word: dict
    lines: list
    routes: dict = None
    """Phase 7: line index -> script for lines read by a sequence recogniser."""

    def __post_init__(self):
        if self.routes is None:
            self.routes = {}


def _line_crop(gray: np.ndarray, b: BBox) -> np.ndarray:
    m = max(2, b.h // 6)
    return gray[max(0, b.y - m): b.y2 + m, max(0, b.x - m): b.x2 + m]


def default_recognizer(models: str | Path = "models", lm: bool = True) -> Recognizer:
    """Latin path always; the Phase 7 script router and sequence recognisers are used
    automatically for whichever of their checkpoints exist in ``models``."""
    m = Path(models)
    seq_paths = {s: str(m / f"seq_{s}.pt") for s in ("han", "devanagari", "thai", "arabic")
                 if (m / f"seq_{s}.pt").exists()}
    script_id = m / "script_id.pt"
    return Recognizer(RecogConfig(
        cnn_path=str(m / "glyph_cnn.pt"),
        lm_path=str(m / "char_lm.pt") if lm else None,
        script_id_path=str(script_id) if script_id.exists() and seq_paths else None,
        seq_paths=seq_paths or None,
    ))
