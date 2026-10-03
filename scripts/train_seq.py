"""Train a Phase 7 sequence recogniser (CRNN + CTC) for one script.

    uv run python scripts/train_seq.py --script devanagari --data data/lines/devanagari \
        --out models/seq_devanagari.pt

Lines come from ``scripts/build_lines.py`` (and, as real line data is added, from real
documents in the same format). The charset is built from the script's corpus plus the
labels (``ocr.recog.seq.build_charset``) and stored in the checkpoint. 3% of lines are
held out; the report gives held-out CER in *logical* order (after RTL reversal), the
number that is comparable with the document benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog import seq  # noqa: E402
from ocrbench.metrics import CASE_FOLDED_SOURCES, levenshtein  # noqa: E402


def load(dirs: list[str]):
    """Crops, labels, texts, and whether each line's transcript is case-folded."""
    crops, labels, texts, folded = [], [], [], []
    for d in dirs:
        for f in sorted(Path(d).glob("shard_*.npz")):
            z = np.load(f, allow_pickle=True)
            strip, offs, widths = z["strip"], z["offsets"], z["widths"]
            profiles = z["profiles"] if "profiles" in z else [""] * len(offs)
            for o, w, lab, txt, prof in zip(offs, widths, z["labels"], z["texts"], profiles):
                crops.append(strip[:, o:o + w])
                labels.append(str(lab))
                texts.append(str(txt))
                folded.append(str(prof) in CASE_FOLDED_SOURCES)
    return crops, labels, texts, folded


def fold_case(logp: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    """Log-probs with each lowercase letter's mass added to its uppercase partner: CTC
    against an upper-cased transcript then accepts either case."""
    out = logp.clone()
    out[..., upper] = torch.logaddexp(logp[..., upper], logp[..., lower])
    return out


def to_input(crop: np.ndarray) -> np.ndarray:
    return 1.0 - crop.astype(np.float32) / 255.0


def augment(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Capture-style jitter for real line crops: width stretch, slight rotation and
    vertical shift (CCL's line boxes are not GT boxes), blur, contrast and noise."""
    h, w = crop.shape
    img = crop
    sx = rng.uniform(0.8, 1.2)
    nw = int(np.clip(round(w * sx), 8, seq.MAX_WIDTH))
    img = cv2.resize(img, (nw, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        m = cv2.getRotationMatrix2D((nw / 2, h / 2), rng.uniform(-1.5, 1.5), 1.0)
        m[1, 2] += rng.uniform(-3, 3)
        img = cv2.warpAffine(img, m, (nw, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    f = img.astype(np.float32)
    if rng.random() < 0.5:
        lo, hi = rng.uniform(0, 60), rng.uniform(190, 255)
        f = lo + f * (hi - lo) / 255.0
    if rng.random() < 0.3:
        f += rng.normal(0, rng.uniform(2, 10), f.shape)
    return 1.0 - np.clip(f, 0, 255) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--corpus", help="charset source (default data/corpora/script_<script>.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true", help="capture-style jitter (real line data)")
    ap.add_argument("--glyph-charset", action="store_true",
                    help="also include the Latin glyph classes (accented Latin, Vietnamese, currencies)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    crops, labels, texts, folded = load(args.data)
    corpus = Path(args.corpus or f"data/corpora/script_{args.script}.txt")
    corpus_lines = corpus.read_text(encoding="utf-8").splitlines() if corpus.exists() else []
    extra = []
    if args.glyph_charset:
        from ocr.recog.charset import GLYPH_CLASSES
        extra = ["".join(c for c in GLYPH_CLASSES if len(c) == 1)]
    charset = seq.build_charset(corpus_lines + labels)
    if extra:  # whole, not subject to build_charset's coverage cut
        charset = [seq.BLANK] + sorted(set(charset[1:]) | set(extra[0]))
    index = {c: i for i, c in enumerate(charset)}

    items = [(c, [index[ch] for ch in lab], txt, f) for c, lab, txt, f in zip(crops, labels, texts, folded)
             if lab and all(ch in index for ch in lab)]
    pairs = [(index[c.upper()], index[c]) for c in charset[1:]
             if len(c.upper()) == 1 and c.upper() != c and c.upper() in index]
    upper_idx = torch.tensor([u for u, _ in pairs], device=dev)
    lower_idx = torch.tensor([lo for _, lo in pairs], device=dev)
    rng.shuffle(items)
    n_val = max(200, int(0.03 * len(items)))
    val, train = items[:n_val], items[n_val:]
    print(f"  {args.script}: {len(train)} train / {len(val)} val lines, charset {len(charset)}, device {dev}", flush=True)

    model = seq.CRNN(len(charset)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def batches(data, shuffle):
        # Bucket by width so padding stays small.
        order = sorted(range(len(data)), key=lambda i: data[i][0].shape[1])
        chunks = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        if shuffle:
            rng.shuffle(chunks)
        for ch in chunks:
            yield [data[i] for i in ch]

    def evaluate() -> float:
        model.eval()
        edits = total = 0
        with torch.no_grad():
            for b in batches(val, False):
                x, lengths = seq.batch([to_input(c) for c, _, _, _ in b], dev)
                preds = seq.greedy_decode(model(x), lengths, charset)
                for p, (_, _, txt, f) in zip(preds, b):
                    p = seq.visual_to_logical(p, args.script)
                    if f:
                        p = p.upper()
                    edits += levenshtein(txt, p)
                    total += len(txt)
        model.train()
        return edits / max(1, total)

    for epoch in range(args.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for b in batches(train, True):
            prep = (lambda c: augment(c, nrng)) if args.augment else to_input
            x, lengths = seq.batch([prep(c) for c, _, _, _ in b], dev)
            targets = torch.tensor([i for _, lab, _, _ in b for i in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for _, lab, _, _ in b], dtype=torch.long)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logp = model(x)
            logp = logp.float()
            fmask = torch.tensor([f for _, _, _, f in b], device=dev)
            if len(pairs) and bool(fmask.any()):
                logp = torch.where(fmask[:, None, None], fold_case(logp, upper_idx, lower_idx), logp)
            loss = ctc(logp.transpose(0, 1), targets, lengths, tlens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        print(f"  epoch {epoch + 1}: ctc {tot / max(1, n):.3f}  val CER {evaluate():.2%}  ({time.time() - t0:.0f}s)", flush=True)

    cer = evaluate()
    report = {"script": args.script, "train_lines": len(train), "val_lines": len(val),
              "charset": len(charset), "val_cer": cer, "epochs": args.epochs, "data": args.data}
    seq.save(model, args.out, args.script, charset, report)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
