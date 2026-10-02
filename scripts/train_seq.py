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

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog import seq  # noqa: E402
from ocrbench.metrics import levenshtein  # noqa: E402


def load(dirs: list[str]):
    crops, labels, texts = [], [], []
    for d in dirs:
        for f in sorted(Path(d).glob("shard_*.npz")):
            z = np.load(f, allow_pickle=True)
            strip, offs, widths = z["strip"], z["offsets"], z["widths"]
            for o, w, lab, txt in zip(offs, widths, z["labels"], z["texts"]):
                crops.append(strip[:, o:o + w])
                labels.append(str(lab))
                texts.append(str(txt))
    return crops, labels, texts


def to_input(crop: np.ndarray) -> np.ndarray:
    return 1.0 - crop.astype(np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--corpus", help="charset source (default data/corpora/script_<script>.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    crops, labels, texts = load(args.data)
    corpus = Path(args.corpus or f"data/corpora/script_{args.script}.txt")
    corpus_lines = corpus.read_text(encoding="utf-8").splitlines() if corpus.exists() else []
    charset = seq.build_charset(corpus_lines + labels)
    index = {c: i for i, c in enumerate(charset)}

    items = [(c, [index[ch] for ch in lab], txt) for c, lab, txt in zip(crops, labels, texts)
             if lab and all(ch in index for ch in lab)]
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
                x, lengths = seq.batch([to_input(c) for c, _, _ in b], dev)
                preds = seq.greedy_decode(model(x), lengths, charset)
                for p, (_, _, txt) in zip(preds, b):
                    p = seq.visual_to_logical(p, args.script)
                    edits += levenshtein(txt, p)
                    total += len(txt)
        model.train()
        return edits / max(1, total)

    for epoch in range(args.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for b in batches(train, True):
            x, lengths = seq.batch([to_input(c) for c, _, _ in b], dev)
            targets = torch.tensor([i for _, lab, _ in b for i in lab], dtype=torch.long)
            tlens = torch.tensor([len(lab) for _, lab, _ in b], dtype=torch.long)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logp = model(x)
            loss = ctc(logp.float().transpose(0, 1), targets, lengths, tlens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
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
