"""Pretrain the character LM on public text, then fine-tune on financial documents.

    uv run python scripts/train_char_lm.py --stage pretrain --text data/corpora/public.txt \
        --out models/char_lm_pretrain.pt
    uv run python scripts/train_char_lm.py --stage finetune --init models/char_lm_pretrain.pt \
        --text data/corpora/financial.txt data/corpora/real_train.txt --out models/char_lm.pt

Lines are trained as independent sequences (BOS ... EOS), cropped to ``--seq`` tokens.
5% of lines — the *last* 5%, so a document's lines stay together — are held out, and
bits per character on held-out **financial** text is reported for both stages: the
number that says whether fine-tuning helped where it matters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog import char_lm  # noqa: E402
from ocr.recog.charset import LM_INDEX, PAD, lm_encode  # noqa: E402

PAD_ID = LM_INDEX[PAD]


def read_lines(paths: list[str]) -> list[list[int]]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            ids = lm_encode(line)
            if len(ids) > 3:
                out.append(ids)
    return out


def batches(seqs: list[list[int]], batch: int, seq_len: int, rng: random.Random):
    order = list(range(len(seqs)))
    rng.shuffle(order)
    for i in range(0, len(order), batch):
        chunk = []
        for k in order[i:i + batch]:
            s = seqs[k]
            if len(s) > seq_len + 1:
                start = rng.randrange(len(s) - seq_len)
                s = s[start:start + seq_len + 1]
            chunk.append(s)
        width = max(map(len, chunk))
        t = torch.full((len(chunk), width), PAD_ID, dtype=torch.long)
        for r, s in enumerate(chunk):
            t[r, :len(s)] = torch.tensor(s)
        yield t


@torch.no_grad()
def bits_per_char(model, seqs, dev, seq_len) -> float:
    model.eval()
    total_nll, total_n = 0.0, 0
    for t in batches(seqs, 256, seq_len, random.Random(0)):
        t = t.to(dev)
        logits, _ = model(t[:, :-1])
        nll = nn.functional.cross_entropy(logits.transpose(1, 2), t[:, 1:], ignore_index=PAD_ID, reduction="sum")
        total_nll += float(nll)
        total_n += int((t[:, 1:] != PAD_ID).sum())
    model.train()
    return total_nll / max(1, total_n) / math.log(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--eval-financial", nargs="+", default=["data/corpora/financial.txt"])
    ap.add_argument("--init")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    seqs = read_lines(args.text)
    cut = int(len(seqs) * 0.95)
    train, held = seqs[:cut], seqs[cut:]
    fin = read_lines(args.eval_financial)
    fin_held = fin[int(len(fin) * 0.95):]
    if args.stage == "finetune":
        # Fine-tuning text *is* the financial text: evaluate on its held-out tail only.
        fin_held = held

    model = char_lm.load(args.init, dev).train() if args.init else char_lm.CharLM().to(dev)
    print(f"  {len(train)} train lines, {len(held)} held out; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    before = bits_per_char(model, fin_held, dev, args.seq) if args.init else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = max(1, int(args.epochs * len(train) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    step, t0 = 0, time.time()
    while step < steps:
        for t in batches(train, args.batch, args.seq, rng):
            t = t.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, _ = model(t[:, :-1])
                loss = nn.functional.cross_entropy(logits.transpose(1, 2).float(), t[:, 1:], ignore_index=PAD_ID)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"  step {step}/{steps}: loss {float(loss) / math.log(2):.3f} bits ({time.time() - t0:.0f}s)", flush=True)
            if step >= steps:
                break

    report = {
        "stage": args.stage, "train_lines": len(train), "steps": steps,
        "held_out_bpc": bits_per_char(model, held, dev, args.seq),
        "financial_bpc": bits_per_char(model, fin_held, dev, args.seq),
        "financial_bpc_before": before, "text": args.text, "init": args.init,
    }
    char_lm.save(model, args.out, {"embed": 128, "hidden": 512, "layers": 2, **report})
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
