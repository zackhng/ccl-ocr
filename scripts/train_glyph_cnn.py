"""Train the Phase 3 glyph CNN on GPU.

    uv run python scripts/train_glyph_cnn.py --data data/glyphs/real data/glyphs/corpus \
        --weights 1.0 0.5 --out models/glyph_cnn.pt

Real aligned crops are the primary data; corpus pages supplement charset gaps (``--weights``
sets each source's sampling share relative to its size). Validation holds out whole
*documents* (by page seed / sample), never crops from a training document. After
training, a temperature is fitted on the validation set and stored in the checkpoint.

Reports top-1 accuracy overall and on character classes only, by source; the top
confusion pairs; and expected calibration error before and after temperature scaling.
Writes ``<out>.json`` beside the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog import glyph_cnn  # noqa: E402
from ocr.recog.charset import GLYPH_CLASSES, STRUCTURAL  # noqa: E402

STRUCT_IDS = {GLYPH_CLASSES.index(c) for c in STRUCTURAL}


def load_dir(d: Path):
    crops, geom, labels, doc = [], [], [], []
    for k, f in enumerate(sorted(d.glob("shard_*.npz"))):
        z = np.load(f)
        crops.append(z["crops"])
        geom.append(z["geom"].astype(np.float32))
        labels.append(z["labels"].astype(np.int64))
        # Document id: page seed where present, else the shard (real stores write seed 0).
        seed = z["seed"]
        doc.append(seed if seed.any() else np.full(len(seed), k, np.int64))
    if not crops:
        raise SystemExit(f"no shards in {d}")
    return np.concatenate(crops), np.concatenate(geom), np.concatenate(labels), np.concatenate(doc)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Light, on-GPU: shift/scale jitter, contrast, noise. The crops are already real
    capture conditions; this only discourages memorising exact pixel placement."""
    n = x.shape[0]
    theta = torch.zeros(n, 2, 3, device=x.device)
    s = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.16
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) - 0.5) * 0.12
    grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    c = 0.7 + torch.rand(n, 1, 1, 1, device=x.device) * 0.6
    x = (x * c + torch.randn_like(x) * 0.03).clamp_(0, 1)
    return x


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", default="models/glyph_cnn.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    weights = args.weights or [1.0] * len(args.data)

    tr_parts, va_parts, sources = [], [], []
    for si, (d, w) in enumerate(zip(args.data, weights)):
        c, g, y, doc = load_dir(Path(d))
        docs = np.unique(doc)
        val_docs = set(rng.choice(docs, max(1, int(len(docs) * args.val_frac)), replace=False).tolist())
        is_val = np.isin(doc, list(val_docs))
        tr_idx = np.flatnonzero(~is_val)
        if w != 1.0:  # resample this source to weight x its size
            tr_idx = rng.choice(tr_idx, int(len(tr_idx) * w), replace=w > 1.0)
        tr_parts.append((c[tr_idx], g[tr_idx], y[tr_idx]))
        va_parts.append((c[is_val], g[is_val], y[is_val], np.full(int(is_val.sum()), si)))
        sources.append(Path(d).name)
        print(f"  {d}: {len(y)} crops, {len(docs)} docs -> train {len(tr_idx)}, val {int(is_val.sum())}", flush=True)

    Xtr = np.concatenate([p[0] for p in tr_parts]); Gtr = np.concatenate([p[1] for p in tr_parts])
    Ytr = np.concatenate([p[2] for p in tr_parts])
    Xva = np.concatenate([p[0] for p in va_parts]); Gva = np.concatenate([p[1] for p in va_parts])
    Yva = np.concatenate([p[2] for p in va_parts]); Sva = np.concatenate([p[3] for p in va_parts])

    xtr, gtr = glyph_cnn.prepare(Xtr, Gtr, dev)
    ytr = torch.as_tensor(Ytr, device=dev)
    xva, gva = glyph_cnn.prepare(Xva, Gva, dev)
    model = glyph_cnn.GlyphCNN(width=args.width).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params / 1e6:.2f}M params, {len(Ytr)} train crops, device {dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Ytr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def predict() -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Yva), 4096):
                out.append(model(xva[i:i + 4096], gva[i:i + 4096]).float().cpu())
        model.train()
        return torch.cat(out).numpy()

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(Ytr), device=dev)
        total = 0.0
        for i in range(0, len(Ytr), args.batch):
            idx = perm[i:i + args.batch]
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = loss_fn(model(augment(xtr[idx]), gtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        acc = (predict().argmax(1) == Yva).mean()
        print(f"  epoch {epoch + 1}: loss {total / len(Ytr):.3f}  val acc {acc:.2%}  ({time.time() - t0:.0f}s)", flush=True)

    # Temperature: minimise held-out NLL over a 1-D grid (robust, no extra optimiser).
    logits = predict()
    lt = torch.as_tensor(logits)
    yv = torch.as_tensor(Yva)
    best_t = min(np.linspace(0.5, 3.0, 51), key=lambda t: float(nn.functional.cross_entropy(lt / t, yv)))
    model.temperature.fill_(float(best_t))
    p_raw = torch.softmax(lt, 1).numpy()
    p_cal = torch.softmax(lt / best_t, 1).numpy()

    pred = logits.argmax(1)
    is_char = ~np.isin(Yva, list(STRUCT_IDS))
    report = {
        "params": n_params, "train_crops": int(len(Ytr)), "val_crops": int(len(Yva)),
        "val_acc": float((pred == Yva).mean()),
        "val_acc_chars": float((pred[is_char] == Yva[is_char]).mean()),
        "val_acc_by_source": {s: float((pred[Sva == i] == Yva[Sva == i]).mean()) for i, s in enumerate(sources)},
        "temperature": float(best_t), "ece_raw": ece(p_raw, Yva), "ece_calibrated": ece(p_cal, Yva),
        "top_confusions": [
            {"true": GLYPH_CLASSES[a], "pred": GLYPH_CLASSES[b], "n": n}
            for (a, b), n in Counter(zip(Yva[pred != Yva].tolist(), pred[pred != Yva].tolist())).most_common(25)
        ],
        "data": args.data, "weights": weights, "epochs": args.epochs, "width": args.width,
        "charset": glyph_cnn.CHARSET_HASH,
    }
    glyph_cnn.save(model, args.out, {"width": args.width, **{k: v for k, v in report.items() if k != "top_confusions"}})
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "top_confusions"}, ensure_ascii=False, indent=1))
    print("top confusions:", ", ".join(f"{c['true']}->{c['pred']} {c['n']}" for c in report["top_confusions"][:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
