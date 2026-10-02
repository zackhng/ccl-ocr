"""Train the Phase 7 line script classifier.

    uv run python scripts/train_script_id.py --data data/lines --out models/script_id.pt

``--data`` holds one sub-directory per script (``latin``, ``han``, ``devanagari``,
``thai``, ``arabic``) as written by ``scripts/build_lines.py``. Classes are balanced by
sampling at most ``--per-script`` lines from each. Reports held-out accuracy overall and
per script, and the confusion counts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr.recog import script_id as sid  # noqa: E402
from ocr.recog import seq  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/lines")
    ap.add_argument("--out", default="models/script_id.pt")
    ap.add_argument("--per-script", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    items = []
    for k, script in enumerate(sid.SCRIPTS):
        crops = []
        for f in sorted((Path(args.data) / script).glob("shard_*.npz")):
            z = np.load(f, allow_pickle=True)
            for o, w in zip(z["offsets"], z["widths"]):
                crops.append(z["strip"][:, o:o + w])
        rng.shuffle(crops)
        items += [(c, k) for c in crops[: args.per_script]]
        print(f"  {script}: {min(len(crops), args.per_script)} lines", flush=True)
    rng.shuffle(items)
    n_val = int(0.05 * len(items))
    val, train = items[:n_val], items[n_val:]

    model = sid.ScriptCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    def run(data, training):
        model.train(training)
        preds, gold = [], []
        order = sorted(range(len(data)), key=lambda i: data[i][0].shape[1])
        chunks = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        if training:
            rng.shuffle(chunks)
        for ch in chunks:
            x, lengths = seq.batch([1.0 - data[i][0].astype(np.float32) / 255.0 for i in ch], dev)
            y = torch.tensor([data[i][1] for i in ch], device=dev)
            with torch.set_grad_enabled(training):
                logits = model(x, lengths)
                if training:
                    loss = loss_fn(logits, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            preds += logits.argmax(1).tolist()
            gold += y.tolist()
        return np.array(preds), np.array(gold)

    for epoch in range(args.epochs):
        run(train, True)
        p, g = run(val, False)
        print(f"  epoch {epoch + 1}: val acc {(p == g).mean():.2%}", flush=True)

    p, g = run(val, False)
    report = {
        "val_acc": float((p == g).mean()),
        "per_script": {s: float((p[g == k] == k).mean()) for k, s in enumerate(sid.SCRIPTS) if (g == k).any()},
        "confusions": {f"{sid.SCRIPTS[a]}->{sid.SCRIPTS[b]}": n
                       for (a, b), n in Counter(zip(g[p != g].tolist(), p[p != g].tolist())).most_common(10)},
    }
    sid.save(model, args.out, report)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
