"""Real-tail inputs for trunk --tail real, built exactly like nodus-elastic-inference training_prefix.

python h100/tail_prep.py --data /data [--prefix-run /prefix] --out DIR --steps S --jobs J
  /data = training-bubble data dir (train.jsonl, models/<train_size> tokenizer); /prefix = audited training_prefix run (config.json).
Writes DIR/tail_tokens.i32 and tail_labels.i32 ([S][mb*seq], pad 0 / -100, labels already shifted by one), tail_A.f32
([J][2R][D]: q rows then v rows, JobLoRA seeding), tail_lr.f32 ([J]) and tail_rows.json (row ids per microbatch).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prefix-run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--jobs", type=int, required=True)
    ap.add_argument("--train-size", default="4B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--top-layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=413)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--prefill-chunks", type=int, default=0, help="also write prefill_tokens.i32: the next K x mb rows of the same trace (after the training rows)")
    ap.add_argument("--lr-cycle", action="store_true", help="jobs past the prefix run's count reuse its learning rates cyclically (own seed-derived adapter init)")
    a = ap.parse_args()
    cfg = {}
    if a.prefix_run and (a.prefix_run / "config.json").exists():
        cfg = json.loads((a.prefix_run / "config.json").read_text())
        for k, v in (("seq_len", a.seq), ("microbatch", a.mb), ("rank", a.rank), ("top_layers", a.top_layers), ("train_size", a.train_size)):
            if k in cfg and cfg[k] != v:
                raise SystemExit(f"prefix run {k}={cfg[k]} != {v}")
    seed = int(cfg.get("seed", a.seed))
    lrs = cfg.get("learning_rates")
    if lrs is None:
        span = min(3, a.jobs - 1)
        lrs = [1e-4 * 2 ** (span * j / (a.jobs - 1) - span / 2) if a.jobs > 1 else 1e-4 for j in range(a.jobs)]
    if lrs and a.lr_cycle:
        lrs = [lrs[j % len(lrs)] for j in range(max(a.jobs, len(lrs)))]
    if len(lrs) < a.jobs:
        raise SystemExit(f"prefix run has {len(lrs)} jobs < {a.jobs}")
    tok = AutoTokenizer.from_pretrained(str(a.data / "models" / a.train_size))
    rows = []
    with open(a.data / "train.jsonl", "rb") as f:
        for line in f:
            if len(rows) == (a.steps + a.prefill_chunks) * a.mb:
                break
            row = json.loads(line)
            ids = tok.encode(row["text"], add_special_tokens=False)[: a.seq]
            if len(ids) < 2:
                raise SystemExit("training row contains fewer than two tokens")
            rows.append((row["id"], ids))
    if len(rows) < (a.steps + a.prefill_chunks) * a.mb:
        raise SystemExit(f"only {len(rows)} rows for {a.steps} microbatches + {a.prefill_chunks} prefill chunks")
    prow, rows = rows[a.steps * a.mb:], rows[: a.steps * a.mb]
    toks = np.zeros((a.steps * a.mb, a.seq), np.int32)
    labs = np.full((a.steps * a.mb, a.seq), -100, np.int32)
    for i, (_, ids) in enumerate(rows):
        toks[i, : len(ids)] = ids
        labs[i, : len(ids) - 1] = ids[1:]
    a.out.mkdir(parents=True, exist_ok=True)
    toks.tofile(a.out / "tail_tokens.i32")
    labs.tofile(a.out / "tail_labels.i32")
    layer = a.layers - a.top_layers
    A = np.zeros((a.jobs, 2 * a.rank, a.hidden), np.float32)
    for p, off in enumerate((0, 503)):
        base = seed + 1009 * layer + off
        for j in range(a.jobs):
            g = torch.Generator(device="cpu").manual_seed(base + j)
            A[j, p * a.rank:(p + 1) * a.rank] = (torch.randn(a.rank, a.hidden, generator=g) / math.sqrt(a.hidden)).numpy()
    A.tofile(a.out / "tail_A.f32")
    np.array(lrs[: a.jobs], np.float32).tofile(a.out / "tail_lr.f32")
    ids = [[r[0] for r in rows[s * a.mb:(s + 1) * a.mb]] for s in range(a.steps)]
    (a.out / "tail_rows.json").write_text(json.dumps(dict(seed=seed, layer=layer, learning_rates=lrs[: a.jobs], row_ids=ids)))
    if prow:
        pt = np.zeros((len(prow), a.seq), np.int32)
        for i, (_, pids) in enumerate(prow):
            pt[i, : len(pids)] = pids
        pt.tofile(a.out / "prefill_tokens.i32")
        (a.out / "prefill_rows.json").write_text(json.dumps(dict(chunks=a.prefill_chunks, rows_per_chunk=a.mb, row_ids=[r[0] for r in prow])))
    print(f"tail_prep: {a.steps} microbatches, {int((labs >= 0).sum())} labels, jobs {a.jobs}, seed {seed}, layer {layer}")


if __name__ == "__main__":
    main()
