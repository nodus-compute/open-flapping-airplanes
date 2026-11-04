"""Score the tensor-core attention backward (--attn-bwd tc): python3 h100/trunk_bwd_score.py OUT [--prefix-run DIR] -> OUT/trunk_bwd_summary.json.

Declared tolerance (bitwise vs the CUDA-core backward is not expected: dS is rounded to bf16 for the mma and the reduction order differs):
1-step LoRA grads rel-L2 <= 2e-2 and cosine >= 0.9999 vs --attn-bwd cuda; 16-step losses within 1e-3 relative of the cuda arm and of the
training_prefix dedicated-0 arm. tc run twice is byte-equal (losses and adapters). Reports per-kind max tile us (T_RADQ/T_RADV vs
T_RADQC/T_RADVC), N-job steps/s tc vs cuda, 100 ms DCGM, zero errors.
"""

import argparse
import json
import math
import struct
from pathlib import Path

from trunk_ckpt_score import log, rate, raw
from trunk_tail_score import prefix_losses

PRE = "tbw"
KINDS = {29: "T_RADQ", 30: "T_RADV", 33: "T_RADQC", 34: "T_RADVC"}


def f32(out, run, name):
    b = raw(out, run, name, PRE)
    return struct.unpack(f"{len(b) // 4}f", b) if b else ()


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prefix-run")
    ap.add_argument("--grad-tol", type=float, default=2e-2)
    ap.add_argument("--loss-tol", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out)
    errors = []
    gc, gt = f32(out, "g1_cuda", "G.f32"), f32(out, "g1_tc", "G.f32")
    grad = None
    if gc and len(gc) == len(gt):
        dn = math.sqrt(sum((x - y) ** 2 for x, y in zip(gt, gc)))
        nc, nt = math.sqrt(sum(x * x for x in gc)), math.sqrt(sum(x * x for x in gt))
        grad = dict(rel_l2=dn / max(nc, 1e-30), cosine=sum(x * y for x, y in zip(gt, gc)) / max(nc * nt, 1e-30))
    else:
        errors.append("missing 1-step grads")
    lc, lt = f32(out, "l16_cuda", "loss.f32")[:16], f32(out, "l16_tc", "loss.f32")[:16]
    loss_vs_cuda = max((rel(x, y) for x, y in zip(lt, lc)), default=None) if len(lt) == 16 == len(lc) else None
    pref = prefix_losses(Path(a.prefix_run), 0) if a.prefix_run else []
    loss_vs_prefix = max((rel(x, float(r[1])) for x, r in zip(lt, pref)), default=None) if len(pref) == 16 and len(lt) == 16 else None
    rep = all(raw(out, "l16_tc", n, PRE) and raw(out, "l16_tc", n, PRE) == raw(out, "l16_tc2", n, PRE) for n in ("loss.f32", "P.f32"))
    tiles = {}
    for ab in ("cuda", "tc"):
        b = raw(out, f"g1_{ab}", "tmax.u64", PRE)
        tm = struct.unpack(f"{len(b) // 8}Q", b) if b else ()
        tiles[ab] = {v: tm[k] * 1e-3 for k, v in KINDS.items() if k < len(tm) and tm[k]}
    runs = sorted(p.name for p in (out / PRE).glob("n*_cuda")) if (out / PRE).exists() else []
    n = runs[0][: -len("_cuda")] if runs else None
    rc, rt = (rate(log(out, f"{n}_{ab}", PRE)) for ab in ("cuda", "tc")) if n else (None, None)
    if not (rc and rt):
        errors.append("missing N-job rates")
    gates = dict(grad_tol=bool(grad) and grad["rel_l2"] <= a.grad_tol and grad["cosine"] >= 0.9999,
                 loss_vs_cuda=loss_vs_cuda is not None and loss_vs_cuda <= a.loss_tol,
                 loss_vs_prefix=loss_vs_prefix is not None and loss_vs_prefix <= a.loss_tol if a.prefix_run else None,
                 tc_repeat_bitwise=rep, dcgm_100ms=bool(n) and all((out / f"dcgm_{PRE}_{n}_{ab}.txt").exists() for ab in ("cuda", "tc")))
    gates["zero_errors"] = not errors
    s = dict(grad=grad, loss_rel_vs_cuda=loss_vs_cuda, loss_rel_vs_prefix=loss_vs_prefix, max_tile_us=tiles, arm=n,
             steps_s=dict(cuda=rc, tc=rt), tc_speedup=rt / rc if rc and rt else None, gates=gates,
             passed=all(v for v in gates.values() if v is not None), errors=errors)
    (out / "trunk_bwd_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(dict(gates=gates, tc_speedup=s["tc_speedup"], max_tile_us=tiles)))


if __name__ == "__main__":
    main()
