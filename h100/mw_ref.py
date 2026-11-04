"""PyTorch reference for the mw.cu LoRA-tail training workload: replays the dumped run (same init, data, targets, hyperparameters)
in fp32 eager PyTorch (base projection as a bf16 cuBLAS matmul, like the shared-base arm) and compares loss, params and last-step
gradients with the megakernel dump. Bitwise equality is impossible across reduction orders; this is the tolerance check.
  python h100/mw_ref.py <dump dir> --out mw_ref.json
"""
import argparse
import json
import pathlib

import numpy as np
import torch

D, NQ, NKV, R = 2560, 4096, 1024, 8
NQV, PJ, NX = NQ + NKV, 2 * R * D + (NQ + NKV) * R, 4
SCALE, LR, B1, B2, EPS = 2.0, 1e-3, 0.9, 0.999, 1e-8
TOL = {"loss_rel": 1e-3, "param_rel": 1e-2, "grad_rel": 1e-2}


def load(d, name, dtype):
    raw = np.fromfile(d / name, dtype=np.uint16 if dtype == "bf16" else dtype)
    t = torch.from_numpy(raw.astype(np.int16) if dtype == "bf16" else raw).cuda()
    return t.view(torch.bfloat16) if dtype == "bf16" else t


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    P = load(d, "init_P.f32", np.float32).clone()
    jobs = P.numel() // PJ
    P = P.view(jobs, PJ)
    X = load(d, "X.bf16", "bf16")
    tk = X.numel() // (NX * D)
    X = X.view(NX, tk, D)
    Wq, Wv = load(d, "Wq.bf16", "bf16").view(NQ, D), load(d, "Wv.bf16", "bf16").view(NKV, D)
    T = torch.cat([load(d, "Tq.bf16", "bf16").view(tk, NQ), load(d, "Tv.bf16", "bf16").view(tk, NKV)], 1).float()
    mk_loss = load(d, "loss.f32", np.float32)
    steps = mk_loss.numel() // jobs
    mk_loss = mk_loss.view(steps, jobs)
    M, V = torch.zeros_like(P), torch.zeros_like(P)
    inv = 1.0 / (tk * NQV)
    losses = []
    G = None
    for s in range(steps):
        x = X[s % NX]
        Y = torch.cat([x @ Wq.T, x @ Wv.T], 1).float()
        xf = x.float()
        A = P[:, :2 * R * D].view(jobs, 2 * R, D)
        Bq = P[:, 2 * R * D:2 * R * D + NQ * R].view(jobs, NQ, R)
        Bv = P[:, 2 * R * D + NQ * R:].view(jobs, NKV, R)
        U = torch.einsum("td,jrd->jtr", xf, A)
        out = Y + SCALE * torch.cat([U[:, :, :R] @ Bq.transpose(1, 2), U[:, :, R:] @ Bv.transpose(1, 2)], 2)
        diff = out - T
        losses.append((0.5 * diff.square().sum((1, 2)) * inv))
        dO = diff * inv
        dBq = SCALE * dO[:, :, :NQ].transpose(1, 2) @ U[:, :, :R]
        dBv = SCALE * dO[:, :, NQ:].transpose(1, 2) @ U[:, :, R:]
        dU = SCALE * torch.cat([dO[:, :, :NQ] @ Bq, dO[:, :, NQ:] @ Bv], 2)
        dA = dU.transpose(1, 2) @ xf
        G = torch.cat([dA.reshape(jobs, -1), dBq.reshape(jobs, -1), dBv.reshape(jobs, -1)], 1)
        M = B1 * M + (1 - B1) * G
        V = B2 * V + (1 - B2) * G * G
        bc1, bc2 = 1 - B1 ** (s + 1), 1 - B2 ** (s + 1)
        P = P - LR * (M / bc1) / ((V / bc2).sqrt() + EPS)
    ref_loss = torch.stack(losses)
    res = {
        "jobs": jobs, "tokens": tk, "steps": steps,
        "loss_rel": rel(mk_loss, ref_loss),
        "param_rel": rel(load(d, "P.f32", np.float32).view(jobs, PJ), P),
        "grad_rel": rel(load(d, "G.f32", np.float32).view(jobs, PJ), G),
        "loss_first": float(ref_loss[0, 0]), "loss_last": float(ref_loss[-1, 0]),
        "tolerance": TOL,
    }
    res["pass"] = all(res[k] <= v for k, v in TOL.items())
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
