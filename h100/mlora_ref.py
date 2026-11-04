"""PyTorch side of the mlora arm (llm/h100/mlora/mlora.cu).

check DIR : fp32 PyTorch forward of decode token 0 from the inputs the binary dumped with --dump-ref; compares the binary's logits0
            and greedy token 0 (tolerance, not bitwise: different reduction order).
bench     : stock PyTorch eager decode of the same layer shapes (bf16 cuBLAS GEMMs, SDPA GQA attention, tied lm_head, greedy) with
            unfused LoRA: "loop" = one shrink/expand pair per adapter over its rows (the PEFT pattern), "gather" = all rows in one
            batched bmm with per-row gathered adapters (BGMV in plain PyTorch), "none" = no LoRA.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D, NQ, NKV, HD, NQH, NKVH, VOC, R = 2560, 4096, 1024, 128, 32, 8, 151936, 8
NQKV, PJ, SCALE = NQ + 2 * NKV, 2 * R * D + (NQ + NKV) * R, 2.0
TOL = {"logits_rel": 1e-3}


def bf16(path, shape):
    a = np.fromfile(path, dtype=np.uint16).astype(np.int16)
    return torch.from_numpy(a).cuda().view(torch.bfloat16).float().view(*shape)


def f32(path, shape=None):
    t = torch.from_numpy(np.fromfile(path, dtype=np.float32)).cuda()
    return t.view(*shape) if shape else t


def check(d, out):
    d = Path(d)
    res = json.loads((d / "result.json").read_text())
    b, na, ctx = res["b"], max(res["adapters"], 1), res["ctx"]
    h0 = f32(d / "h0.f32", (b, D))
    W = bf16(d / "Wqkv.bf16", (NQKV, D))
    Wo = bf16(d / "Wo.bf16", (D, NQ))
    E = bf16(d / "E.bf16", (VOC, D))
    adp = f32(d / "adp.f32", (na, PJ))
    aid = torch.from_numpy(np.fromfile(d / "aid.i32", dtype=np.int32)).long().cuda()
    kc = bf16(d / "kc.bf16", (b, ctx, NKV))
    vc = bf16(d / "vc.bf16", (b, ctx, NKV))
    with torch.no_grad():
        base = h0 @ W.T
        q, k, v = base[:, :NQ], base[:, NQ:NQ + NKV], base[:, NQ + NKV:]
        if res["lora"] != "none":
            a = adp[aid]
            Aq, Av = a[:, :R * D].view(b, R, D), a[:, R * D:2 * R * D].view(b, R, D)
            Bq = a[:, 2 * R * D:2 * R * D + NQ * R].view(b, NQ, R)
            Bv = a[:, 2 * R * D + NQ * R:].view(b, NKV, R)
            uq, uv = torch.einsum("bd,brd->br", h0, Aq), torch.einsum("bd,brd->br", h0, Av)
            q = q + SCALE * torch.einsum("br,bor->bo", uq, Bq)
            v = v + SCALE * torch.einsum("br,bor->bo", uv, Bv)
        k = k.bfloat16().float()
        v = v.bfloat16().float()
        K = torch.cat([kc, k[:, None]], 1).view(b, ctx + 1, NKVH, HD).repeat_interleave(NQH // NKVH, 2)
        V = torch.cat([vc, v[:, None]], 1).view(b, ctx + 1, NKVH, HD).repeat_interleave(NQH // NKVH, 2)
        s = torch.einsum("bhe,bphe->bhp", q.view(b, NQH, HD), K) / math.sqrt(HD)
        att = torch.einsum("bhp,bphe->bhe", s.softmax(-1), V).reshape(b, NQ)
        h1 = h0 + att @ Wo.T
        ref = h1 @ E.T
    got = f32(d / "logits0.f32", (b, VOC))
    toks = torch.from_numpy(np.fromfile(d / "toks.i32", dtype=np.int32)[:b].copy()).long().cuda()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    ref_tok = ref.argmax(-1)
    # greedy token 0 must match unless the reference top-2 are within the tolerance of each other
    gap = (ref.gather(1, ref_tok[:, None]) - ref.gather(1, toks[:, None])).squeeze(1)
    tok_ok = bool(((toks == ref_tok) | (gap <= TOL["logits_rel"] * ref.abs().max())).all())
    r = {"dir": str(d), "lora": res["lora"], "b": b, "adapters": res["adapters"], "logits_rel_err": rel, "tok0_match": tok_ok,
         "tok0_exact_rows": int((toks == ref_tok).sum()), "tol": TOL, "pass": rel <= TOL["logits_rel"] and tok_ok}
    print(json.dumps(r, indent=1))
    Path(out).write_text(json.dumps(r, indent=1))
    return 0 if r["pass"] else 1


def bench_one(b, na, mode, ctx, ntok, g):
    dev, bt = "cuda", torch.bfloat16
    rnd = lambda *s, sc: (torch.rand(*s, generator=g, device=dev) * 2 - 1).mul_(sc).to(bt)  # noqa: E731
    W, Wo, E = rnd(NQKV, D, sc=math.sqrt(3 / D)), rnd(D, NQ, sc=math.sqrt(3 / NQ)), rnd(VOC, D, sc=math.sqrt(3))
    nad = max(na, 1)
    Aq, Av = rnd(nad, R, D, sc=math.sqrt(3 / D)), rnd(nad, R, D, sc=math.sqrt(3 / D))
    Bq, Bv = rnd(nad, NQ, R, sc=0.02), rnd(nad, NKV, R, sc=0.02)
    Kc = rnd(b, NKVH, ctx + ntok + 8, HD, sc=math.sqrt(3))
    Vc = rnd(b, NKVH, ctx + ntok + 8, HD, sc=math.sqrt(3))
    aid = torch.arange(b, device=dev) % nad
    groups = [(a, (aid == a).nonzero().squeeze(1)) for a in range(na)]
    x = E[1:b + 1].clone()

    def step(x, pos):
        qkv = x @ W.T
        q, k, v = qkv[:, :NQ], qkv[:, NQ:NQ + NKV], qkv[:, NQ + NKV:]
        if mode == "loop":
            q, v = q.clone(), v.clone()
            for a, idx in groups:
                xa = x[idx]
                q[idx] += SCALE * ((xa @ Aq[a].T) @ Bq[a].T)
                v[idx] += SCALE * ((xa @ Av[a].T) @ Bv[a].T)
        elif mode == "gather":
            uq = torch.bmm(Aq[aid], x[:, :, None])
            uv = torch.bmm(Av[aid], x[:, :, None])
            q = q + SCALE * torch.bmm(Bq[aid], uq).squeeze(2)
            v = v + SCALE * torch.bmm(Bv[aid], uv).squeeze(2)
        Kc[:, :, pos] = k.view(b, NKVH, HD)
        Vc[:, :, pos] = v.view(b, NKVH, HD)
        o = F.scaled_dot_product_attention(q.view(b, NQH, 1, HD), Kc[:, :, :pos + 1], Vc[:, :, :pos + 1], enable_gqa=True)
        h = x + o.reshape(b, NQ) @ Wo.T
        tok = (h @ E.T).argmax(-1)
        return E[tok]

    with torch.no_grad():
        for t in range(8):
            x = step(x, ctx + t)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for t in range(8, ntok):
            x = step(x, ctx + t)
        e1.record()
        torch.cuda.synchronize()
    sec = e0.elapsed_time(e1) / 1e3
    tps = (ntok - 8) * b / sec
    return {"system": f"pytorch_eager_{mode}", "b": b, "adapters": na, "tok_s": tps, "tok_per_gpu_hour": tps * 3600,
            "tpot_mean_ms": sec * 1e3 / (ntok - 8)}


def bench(args):
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for b in args.batches:
        rows.append(bench_one(b, 0, "none", args.ctx, args.tokens, g))
        for na in args.adapters:
            if na <= b:
                for mode in ("loop", "gather"):
                    rows.append(bench_one(b, na, mode, args.ctx, args.tokens, g))
        torch.cuda.empty_cache()
    for r in rows:
        print(json.dumps(r))
    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--out", required=True)
    bp = sub.add_parser("bench")
    bp.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64])
    bp.add_argument("--adapters", type=int, nargs="+", default=[1, 8, 32])
    bp.add_argument("--ctx", type=int, default=1024)
    bp.add_argument("--tokens", type=int, default=256)
    bp.add_argument("--out", required=True)
    a = ap.parse_args()
    return check(a.dir, a.out) if a.cmd == "check" else bench(a)


if __name__ == "__main__":
    sys.exit(main())
