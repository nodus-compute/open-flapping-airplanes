"""Score --sm-split dynamic vs static 33 SMs vs stock MPS: python3 h100/dyn_score.py OUT -> OUT/dyn_summary.json, OUT/dyn_trace.json.

Per (decode batch b, Poisson rate r): training useful steps per paid GPU-hour vs per-job dedicated (solo_j0), decode tokens per
paid GPU-hour, DCGM SM-busy, combined gain U = train/train-alone + decode/decode-solo (vs 2 dedicated GPUs). Gates: per-job 16-step
losses byte-equal to each job alone; decode tokens byte-equal to decode solo; dynamic TPOT p99 <= 1.10x solo; every training tile
<= 5% of the solo decode step (TPOT p50); one launch per run; 100 ms DCGM; zero errors.
"""

import json
import re
import struct
import sys
from pathlib import Path

TRAIN_KINDS = {0: "base_q", 1: "base_v", 2: "lora_down", 3: "lora_up", 4: "dB", 5: "dU", 6: "dA", 7: "adam", 8: "loss",
               15: "trunk_up", 16: "trunk_down", 17: "trunk_cvt"}


def log(out, name):
    p = out / f"dyn_{name}.log"
    return p.read_text(errors="replace") if p.exists() else ""


def num(pat, text, g=1):
    m = re.search(pat, text)
    return float(m.group(g)) if m else None


def raw(out, run, name):
    p = out / "dyn" / run / name
    return p.read_bytes() if p.exists() else None


def u64s(b):
    return list(struct.unpack(f"<{len(b) // 8}Q", b)) if b else []


def sm_busy(out, name):
    p = out / f"dcgm_dyn_{name}.txt"
    if not p.exists():
        return None, 0
    vals, src = [], "dcgm_sm_active"
    for line in p.read_text(errors="replace").splitlines():
        t = line.replace(",", " ").split()
        if t and t[0] == "GPU" and len(t) >= 3:
            try:
                vals.append(float(t[2]))
            except ValueError:
                pass
        elif t and t[0].startswith("20") and len(t) >= 4:
            src = "nvidia_smi_util_gpu"
            try:
                vals.append(float(t[2].rstrip("%")) / 100)
            except ValueError:
                pass
    return (dict(mean=sum(vals) / len(vals), source=src) if vals else None), len(vals)


def loss_prefix_ok(out, run, n, steps=16):
    b = raw(out, run, "loss.f32")
    if not b or len(b) < steps * n * 4:
        return False
    for j in range(n):
        sl = raw(out, f"solo16_j{j}", "loss.f32")
        if not sl or any(b[(s * n + j) * 4:(s * n + j + 1) * 4] != sl[s * 4:(s + 1) * 4] for s in range(steps)):
            return False
    return True


def trace(out, run, L):
    td, arr, t0 = u64s(raw(out, run, "tok_done.u64")), u64s(raw(out, run, "arr.u64")), u64s(raw(out, run, "t0.u64"))
    if not td or not arr or not t0:
        return []
    rows = []
    for k, a in enumerate(arr):
        last = k * L + L - 1
        if last >= len(td) or not td[last]:
            break
        gaps = [(td[t] - td[t - 1]) * 1e-6 for t in range(k * L + 1, last + 1)]
        rows.append(dict(req=k, arrival_s=a * 1e-9, ttft_ms=(td[k * L] - t0[0] - a) * 1e-6, tpot_mean_ms=sum(gaps) / len(gaps), tpot_max_ms=max(gaps)))
    return rows


def main():
    out = Path(sys.argv[1])
    d = out / "dyn"
    n = len(list(d.glob("solo16_j*"))) if d.exists() else 0
    r_solo = num(r"-> ([0-9.]+) steps/s", log(out, "solo_j0"))
    r_alone = num(r"-> ([0-9.]+) steps/s", log(out, "train"))
    L = int(num(r"(\d+)", (out / "dyn_req_tokens").read_text()) if (out / "dyn_req_tokens").exists() else 64)
    keys = sorted({m.groups() for p in (d.glob("dec_b*_r*") if d.exists() else []) for m in [re.match(r"dec_b(\d+)_r([0-9.]+)$", p.name)] if m},
                  key=lambda x: (int(x[0]), float(x[1])))
    errors, rows, traces = [], [], {}
    gates = dict(bitwise_jobs=loss_prefix_ok(out, "train", n) if n else False, bitwise_decode=True, p99_le_1p10_solo=True,
                 tile_le_5pct_decode_step=True, one_launch=True, dcgm_100ms=True, zero_errors=True)
    dec_re = r"-> ([0-9.]+) tok/s, tpot p50 ([0-9.]+) ms p99 ([0-9.]+) ms"
    for b, r in keys:
        x = f"b{b}_r{r}"
        sl = log(out, f"dec_{x}")
        s_tok, s_p50, s_p99 = (num(dec_re, sl, i) for i in (1, 2, 3))
        toks_solo = raw(out, f"dec_{x}", "toks.i32")
        row = dict(b=int(b), rate=float(r), decode_solo=dict(tok_s=s_tok, tpot_p50_ms=s_p50, tpot_p99_ms=s_p99))
        for arm in ("static", "dynamic", "mps"):
            tl = log(out, f"mps_train_{x}") if arm == "mps" else log(out, f"{arm}_{x}")
            dl = log(out, f"mps_dec_{x}") if arm == "mps" else tl
            st = num(r"-> ([0-9.]+) steps/s", tl)
            tok, p99 = num(dec_re, dl, 1), num(dec_re, dl, 3)
            busy, ns = sm_busy(out, f"{arm}_{x}")
            run_dec = f"mps_dec_{x}" if arm == "mps" else f"{arm}_{x}"
            tc = raw(out, run_dec, "toks.i32")
            m = min(len(tc or b""), len(toks_solo or b""))
            dec_bit = bool(m) and tc[:m] == toks_solo[:m]
            a = dict(train_steps_s=st, useful_steps_per_paid_gpu_hour=3600 * n * st / r_solo if st and r_solo else None,
                     decode_tok_per_paid_gpu_hour=tok * 3600 if tok else None, tpot_p99_vs_solo=p99 / s_p99 if p99 and s_p99 else None,
                     sm_busy=busy, combined_gain_vs_2_dedicated=(st / r_alone + tok / s_tok) if st and r_alone and tok and s_tok else None,
                     decode_bitwise_vs_solo=dec_bit)
            if arm != "mps":
                a["train_bitwise_vs_solo"] = loss_prefix_ok(out, f"{arm}_{x}", n)
                tm = u64s(raw(out, f"{arm}_{x}", "tmax.u64"))
                a["max_tile_us"] = {v: tm[k] * 1e-3 for k, v in TRAIN_KINDS.items() if k < len(tm) and tm[k]}
                worst = max(a["max_tile_us"].values(), default=None)
                a["max_tile_vs_decode_step"] = worst * 1e-3 / s_p50 if worst and s_p50 else None
                gates["bitwise_jobs"] &= a["train_bitwise_vs_solo"]
                gates["one_launch"] &= "kernel launches 1 for" in tl
                if arm == "dynamic":
                    gates["tile_le_5pct_decode_step"] &= a["max_tile_vs_decode_step"] is not None and a["max_tile_vs_decode_step"] <= 0.05
                    gates["p99_le_1p10_solo"] &= a["tpot_p99_vs_solo"] is not None and a["tpot_p99_vs_solo"] <= 1.10
            gates["bitwise_decode"] &= dec_bit
            gates["dcgm_100ms"] &= ns > 0
            if st is None or tok is None:
                errors.append(f"{arm} {x}: missing train or decode rate")
            traces[f"{arm}_{x}"] = trace(out, run_dec, L)
            row[arm] = a
        rows.append(row)
    gates["zero_errors"] = not errors and bool(rows) and r_solo is not None and r_alone is not None
    gb = (out / "gemm_backend").read_text().strip() if (out / "gemm_backend").exists() else "mma"
    summary = dict(gemm_backend="wgmma" if gb == "wgmma" else "mma_sync", jobs=n, dedicated_job_steps_s=r_solo, train_alone_steps_s=r_alone,
                   rows=rows, gates=gates, passed=all(gates.values()), errors=errors,
                   notes="train bitwise = first 16 per-job losses vs each job alone; launch count is host-issued (1 persistent launch per run)")
    (out / "dyn_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "dyn_trace.json").write_text(json.dumps(traces))
    for row in rows:
        print(f"b{row['b']} r{row['rate']}: " + " | ".join(
            f"{a} train x{(row[a]['useful_steps_per_paid_gpu_hour'] or 0) / 3600:.2f} dec {row[a]['decode_tok_per_paid_gpu_hour'] or 0:.0f} tok/GPU-h "
            f"p99 {row[a]['tpot_p99_vs_solo'] or 0:.2f} U {row[a]['combined_gain_vs_2_dedicated'] or 0:.2f}" for a in ("static", "dynamic", "mps")))
    print(f"gates {gates} passed={summary['passed']}")
    for e in errors:
        print("ERROR", e)


if __name__ == "__main__":
    main()
