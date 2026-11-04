"""Read-only verifier for a codec bundle: python3 h100/codec_verify.py OUT [--min-co-vs-time-slice X] -> JSON on stdout; exit 0 only on a passing REAL H100 bundle.

Writes nothing. Recomputes from raw artifacts, not from codec_summary.json numbers:
  counts:  decode tokens = token steps x b (decode log line, toks.i32 must exist), job-steps = len(loss.f32) / 4
  billed:  host wall seconds the GPU was held per arm (<arm>.wall from run_h100.sh dk: start/end/rc); time-slice = union of its 2 processes
  U = (co tokens/billed) / (decode-solo tokens/billed) + (co job-steps/billed) / (train-alone job-steps/billed), per paid GPU-hour;
  2 dedicated GPUs = 1.0 per paid GPU-hour; co_vs_time_slice = U_co / U_ts; best co arm (reserve) per b.
Real-H100 evidence: preflight.json passed with gpu_name containing --gpu-name (default H100 80GB HBM3; H100 PCIe for the Moeslim class) and the in-container MATMUL_OK; trunk_ckpt_summary
hf_cut_hidden; every arm rc 0; DCGM/nvidia-smi 100 ms samples covering >= 50% of each gated arm's billed time.
CODEC_PREFILL=1 bundles (pf solo present): prefill tokens = chunks done x chunk tokens (prefill log lines, all chunks done), pf_out.bf16
byte-equal to pf solo in every co arm and the time-slice third process, U adds f_prefill, the dedicated baseline is 3 GPUs (1.0 per paid GPU-hour).
Gates rechecked: decode ids bitwise vs solo and per-job first-16 losses bitwise vs solo16 (co and time-slice), p99 TPOT <= 1.10x solo,
decode outlasts training (co), one launch, zero errors; and codec_summary.json itself passed with the same gate names true.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from codec_score import PRE, dec, losses_ok, u64s
from trunk_ckpt_score import log, raw

PFT = re.compile(r"prefill: (\d+) chunks x (\d+) tokens")
PFD = re.compile(r"prefill: (\d+) of (\d+) chunks done")
STEPS = re.compile(r"decode: b=(\d+), (\d+) token steps")


def wall(out, name):
    p = out / f"{PRE}_{name}.wall"
    t = dict(ln.split()[:2] for ln in p.read_text().splitlines() if len(ln.split()) >= 2) if p.exists() else {}
    return (float(t["start"]), float(t["end"]), int(t["rc"])) if {"start", "end", "rc"} <= t.keys() else None


def dcgm_ok(out, name, secs):
    p = out / f"dcgm_{PRE}_{name}.txt"
    rows = [ln for ln in p.read_text(errors="replace").splitlines() if re.search(r"\d", ln) and not ln.startswith(("#", "timestamp"))] if p.exists() else []
    return bool(secs) and len(rows) >= 0.5 * secs / 0.1, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tpot-ratio", type=float, default=1.10)
    ap.add_argument("--gpu-name", default="H100 80GB HBM3")
    ap.add_argument("--min-co-vs-time-slice", type=float, default=0.0)
    a = ap.parse_args()
    out, fail = Path(a.out), []

    def need(name, ok, detail=""):
        if not ok:
            fail.append(f"{name}: {detail}" if detail else name)
        return ok

    def jread(p):
        try:
            return json.loads((out / p).read_text())
        except (OSError, ValueError):
            return None

    pf, tk, sm = jread("preflight.json"), jread("trunk_ckpt_summary.json"), jread("codec_summary.json")
    ck = (pf or {}).get("checks", {})
    need("preflight_passed", bool(pf and pf.get("passed")))
    need("real_h100", a.gpu_name in ck.get("gpu_name", {}).get("detail", "") and ck.get("gpu_name", {}).get("ok"), ck.get("gpu_name"))
    need("real_cuda_matmul", "MATMUL_OK" in ck.get("container_cuda_matmul_nvcc", {}).get("detail", ""))
    need("hf_cut_hidden", bool(tk and tk.get("gates", {}).get("hf_cut_hidden")))
    need("summary_passed", bool(sm and sm.get("passed")), (sm or {}).get("gates"))
    for g in ("decode_bitwise", "train_bitwise", "tpot_p99", "rates_overlap", "one_launch", "dcgm_100ms", "ts_decode_bitwise", "ts_train_bitwise", "ts_present", "zero_errors"):
        need(f"summary_gate_{g}", bool(sm and sm.get("gates", {}).get(g)))

    d = out / PRE
    solos = sorted(int(m.group(1)) for p in d.glob("solo16_j*") for m in [re.match(r"solo16_j(\d+)$", p.name)] if m) if d.is_dir() else []
    n = len(solos)
    need("solo16_refs", n > 0 and solos == list(range(n)), solos[:4])

    def per_s(x, s):
        return x / s if x and s else None

    def billed(*names):
        w = [wall(out, x) for x in names]
        if not all(w):
            need(f"wall_{'+'.join(names)}", False, "missing .wall")
            return None
        for x, t in zip(names, w):
            need(f"rc0_{x}", t[2] == 0, t[2])
        return max(t[1] for t in w) - min(t[0] for t in w)

    def tokens(name):
        m = STEPS.search(log(out, name, PRE))
        need(f"toks_{name}", bool(raw(out, name, "toks.i32", PRE)) and bool(m))
        return int(m.group(1)) * int(m.group(2)) if m else None

    def job_steps(name):
        b = raw(out, name, "loss.f32", PRE)
        need(f"losses_{name}", bool(b))
        return len(b) // 4 if b else None

    t_alone_s = billed("train")
    r_train = per_s(job_steps("train"), t_alone_s)
    for x in ["train", *(f"solo16_j{j}" for j in solos)]:
        billed(x)
    dc_ok = {"train": dcgm_ok(out, "train", t_alone_s)}
    r_pf = pf_solo = None
    def pf_tokens(name):
        t = log(out, name, PRE)
        m, k = PFT.search(t), PFD.search(t)
        need(f"prefill_{name}", bool(m and k) and k.group(1) == k.group(2) and bool(raw(out, name, "pf_out.bf16", PRE)))
        return int(k.group(1)) * int(m.group(2)) if m and k else None

    has_pf = bool(log(out, "pf", PRE))
    if has_pf:
        need("summary_gate_prefill_bitwise", bool(sm and sm.get("gates", {}).get("prefill_bitwise")))
        need("summary_gate_ts_prefill_bitwise", bool(sm and sm.get("gates", {}).get("ts_prefill_bitwise")))
        pf_s = billed("pf")
        r_pf = per_s(pf_tokens("pf"), pf_s)
        pf_solo = raw(out, "pf", "pf_out.bf16", PRE)
        dc_ok["pf"] = dcgm_ok(out, "pf", pf_s)

    by_b = {}
    for p in sorted(d.glob("co_b*_r*")) if d.is_dir() else []:
        m = re.match(r"co_b(\d+)_r(\d+)$", p.name)
        if m:
            by_b.setdefault(int(m.group(1)), []).append((int(m.group(2)), p.name))
    need("co_arms", bool(by_b))
    cmp = []
    for b, arms in sorted(by_b.items()):
        sd_s = billed(f"dec_b{b}")
        r_dec = per_s(tokens(f"dec_b{b}"), sd_s)
        ds = dec(log(out, f"dec_b{b}", PRE))
        dc_ok[f"dec_b{b}"] = dcgm_ok(out, f"dec_b{b}", sd_s)
        toks_solo = raw(out, f"dec_b{b}", "toks.i32", PRE)

        def u(name, secs, tok_name, trn_name, pf_name=None):
            rd, rt = per_s(tokens(tok_name), secs), per_s(job_steps(trn_name), secs)
            fd, ft = (rd / r_dec if rd and r_dec else None), (rt / r_train if rt and r_train else None)
            rp = fp = None
            if has_pf:
                rp = per_s(pf_tokens(pf_name), secs)
                fp = rp / r_pf if rp and r_pf else None
                need(f"bitwise_prefill_{pf_name}", bool(pf_solo) and raw(out, pf_name, "pf_out.bf16", PRE) == pf_solo)
                if fp is None:
                    fd = None
            need(f"bitwise_tokens_{tok_name}", bool(toks_solo) and raw(out, tok_name, "toks.i32", PRE) == toks_solo)
            need(f"bitwise_losses_{trn_name}", losses_ok(out, trn_name, solos, ""))
            return dict(arm=name, billed_s=secs, decode_tokens_per_paid_gpu_hour=3600 * rd if rd else None,
                        job_steps_per_paid_gpu_hour=3600 * rt if rt else None, f_decode=fd, f_train=ft,
                        prefill_tokens_per_paid_gpu_hour=3600 * rp if rp else None, f_prefill=fp,
                        U=fd + ft + (fp or 0.0) if fd is not None and ft is not None else None)

        rows = []
        for rs, name in arms:
            s = billed(name)
            r = u(name, s, name, name, name)
            r["reserve"] = rs
            t = log(out, name, PRE)
            dcx = dec(t)
            r["tpot_p99_ok"] = need(f"tpot_{name}", bool(ds and dcx) and dcx["p99_ms"] <= a.tpot_ratio * ds["p99_ms"], (ds, dcx))
            sdn, tdn = [x for x in u64s(out, name, "step_done.u64") if x], u64s(out, name, "tok_done.u64")
            need(f"overlap_{name}", bool(sdn and tdn) and tdn[-1] >= sdn[-1])
            need(f"one_launch_{name}", "kernel launches 1 for" in t)
            dc_ok[name] = dcgm_ok(out, name, s)
            rows.append(r)
        ts_s = billed(f"ts_dec_b{b}", f"ts_train_b{b}", *([f"ts_pf_b{b}"] if has_pf else []))
        ts = u(f"ts_b{b}", ts_s, f"ts_dec_b{b}", f"ts_train_b{b}", f"ts_pf_b{b}")
        dc_ok[f"ts_b{b}"] = dcgm_ok(out, f"ts_b{b}", ts_s)
        best = max((r for r in rows if r["U"] is not None), key=lambda r: r["U"], default=None)
        cvt = best["U"] / ts["U"] if best and ts["U"] else None
        need(f"co_vs_time_slice_b{b}", cvt is not None and cvt >= a.min_co_vs_time_slice, cvt)
        cmp.append(dict(b=b, co_rows=rows, best_co=best and best["arm"], time_slice=ts,
                        dedicated=dict(gpus=3 if has_pf else 2, U=1.0,
                                       decode_tokens_per_paid_gpu_hour=3600 * r_dec / (3 if has_pf else 2) if r_dec else None,
                                       job_steps_per_paid_gpu_hour=3600 * r_train / (3 if has_pf else 2) if r_train else None,
                                       prefill_tokens_per_paid_gpu_hour=3600 * r_pf / 3 if has_pf and r_pf else None),
                        co_vs_time_slice=cvt, co_vs_2_dedicated=best and best["U"],
                        summary_co_vs_time_slice=next((c.get("co_vs_time_slice") for c in (sm or {}).get("comparison", []) if c.get("b") == b), None),
                        summary_co_vs_2_dedicated=next((c.get("co_vs_2_dedicated") for c in (sm or {}).get("comparison", []) if c.get("b") == b), None)))
    for k, (ok, rows) in dc_ok.items():
        need(f"dcgm_100ms_{k}", ok, rows)
    need("zero_errors", not (sm or {}).get("errors"), (sm or {}).get("errors"))
    res = dict(bundle=str(out), jobs=n, train_alone_job_steps_per_s_billed=r_train, comparison=cmp, passed=not fail, failed=fail)
    print(json.dumps(res, indent=1))
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
