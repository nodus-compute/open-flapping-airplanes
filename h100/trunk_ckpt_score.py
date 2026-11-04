"""Score the real-model (--trunk-ckpt) trunk: python3 h100/trunk_ckpt_score.py OUT [--pytorch-dedicated-steps-s R | --dedicated-rate-from SUMMARY] -> OUT/trunk_ckpt_summary.json.

Gates: cut hidden vs HF fp32 (trunk_ckpt_ref.json); per-job 16-step losses and final adapters of shared and perjob arms byte-equal to
each job alone; one launch per run; 100 ms DCGM; zero errors. Report useful steps per paid GPU-hour vs the megakernel dedicated arm
(solo_j0) and, if given, vs the audited PyTorch dedicated per-job rate.
"""

import argparse
import json
import re
from pathlib import Path

PJ = 2 * 8 * 2560 + (4096 + 1024) * 8


def raw(out, run, name, pre="tck"):
    p = out / pre / run / name
    return p.read_bytes() if p.exists() else None


def log(out, name, pre="tck"):
    p = out / f"{pre}_{name}.log"
    return p.read_text(errors="replace") if p.exists() else ""


def rate(text):
    m = re.search(r"-> ([0-9.]+) steps/s", text)
    return float(m.group(1)) if m else None


def bitwise(out, run, n, steps=16, pre="tck"):
    lo, pa = raw(out, run, "loss.f32", pre), raw(out, run, "P.f32", pre)
    if not lo or not pa or len(lo) < steps * n * 4 or len(pa) < n * PJ * 4:
        return False
    for j in range(n):
        sl, sp = raw(out, f"solo16_j{j}", "loss.f32", pre), raw(out, f"solo16_j{j}", "P.f32", pre)
        if not sl or not sp or sp[: PJ * 4] != pa[j * PJ * 4:(j + 1) * PJ * 4]:
            return False
        if any(lo[(s * n + j) * 4:(s * n + j + 1) * 4] != sl[s * 4:(s + 1) * 4] for s in range(steps)):
            return False
    return True


def pytorch_rate(path):
    """Mean per-job dedicated steps/s of an audited training_prefix summary.json (count / duration, as training_prefix_score)."""
    p = Path(path)
    s = json.loads(p.read_text())
    cfgp = p.parent / "config.json"
    dur = json.loads(cfgp.read_text()).get("duration") if cfgp.exists() else None
    rates = []
    for name, arm in s.get("arms", {}).items():
        if name.startswith("dedicated-"):
            job = name.split("-", 1)[1]
            secs = dur or arm.get("billed_gpu_seconds")
            if secs and job in arm.get("count", {}):
                rates.append(arm["count"][job] / secs)
    return sum(rates) / len(rates) if rates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pytorch-dedicated-steps-s", type=float)
    ap.add_argument("--dedicated-rate-from", help="audited training_prefix summary.json")
    a = ap.parse_args()
    if a.dedicated_rate_from and not a.pytorch_dedicated_steps_s:
        a.pytorch_dedicated_steps_s = pytorch_rate(a.dedicated_rate_from)
    out = Path(a.out)
    refp = out / "tck" / "ref1" / "trunk_ckpt_ref.json"
    ref = json.loads(refp.read_text()) if refp.exists() else {}
    r_solo = rate(log(out, "solo_j0"))
    ns = sorted({int(m.group(1)) for p in (out / "tck").glob("shared_n*") for m in [re.match(r"shared_n(\d+)$", p.name)] if m}) if (out / "tck").exists() else []
    rows, errors = [], []
    gates = dict(hf_cut_hidden=bool(ref.get("gate_pass")), bitwise_jobs=bool(ns), one_launch=True, dcgm_100ms=True, zero_errors=True)
    for n in ns:
        row = dict(jobs=n)
        for arm in ("shared", "perjob"):
            t = log(out, f"{arm}_n{n}")
            r = rate(t)
            ok = bitwise(out, f"{arm}16_n{n}", n)
            gates["bitwise_jobs"] &= ok
            gates["one_launch"] &= "kernel launches 1 for" in t
            gates["dcgm_100ms"] &= (out / f"dcgm_tck_{arm}_n{n}.txt").exists()
            if r is None or r_solo is None:
                errors.append(f"{arm} n{n}: missing rate")
            row[arm] = dict(steps_s=r, bitwise_vs_solo=ok, useful_steps_per_paid_gpu_hour=3600 * n * r if r else None,
                            vs_megakernel_dedicated=n * r / r_solo if r and r_solo else None,
                            vs_pytorch_dedicated=n * r / a.pytorch_dedicated_steps_s if r and a.pytorch_dedicated_steps_s else None)
        rows.append(row)
    gates["zero_errors"] = not errors and bool(rows)
    s = dict(ref=ref, megakernel_dedicated_steps_s=r_solo, pytorch_dedicated_steps_s=a.pytorch_dedicated_steps_s, rows=rows,
             gates=gates, passed=all(gates.values()), errors=errors)
    (out / "trunk_ckpt_summary.json").write_text(json.dumps(s, indent=1))
    for row in rows:
        print(f"N{row['jobs']}: shared x{row['shared']['vs_megakernel_dedicated'] or 0:.2f} perjob x{row['perjob']['vs_megakernel_dedicated'] or 0:.2f} vs megakernel dedicated")
    print(f"gates {gates} passed={s['passed']}")


if __name__ == "__main__":
    main()
