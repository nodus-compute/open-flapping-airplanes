"""Score the real-model (--trunk-ckpt) trunk: python3 h100/trunk_ckpt_score.py OUT [--pytorch-dedicated-steps-s R] -> OUT/trunk_ckpt_summary.json.

Gates: cut hidden vs HF fp32 (trunk_ckpt_ref.json); per-job 16-step losses and final adapters of shared and perjob arms byte-equal to
each job alone; one launch per run; 100 ms DCGM; zero errors. Report useful steps per paid GPU-hour vs the megakernel dedicated arm
(solo_j0) and, if given, vs the audited PyTorch dedicated per-job rate.
"""

import argparse
import json
import re
from pathlib import Path

PJ = 2 * 8 * 2560 + (4096 + 1024) * 8


def raw(out, run, name):
    p = out / "tck" / run / name
    return p.read_bytes() if p.exists() else None


def log(out, name):
    p = out / f"tck_{name}.log"
    return p.read_text(errors="replace") if p.exists() else ""


def rate(text):
    m = re.search(r"-> ([0-9.]+) steps/s", text)
    return float(m.group(1)) if m else None


def bitwise(out, run, n, steps=16):
    lo, pa = raw(out, run, "loss.f32"), raw(out, run, "P.f32")
    if not lo or not pa or len(lo) < steps * n * 4 or len(pa) < n * PJ * 4:
        return False
    for j in range(n):
        sl, sp = raw(out, f"solo16_j{j}", "loss.f32"), raw(out, f"solo16_j{j}", "P.f32")
        if not sl or not sp or sp[: PJ * 4] != pa[j * PJ * 4:(j + 1) * PJ * 4]:
            return False
        if any(lo[(s * n + j) * 4:(s * n + j + 1) * 4] != sl[s * 4:(s + 1) * 4] for s in range(steps)):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pytorch-dedicated-steps-s", type=float)
    a = ap.parse_args()
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
