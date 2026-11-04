"""Score the real top-layer tail (--tail real): python3 h100/trunk_tail_score.py OUT [--prefix-run DIR] [--dedicated-rate-from SUMMARY]
-> OUT/trunk_tail_summary.json.

Gates: each job's first 16 losses within 1e-3 relative of the training_prefix dedicated-<job> arm (same rows, seed, lr); shared and
perjob 16-step losses and adapters byte-equal to each job alone; one launch per run; 100 ms DCGM; zero errors. Report useful steps per
paid GPU-hour vs the megakernel dedicated arm and vs the audited PyTorch dedicated rate.
"""

import argparse
import json
import re
import struct
from pathlib import Path

from trunk_ckpt_score import bitwise, log, pytorch_rate, rate, raw

PRE = "ttl"


def prefix_losses(run, job, n=16):
    rows = []
    for p in (run / f"dedicated-{job}").rglob("*.jsonl"):
        for line in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "losses" in r and "unit" in r:
                k = r["jobs"].index(job) if job in r.get("jobs", []) else 0
                rows.append((r["unit"], r["losses"][k], r.get("row_ids")))
    rows.sort(key=lambda x: x[0])
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prefix-run")
    ap.add_argument("--dedicated-rate-from")
    ap.add_argument("--rtol", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out)
    d = out / PRE
    py_rate = pytorch_rate(a.dedicated_rate_from) if a.dedicated_rate_from else None
    rowsf = d / "data" / "tail_rows.json"
    want_ids = json.loads(rowsf.read_text())["row_ids"] if rowsf.exists() else None
    solos = sorted(int(m.group(1)) for p in d.glob("solo16_j*") for m in [re.match(r"solo16_j(\d+)$", p.name)] if m) if d.exists() else []
    errors, loss_rows, worst = [], [], 0.0
    loss_gate = bool(solos) and bool(a.prefix_run)
    for j in solos:
        b = raw(out, f"solo16_j{j}", "loss.f32", PRE)
        ours = list(struct.unpack(f"{len(b) // 4}f", b))[:16] if b else []
        ref = prefix_losses(Path(a.prefix_run), j) if a.prefix_run else []
        rel = [abs(x - r[1]) / abs(r[1]) for x, r in zip(ours, ref) if r[1]]
        ids_ok = want_ids is None or all(r[2] is None or list(map(str, r[2])) == list(map(str, w)) for r, w in zip(ref, want_ids))
        ok = len(ours) == 16 and len(ref) == 16 and len(rel) == 16 and max(rel) <= a.rtol and ids_ok
        worst = max([worst] + rel)
        loss_gate &= ok
        loss_rows.append(dict(job=j, ours=ours, pytorch=[r[1] for r in ref], max_rel=max(rel) if rel else None, row_ids_match=ids_ok, ok=ok))
    if not a.prefix_run:
        errors.append("no --prefix-run: first-16-loss gate not evaluated")
    r_solo = rate(log(out, "solo_j0", PRE))
    ns = sorted({int(m.group(1)) for p in d.glob("shared_n*") for m in [re.match(r"shared_n(\d+)$", p.name)] if m}) if d.exists() else []
    gates = dict(loss_vs_training_prefix=loss_gate, bitwise_jobs=bool(ns), one_launch=True, dcgm_100ms=True, zero_errors=True)
    rows = []
    for n in ns:
        row = dict(jobs=n)
        for arm in ("shared", "perjob"):
            t = log(out, f"{arm}_n{n}", PRE)
            r = rate(t)
            ok = bitwise(out, f"{arm}16_n{n}", n, pre=PRE)
            gates["bitwise_jobs"] &= ok
            gates["one_launch"] &= "kernel launches 1 for" in t
            gates["dcgm_100ms"] &= (out / f"dcgm_{PRE}_{arm}_n{n}.txt").exists()
            if r is None or r_solo is None:
                errors.append(f"{arm} n{n}: missing rate")
            row[arm] = dict(steps_s=r, bitwise_vs_solo=ok, useful_steps_per_paid_gpu_hour=3600 * n * r if r else None,
                            vs_megakernel_dedicated=n * r / r_solo if r and r_solo else None,
                            vs_pytorch_dedicated=n * r / py_rate if r and py_rate else None)
        rows.append(row)
    gates["zero_errors"] = not errors and bool(rows)
    s = dict(megakernel_dedicated_steps_s=r_solo, pytorch_dedicated_steps_s=py_rate, loss_rtol=a.rtol, worst_loss_rel=worst,
             losses=loss_rows, rows=rows, gates=gates, passed=all(gates.values()), errors=errors)
    (out / "trunk_tail_summary.json").write_text(json.dumps(s, indent=1))
    for row in rows:
        print(f"N{row['jobs']}: shared x{row['shared']['vs_megakernel_dedicated'] or 0:.2f} (vs PyTorch dedicated x{row['shared']['vs_pytorch_dedicated'] or 0:.2f})")
    print(f"worst first-16 loss rel {worst:.2e}; gates {gates} passed={s['passed']}")


if __name__ == "__main__":
    main()
