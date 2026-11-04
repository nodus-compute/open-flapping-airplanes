"""Score the in-megakernel shared-trunk arm: python3 h100/trunk_score.py OUT -> OUT/trunk_summary.json (+ table on stdout).

Dedicated calibration as in the audited PyTorch shared-trunk runs: each job alone on the GPU (solo_j*), useful equivalent
GPU-seconds = sum_j steps_j / r_j, per paid GPU-hour of the arm. Bitwise gates: every job's 16-step losses and final adapters
in perjob/shared/co runs equal its solo run; co decode tokens equal decode solo. p99 TPOT <= 1.10x decode solo.
"""

import json
import re
import sys
from pathlib import Path

PJ = 2 * 8 * 2560 + (4096 + 1024) * 8


def log(out, name):
    p = out / f"trunk_{name}.log"
    return p.read_text(errors="replace") if p.exists() else ""


def rate(text):
    m = re.search(r"-> ([0-9.]+) steps/s", text)
    return float(m.group(1)) if m else None


def dec(text):
    m = re.search(r"-> ([0-9.]+) tok/s, tpot p50 ([0-9.]+) ms p99 ([0-9.]+) ms", text)
    return dict(tok_s=float(m.group(1)), p50_ms=float(m.group(2)), p99_ms=float(m.group(3))) if m else None


def raw(out, run, name):
    p = out / "trunk" / run / name
    return p.read_bytes() if p.exists() else None


def samples(out, name):
    p = out / f"dcgm_trunk_{name}.txt"
    return sum(1 for line in p.read_text(errors="replace").splitlines() if line.startswith(("GPU", "20"))) if p.exists() else 0


def job_bitwise(out, run, n):
    loss, P = raw(out, run, "loss.f32"), raw(out, run, "P.f32")
    if loss is None or P is None:
        return False
    for j in range(n):
        sl, sp = raw(out, f"solo16_j{j}", "loss.f32"), raw(out, f"solo16_j{j}", "P.f32")
        if sl is None or sp is None or len(loss) != 16 * n * 4:
            return False
        if any(loss[(s * n + j) * 4:(s * n + j + 1) * 4] != sl[s * 4:(s + 1) * 4] for s in range(16)) or P[j * PJ * 4:(j + 1) * PJ * 4] != sp:
            return False
    return True


def main():
    out = Path(sys.argv[1])
    solo = sorted(int(p.name[6:]) for p in (out / "trunk").glob("solo_j*")) if (out / "trunk").exists() else []
    r = {j: rate(log(out, f"solo_j{j}")) for j in solo}
    errors = [f"solo_j{j}: no rate" for j, v in r.items() if not v]
    ns = sorted({int(m.group(1)) for p in (out / "trunk").glob("shared_n*") for m in [re.match(r"shared_n(\d+)$", p.name)] if m}) \
        if (out / "trunk").exists() else []
    bs = sorted({int(p.name[5:]) for p in (out / "trunk").glob("dec_b*")}) if (out / "trunk").exists() else []
    decs = {b: dec(log(out, f"dec_b{b}")) for b in bs}
    rows, gates = [], dict(bitwise_jobs=True, bitwise_decode=True, p99_le_1p10_solo=True, one_launch=True, dcgm_100ms=True, zero_errors=True)

    def useful(n, s):
        return 3600 * sum(s / r[j] for j in range(n) if r.get(j)) if s and all(r.get(j) for j in range(n)) else None
    for n in ns:
        row = dict(jobs=n)
        for a in ("perjob", "shared"):
            s = rate(log(out, f"{a}_n{n}"))
            row[a] = dict(steps_s=s, useful_gpu_s_per_paid_gpu_hour=useful(n, s), ratio_vs_dedicated=(useful(n, s) or 0) / 3600 or None,
                          bitwise_vs_solo=job_bitwise(out, f"{a}16_n{n}", n))
            gates["bitwise_jobs"] &= row[a]["bitwise_vs_solo"]
            gates["one_launch"] &= "kernel launches 1 for" in log(out, f"{a}_n{n}")
            gates["dcgm_100ms"] &= samples(out, f"{a}_n{n}") > 0
        for b in bs:
            co, solo_d = log(out, f"co_n{n}_b{b}"), decs.get(b)
            s, d = rate(co), dec(co)
            mt, md = rate(log(out, f"mps_train_n{n}_b{b}")), dec(log(out, f"mps_dec_n{n}_b{b}"))
            toks_co, toks_solo = raw(out, f"co16_n{n}_b{b}", "toks.i32"), raw(out, f"dec_b{b}", "toks.i32")
            dec_bit = bool(toks_co) and bool(toks_solo) and toks_co[:min(len(toks_co), len(toks_solo))] == toks_solo[:min(len(toks_co), len(toks_solo))]
            p99 = d["p99_ms"] / solo_d["p99_ms"] if d and solo_d else None
            row[f"co_b{b}"] = dict(steps_s=s, useful_gpu_s_per_paid_gpu_hour=useful(n, s), ratio_vs_dedicated=(useful(n, s) or 0) / 3600 or None,
                                   decode_tok_per_paid_gpu_hour=d["tok_s"] * 3600 if d else None, p99_vs_solo=p99,
                                   train_bitwise_vs_solo=job_bitwise(out, f"co16_n{n}_b{b}", n), decode_bitwise_vs_solo=dec_bit)
            row[f"mps_b{b}"] = dict(steps_s=mt, useful_gpu_s_per_paid_gpu_hour=useful(n, mt), ratio_vs_dedicated=(useful(n, mt) or 0) / 3600 or None,
                                    decode_tok_per_paid_gpu_hour=md["tok_s"] * 3600 if md else None, p99_vs_solo=md["p99_ms"] / solo_d["p99_ms"] if md and solo_d else None)
            gates["bitwise_jobs"] &= row[f"co_b{b}"]["train_bitwise_vs_solo"]
            gates["bitwise_decode"] &= dec_bit
            gates["p99_le_1p10_solo"] &= p99 is not None and p99 <= 1.10
            gates["one_launch"] &= "kernel launches 1 for" in co
            gates["dcgm_100ms"] &= samples(out, f"co_n{n}_b{b}") > 0 and samples(out, f"mps_n{n}_b{b}") > 0
            if s is None or d is None or mt is None or md is None:
                errors.append(f"n{n} b{b}: missing co or MPS rates")
        rows.append(row)
    gates["zero_errors"] = not errors and bool(rows)
    gb = (out / "gemm_backend").read_text().strip() if (out / "gemm_backend").exists() else "mma"
    summary = dict(gemm_backend="wgmma" if gb == "wgmma" else "mma_sync", dedicated_steps_s=r, decode_solo=decs, rows=rows,
                   launch_count_source="host-issued launches (one persistent launch per run), not a profiler trace", gates=gates,
                   passed=all(gates.values()), errors=errors)
    (out / "trunk_summary.json").write_text(json.dumps(summary, indent=1))
    for row in rows:
        print(f"N={row['jobs']:3d} perjob x{row['perjob']['ratio_vs_dedicated'] or 0:.2f} shared x{row['shared']['ratio_vs_dedicated'] or 0:.2f} "
              + " ".join(f"co_b{b} x{row[f'co_b{b}']['ratio_vs_dedicated'] or 0:.2f} p99 {row[f'co_b{b}']['p99_vs_solo'] or 0:.2f} "
                         f"mps_b{b} x{row[f'mps_b{b}']['ratio_vs_dedicated'] or 0:.2f}" for b in bs))
    print(f"gates {gates} passed={summary['passed']}")
    for e in errors:
        print("ERROR", e)


if __name__ == "__main__":
    main()
