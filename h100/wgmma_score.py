"""Score the wgmma GEMM microbenchmark: python3 h100/wgmma_score.py OUT -> OUT/wgmma_summary.json (+ table on stdout).

Inputs: OUT/wgmma_cases.jsonl (one line per case and backend from test_gemm_r2.cu with GEMM_JSON), OUT/dcgm_wgmma_*.txt.
Backend 0 = OFA mma.sync, 3 = OFA wgmma/TMA; cuBLAS is timed beside each.
"""

import json
import sys
from pathlib import Path

NAMES = {0: "mma_sync", 3: "wgmma"}


def dcgm_mean(path):
    vals = []
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        parts = line.replace(",", " ").split()
        if parts and parts[0] == "GPU" and len(parts) >= 6:
            try:
                vals.append([float(x) for x in parts[2:6]])
            except ValueError:
                pass
        elif parts and parts[0][:2] == "20":  # nvidia-smi -lms 100 fallback: timestamp,util.gpu,util.mem,clock
            vals.append(None)
    if vals and all(v is None for v in vals):
        return dict(samples=len(vals), source="nvidia-smi")
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(samples=0)
    return dict(samples=len(vals), sm_active=sum(v[0] for v in vals) / len(vals), sm_occupancy=sum(v[1] for v in vals) / len(vals),
                tensor_active=sum(v[2] for v in vals) / len(vals), dram_active=sum(v[3] for v in vals) / len(vals))


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(line) for line in (out / "wgmma_cases.jsonl").read_text().splitlines() if line.strip()] \
        if (out / "wgmma_cases.jsonl").exists() else []
    cases, errors = {}, []
    for r in rows:
        cases.setdefault(r["case"].strip(), {})[NAMES.get(r["backend"], str(r["backend"]))] = r
    table = []
    for name, by in sorted(cases.items()):
        row = dict(case=name)
        for b in NAMES.values():
            r = by.get(b)
            if r is None:
                errors.append(f"{name}: missing {b}")
                continue
            row[b] = dict(M=r["M"], N=r["N"], K=r["K"], tflops=r["ours_tflops"], gbs=r["ours_gbs"], frac_of_cublas=r["frac_of_cublas"],
                          bitwise_same_tile=r["persist_bitwise"] == "bitexact", within_tol=r["pass"], relmax=r["relmax"])
            row["cublas"] = dict(tflops=r["cublas_tflops"], gbs=r["cublas_gbs"], relmax=r["cublas_relmax"])
            if not r["pass"]:
                errors.append(f"{name}: {b} FAIL (relmax {r['relmax']:.2e}, bitwise {r['persist_bitwise']})")
        table.append(row)
    wg = [row["wgmma"]["frac_of_cublas"] for row in table if "wgmma" in row]
    gates = dict(
        all_cases_present=bool(table) and not any("missing" in e for e in errors),
        bitwise_same_tile_each_backend=bool(table) and all(row[b]["bitwise_same_tile"] for row in table for b in NAMES.values() if b in row),
        within_tol_each_backend=bool(table) and all(row[b]["within_tol"] for row in table for b in NAMES.values() if b in row),
        cublas_within_bf16_tol=bool(table) and all(row["cublas"]["relmax"] <= 2e-2 for row in table if "cublas" in row),
        dcgm_100ms=all(dcgm_mean(out / f"dcgm_wgmma_{b}.txt")["samples"] > 0 for b in ("mma", "wg")),
    )
    min_wg = min(wg) if wg else None
    # 0.9x cuBLAS on every shape (correct and bitwise) -> wgmma becomes the default for the mw/sched phases (GEMM=auto).
    choice = "wgmma" if min_wg is not None and min_wg >= 0.9 and gates["bitwise_same_tile_each_backend"] and gates["within_tol_each_backend"] else "mma"
    (out / "gemm_choice").write_text(choice + "\n")
    summary = dict(cases=table, min_wgmma_frac_of_cublas=min_wg,
                   min_mma_frac_of_cublas=min((row["mma_sync"]["frac_of_cublas"] for row in table if "mma_sync" in row), default=None),
                   dcgm={b: dcgm_mean(out / f"dcgm_wgmma_{b}.txt") for b in ("mma", "wg")}, gemm_choice=choice, gates=gates,
                   passed=all(gates.values()) and not errors, errors=errors)
    (out / "wgmma_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'case':28s} {'M x N x K':20s} | {'mma TF':>7s} {'GB/s':>7s} {'frac':>5s} | {'wg TF':>7s} {'GB/s':>7s} {'frac':>5s} | {'cuBLAS TF':>9s}")
    for row in table:
        m, w, c = row.get("mma_sync", {}), row.get("wgmma", {}), row.get("cublas", {})
        shape = f"{(m or w).get('M')}x{(m or w).get('N')}x{(m or w).get('K')}"
        print(f"{row['case']:28s} {shape:20s} | {m.get('tflops', 0):7.1f} {m.get('gbs', 0):7.0f} {m.get('frac_of_cublas', 0):5.2f} | "
              f"{w.get('tflops', 0):7.1f} {w.get('gbs', 0):7.0f} {w.get('frac_of_cublas', 0):5.2f} | {c.get('tflops', 0):9.1f}")
    print(f"gemm_choice={choice} min_wgmma_frac={min_wg} passed={summary['passed']}")
    for e in errors:
        print("ERROR", e)


if __name__ == "__main__":
    main()
