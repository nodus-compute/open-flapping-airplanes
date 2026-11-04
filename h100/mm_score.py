"""Score the mm arm: python3 h100/mm_score.py OUT  ->  OUT/mm_summary.json (+ table on stdout).

Fused rates are measured inside the window where BOTH models are generating (GPU globaltimer stamps); vLLM-under-MPS rates likewise
inside the window where both servers stream. Normalised work U = sum over models of rate / that model's dedicated-GPU rate
(2 dedicated GPUs = 2.0 GPU-equivalents on 2 paid GPUs, i.e. 1.0 per paid GPU).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sched_score import dcgm

P99_MAX = 1.10


def res(d):
    p = d / "result.json"
    return json.loads(p.read_text()) if p.is_file() else None


def same(a, b, n):
    return (a / n).is_file() and (b / n).is_file() and (a / n).read_bytes() == (b / n).read_bytes()


def co_rates(d, r, b):
    ms = r["models"]
    lo, hi = max(ms["0"]["start_ns"], ms["1"]["start_ns"]), min(ms["0"]["end_ns"], ms["1"]["end_ns"])
    if hi <= lo:
        return None, None, 0.0
    out = []
    for m in ("0", "1"):
        td = np.fromfile(d / f"tok_done{m}.u64", dtype=np.uint64)[ms[m]["P"]:]
        out.append(int(((td >= lo) & (td <= hi)).sum()) * b / ((hi - lo) * 1e-9))
    return out[0], out[1], (hi - lo) * 1e-9


def main(out):
    out = Path(out)
    mm = out / "mm"
    vl = json.loads((out / "mm_vllm.json").read_text())["rows"] if (out / "mm_vllm.json").is_file() else []
    vded = {(r["model"], r["b"]): r for r in vl if r["concurrent_models"] == 1}
    vmps = {(r["model"], r["b"]): r for r in vl if r["concurrent_models"] == 2}
    rows, gates = [], {}
    for b in sorted({int(p.name.split("_b")[1]) for p in mm.glob("co_b*")}):
        s0, s1, co = res(mm / f"solo0_b{b}"), res(mm / f"solo1_b{b}"), res(mm / f"co_b{b}")
        if not (s0 and s1 and co):
            gates[f"b{b}_all_arms_ran"] = False
            continue
        r0, r1, win = co_rates(mm / f"co_b{b}", co, b)
        d0, d1 = s0["models"]["0"]["tok_s"], s1["models"]["1"]["tok_s"]
        row = {"b": b, "solo_0p6b_tok_s": d0, "solo_4b_tok_s": d1, "fused_0p6b_tok_s": r0, "fused_4b_tok_s": r1, "fused_overlap_s": win}
        row["fused_tok_per_gpu_hour"] = (r0 + r1) * 3600 if r0 and r1 else None
        row["dedicated_2gpu_tok_per_gpu_hour"] = (d0 + d1) * 3600 / 2
        row["fused_vs_2_dedicated_tokens"] = row["fused_tok_per_gpu_hour"] / row["dedicated_2gpu_tok_per_gpu_hour"] if r0 and r1 else None
        row["fused_U_gpu_equiv"] = r0 / d0 + r1 / d1 if r0 and r1 else None
        for m, s in (("0", s0), ("1", s1)):
            tag = "0p6b" if m == "0" else "4b"
            bit = same(mm / f"solo{m}_b{b}", mm / f"co_b{b}", f"toks{m}.i32")
            p99r = co["models"][m]["tpot_p99_ms"] / s["models"][m]["tpot_p99_ms"] if s["models"][m]["tpot_p99_ms"] else None
            row[f"{tag}_tokens_bitwise_vs_solo"] = bit
            row[f"{tag}_p99_vs_solo"] = p99r
            gates[f"b{b}_{tag}_tokens_bitwise_vs_solo"] = bit
            gates[f"b{b}_{tag}_p99_within_{P99_MAX}x_solo"] = p99r is not None and p99r <= P99_MAX
            gates[f"b{b}_{tag}_zero_errors"] = all(x["done"] == x["steps"] for x in (s["models"][m], co["models"][m]))
        v0, v1, m0, m1 = vded.get(("m0", b)), vded.get(("m1", b)), vmps.get(("m0", b)), vmps.get(("m1", b))
        if v0 and v1 and m0 and m1 and m0["tok_s"] and m1["tok_s"]:
            lo, hi = max(m0["all_rows_start_s"], m1["all_rows_start_s"]), min(m0["all_rows_end_s"], m1["all_rows_end_s"])
            row.update({"vllm_dedicated_0p6b_tok_s": v0["tok_s"], "vllm_dedicated_4b_tok_s": v1["tok_s"],
                        "vllm_mps_0p6b_tok_s": m0["tok_s"], "vllm_mps_4b_tok_s": m1["tok_s"], "vllm_mps_streams_overlap_s": hi - lo,
                        "vllm_mps_tok_per_gpu_hour": (m0["tok_s"] + m1["tok_s"]) * 3600,
                        "vllm_mps_U_gpu_equiv": m0["tok_s"] / v0["tok_s"] + m1["tok_s"] / v1["tok_s"],
                        "vllm_mps_p99_vs_dedicated_0p6b": m0["tpot_p99_ms"] / v0["tpot_p99_ms"] if v0["tpot_p99_ms"] else None,
                        "vllm_mps_p99_vs_dedicated_4b": m1["tpot_p99_ms"] / v1["tpot_p99_ms"] if v1["tpot_p99_ms"] else None})
            if row["fused_tok_per_gpu_hour"]:
                row["fused_vs_vllm_mps_tokens"] = row["fused_tok_per_gpu_hour"] / row["vllm_mps_tok_per_gpu_hour"]
                row["fused_U_vs_vllm_mps_U"] = row["fused_U_gpu_equiv"] / row["vllm_mps_U_gpu_equiv"]
            gates[f"b{b}_vllm_zero_errors"] = not any(x["errors"] for x in (v0, v1, m0, m1))
        row["dcgm"] = {a: dcgm(out, f"mm_{a}_b{b}") for a in ("solo0", "solo1", "co")}
        rows.append(row)
    for p in sorted(out.glob("mm_ref_*.json")):
        r = json.loads(p.read_text())
        gates[f"hf_tolerance_{p.stem}"] = r["pass"]
    summary = {"rows": rows, "gates": gates, "all_gates_pass": bool(gates) and all(gates.values()), "dcgm_vllm_mps": dcgm(out, "mm_vllm_mps")}
    (out / "mm_summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["b", "solo_0p6b_tok_s", "solo_4b_tok_s", "fused_0p6b_tok_s", "fused_4b_tok_s", "fused_vs_2_dedicated_tokens", "fused_U_gpu_equiv",
            "vllm_mps_U_gpu_equiv", "fused_vs_vllm_mps_tokens", "0p6b_p99_vs_solo", "4b_p99_vs_solo"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(f"{r.get(k):.4g}" if isinstance(r.get(k), float) else str(r.get(k)) for k in keys))
    print("gates:", json.dumps(gates))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
