"""Score the mlora arm: python3 h100/mlora_score.py OUT  ->  OUT/ml_summary.json (+ table on stdout)."""
import json
import sys
from pathlib import Path


def same_bytes(a, b, names):
    return all((a / n).is_file() and (b / n).is_file() and (a / n).read_bytes() == (b / n).read_bytes() for n in names)


def ratio(x, y):
    return x / y if x and y else None


def main(out):
    out = Path(out)
    ml = out / "ml"
    mk = {}
    for p in sorted(ml.glob("*/result.json")):
        r = json.loads(p.read_text())
        r["dir"] = p.parent
        if p.parent.name.startswith("ref_"):
            continue
        mk[(r["lora"], r["b"], r["adapters"])] = r
    load = lambda f: json.loads((out / f).read_text()) if (out / f).is_file() else {"rows": []}  # noqa: E731
    torch_rows, vllm = load("ml_torch.json")["rows"], load("ml_vllm.json")
    tr = {(r["system"], r["b"], r["adapters"]): r for r in torch_rows}
    vr = {(r["b"], r["adapters"]): r for r in vllm["rows"]}
    refs = [json.loads(p.read_text()) for p in sorted(out.glob("ml_ref_*.json"))]
    rows, gates = [], {}
    for (lora, b, na), s in sorted(mk.items()):
        if lora != "sgmv":
            continue
        loop, none = mk.get(("loop", b, na)), mk.get(("none", b, 0))
        bit = bool(loop) and same_bytes(s["dir"], loop["dir"], ["toks.i32", "lsum.f32", "logits0.f32"])
        changed = bool(none) and not same_bytes(s["dir"], none["dir"], ["logits0.f32"])
        ok = s["done"] == s["dec_tokens"] and (not loop or loop["done"] == loop["dec_tokens"])
        tl, tg = tr.get(("pytorch_eager_loop", b, na), {}), tr.get(("pytorch_eager_gather", b, na), {})
        v, v0 = vr.get((b, na), {}), vr.get((b, 0), {})
        row = {
            "b": b, "adapters": na,
            "ofa_none_tok_s": none and none["tok_s"], "ofa_loop_tok_s": loop and loop["tok_s"], "ofa_sgmv_tok_s": s["tok_s"],
            "ofa_sgmv_tok_per_gpu_hour": s["tok_per_gpu_hour"], "ofa_sgmv_p99_ms": s["tpot_p99_ms"],
            "sgmv_vs_loop": ratio(s["tok_s"], loop and loop["tok_s"]), "sgmv_vs_none": ratio(s["tok_s"], none and none["tok_s"]),
            "pytorch_eager_loop_tok_s": tl.get("tok_s"), "pytorch_eager_gather_tok_s": tg.get("tok_s"),
            "sgmv_vs_pytorch_loop": ratio(s["tok_s"], tl.get("tok_s")), "sgmv_vs_pytorch_gather": ratio(s["tok_s"], tg.get("tok_s")),
            "vllm_stock_tok_s_full_36_layer_model": v.get("tok_s"), "vllm_stock_lora_vs_no_adapter": ratio(v.get("tok_s"), v0.get("tok_s")),
            "bitwise_sgmv_vs_loop": bit, "lora_changes_output": changed, "zero_errors": ok,
        }
        rows.append(row)
        gates[f"b{b}_a{na}_bitwise_sgmv_vs_same_tile_loop"] = bit
        gates[f"b{b}_a{na}_lora_changes_output"] = changed
        gates[f"b{b}_a{na}_all_tokens_decoded"] = ok
    for r in refs:
        gates[f"pytorch_tolerance_{r['lora']}_b{r['b']}_a{r['adapters']}"] = r["pass"]
    summary = {
        "note": "OFA arms run ONE Qwen3-4B-shaped decoder layer + lm_head (hash weights); vLLM runs the full 36-layer Qwen3-4B, so raw "
                "tok/s is not like-for-like; compare sgmv_vs_none against vllm_stock_lora_vs_no_adapter for LoRA overhead.",
        "vllm_version": vllm.get("vllm"), "rows": rows, "gates": gates, "all_gates_pass": bool(gates) and all(gates.values()),
        "pytorch_refs": refs,
    }
    summary["gemm_backend"] = "cuda_core_gemv"
    (out / "ml_summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["b", "adapters", "ofa_none_tok_s", "ofa_loop_tok_s", "ofa_sgmv_tok_s", "sgmv_vs_loop", "sgmv_vs_none", "sgmv_vs_pytorch_loop",
            "sgmv_vs_pytorch_gather", "vllm_stock_lora_vs_no_adapter", "bitwise_sgmv_vs_loop"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    print("gates:", json.dumps(gates))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
