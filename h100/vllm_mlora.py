"""Stock vLLM multi-LoRA decode throughput (theirs, unmodified) on the full Qwen3-4B, for the mlora arm.

Writes random rank-8 q_proj/v_proj PEFT adapters, then for each batch size B and adapter count A (row i -> adapter i % A) measures
decode tok/s = B * T / (t(prefill + T+1 tokens) - t(prefill + 1 token)) with greedy, ignore_eos, fixed 1024-token random prompts.
A = 0 runs the same LoRA-enabled engine without lora_request.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def make_adapters(root, n, layers, hidden, q_out, v_out, rank):
    paths = []
    for a in range(n):
        p = Path(root) / f"a{a}"
        p.mkdir(parents=True, exist_ok=True)
        g = torch.Generator().manual_seed(1000 + a)
        t = {}
        for layer in range(layers):
            for m, out in (("q_proj", q_out), ("v_proj", v_out)):
                pre = f"base_model.model.model.layers.{layer}.self_attn.{m}"
                t[f"{pre}.lora_A.weight"] = (torch.rand(rank, hidden, generator=g) * 2 - 1) * (3 / hidden) ** 0.5
                t[f"{pre}.lora_B.weight"] = (torch.rand(out, rank, generator=g) * 2 - 1) * 0.01
        save_file({k: v.to(torch.bfloat16).contiguous() for k, v in t.items()}, str(p / "adapter_model.safetensors"))
        (p / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": 2 * rank, "lora_dropout": 0.0, "bias": "none",
            "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": "Qwen/Qwen3-4B"}))
        paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64])
    ap.add_argument("--adapters", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--adapter-dir", default="/root/data/mlora_adapters")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = AutoConfig.from_pretrained(a.model)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    paths = make_adapters(a.adapter_dir, max(a.adapters), cfg.num_hidden_layers, cfg.hidden_size,
                          cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd, 8)
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=8, max_loras=max(a.adapters), max_cpu_loras=max(a.adapters),
              max_num_seqs=max(a.batches), max_model_len=a.ctx + a.tokens + 64, gpu_memory_utilization=0.85, seed=0)
    loras = [LoRARequest(f"a{i}", i + 1, p) for i, p in enumerate(paths)]
    rng = random.Random(0)
    rows = []
    for b in a.batches:
        prompts = [{"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(a.ctx)]} for _ in range(b)]
        for na in [0] + [n for n in a.adapters if n <= b]:
            req = [loras[i % na] for i in range(b)] if na else None

            def run(n):
                sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sp, lora_request=req, use_tqdm=False)
                dt = time.perf_counter() - t0
                assert all(len(o.outputs[0].token_ids) == n for o in outs), "short generation"
                return dt
            run(a.tokens + 1)                                  # warm-up: loads adapters, compiles graphs
            t1, tf = min(run(1) for _ in range(2)), min(run(a.tokens + 1) for _ in range(2))
            tps = b * a.tokens / (tf - t1)
            r = {"system": "vllm_stock_multilora" if na else "vllm_stock_lora_engine_no_adapter", "b": b, "adapters": na, "tok_s": tps,
                 "tok_per_gpu_hour": tps * 3600, "tpot_mean_ms": (tf - t1) * 1e3 / a.tokens}
            print(json.dumps(r), flush=True)
            rows.append(r)
    import vllm
    Path(a.out).write_text(json.dumps({"vllm": vllm.__version__, "model": a.model, "gpu": torch.cuda.get_device_name(), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
