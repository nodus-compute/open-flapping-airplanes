"""Real-trunk check: megakernel cut hidden (layer-L input_layernorm, Xt.bf16 from a --steps 1 run) vs HF Qwen3-4B fp32 on the same
tokens, plus HF bf16 vs fp32 as the bf16 noise floor, and PyTorch bf16 shared_prefix ms/step.
python3 h100/trunk_ckpt_ref.py --out RUN_DIR --model-dir DIR --repo R --revision V --layers L --tk T --seq S -> RUN_DIR/trunk_ckpt_ref.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

D = 2560


def cut_hidden(model, ids, layers):
    with torch.no_grad():
        hs = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[layers]
        return model.model.layers[layers].input_layernorm(hs).float().reshape(-1, D)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--out", "--model-dir", "--repo", "--revision"):
        ap.add_argument(k, required=True)
    for k in ("--layers", "--tk", "--seq"):
        ap.add_argument(k, type=int, required=True)
    a = ap.parse_args()
    run = Path(a.out)
    tok = np.fromfile(Path(a.model_dir) / "train_tokens.i32", dtype=np.int32)[: a.tk]
    ids = torch.from_numpy(tok.astype(np.int64)).view(a.tk // a.seq, a.seq).cuda()
    ours = torch.from_numpy(np.fromfile(run / "Xt.bf16", dtype=np.int16)).view(torch.bfloat16).float().view(a.tk, D).cuda()
    snap = snapshot_download(a.repo, revision=a.revision, allow_patterns=["*.safetensors", "*.json"])
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32).cuda().eval()
    ref = cut_hidden(model, ids, a.layers)
    model = model.to(torch.bfloat16)
    hf_bf16 = cut_hidden(model, ids, a.layers)

    def rel(x):
        return float((x - ref).norm() / ref.norm())

    model.model.layers = model.model.layers[: a.layers]
    with torch.no_grad():
        for _ in range(3):
            model.model(input_ids=ids, use_cache=False)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            model.model(input_ids=ids, use_cache=False)
        e1.record()
        torch.cuda.synchronize()
    r_ours, r_bf16 = rel(ours), rel(hf_bf16)
    out = dict(layers=a.layers, tokens=a.tk, seq=a.seq, rel_err_megakernel_vs_hf_fp32=r_ours, rel_err_hf_bf16_vs_hf_fp32=r_bf16,
               pytorch_bf16_trunk_ms_per_step=e0.elapsed_time(e1) / 10, gate_pass=r_ours <= max(1e-2, 1.5 * r_bf16))
    (run / "trunk_ckpt_ref.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
