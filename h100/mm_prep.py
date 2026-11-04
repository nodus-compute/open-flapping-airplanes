"""Convert a pinned Qwen3 HF checkpoint into the flat layout llm/h100/mm/mm.cu reads (see load_model there), plus fixed prompts.

python h100/mm_prep.py --repo Qwen/Qwen3-4B --revision <sha> --out /root/data/mm/qwen3_4b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

MAXB = 32
TEXT = ("Once upon a time, in a small village at the edge of a great forest, there lived a curious girl named Lily who loved to explore. "
        "Every morning she walked past the old mill, the river and the bakery, and every evening she wrote down what she had seen in a "
        "little blue notebook that her grandmother had given her. One day she found a strange key buried under the roots of an oak tree, "
        "and she wondered which door it might open. She asked the baker, the miller and the fisherman, but nobody knew. So she decided to "
        "follow the river upstream, past the waterfall and into the hills, where the stories said an old tower stood alone in the fog.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(a.repo, revision=a.revision, allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
    cfg = AutoConfig.from_pretrained(snap)
    H, L, NH, NKV, FF, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size
    hd = getattr(cfg, "head_dim", H // NH)
    assert hd == 128 and cfg.tie_word_embeddings and cfg.rope_scaling is None, "mm.cu supports head_dim 128, tied embeddings, plain RoPE"
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok(TEXT)["input_ids"]
    stride = 2
    assert len(ids) >= (MAXB - 1) * stride + a.prompt_len, len(ids)
    prompt = np.array([ids[i * stride:i * stride + a.prompt_len] for i in range(MAXB)], dtype=np.int32)
    prompt.tofile(out / "prompt.i32")
    (out / "meta.txt").write_text(f"{H} {L} {NH} {NKV} {FF} {V} {a.prompt_len} {cfg.rms_norm_eps} {cfg.rope_theta}\n")
    expect = 2 * (V * H + H + L * (H + (NH + 2 * NKV) * hd * H + 2 * hd + H * NH * hd + H + 2 * FF * H + H * FF))
    wpath = out / "weights.bf16"
    if not (wpath.is_file() and wpath.stat().st_size == expect):
        files = sorted(Path(snap).glob("*.safetensors"))
        idx = {}
        for f in files:
            with safe_open(str(f), framework="pt") as s:
                for k in s.keys():
                    idx[k] = f

        def t(name):
            with safe_open(str(idx[name]), framework="pt") as s:
                return s.get_tensor(name).to(torch.bfloat16).contiguous()

        with open(wpath, "wb") as fo:
            def w(x):
                x.view(torch.int16).numpy().tofile(fo)
            w(t("model.embed_tokens.weight"))
            w(t("model.norm.weight"))
            for layer in range(L):
                p = f"model.layers.{layer}."
                w(t(p + "input_layernorm.weight"))
                w(torch.cat([t(p + "self_attn.q_proj.weight"), t(p + "self_attn.k_proj.weight"), t(p + "self_attn.v_proj.weight")], 0))
                w(t(p + "self_attn.q_norm.weight"))
                w(t(p + "self_attn.k_norm.weight"))
                w(t(p + "self_attn.o_proj.weight"))
                w(t(p + "post_attention_layernorm.weight"))
                g, u = t(p + "mlp.gate_proj.weight").view(FF // 64, 64, H), t(p + "mlp.up_proj.weight").view(FF // 64, 64, H)
                w(torch.stack([g, u], 1).reshape(2 * FF, H))
                w(t(p + "mlp.down_proj.weight"))
        assert wpath.stat().st_size == expect, (wpath.stat().st_size, expect)
    (out / "prep.json").write_text(json.dumps({"repo": a.repo, "revision": a.revision, "snapshot": snap, "bytes": expect,
                                               "prompt_len": a.prompt_len, "prompt_rows": MAXB}, indent=1))
    print(f"{a.repo}@{a.revision}: H {H} L {L} heads {NH}/{NKV} FF {FF} V {V} -> {wpath} ({expect / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
