"""H100 data prep (same recipe as b200-run/modal_app.py prep()): TinyStories -> cl100k u32, starts, vocab, prompt ids, init weights.
Runs inside the pinned PyTorch image with HOME=/root and DATA_DIR=/root/data (= ~/data, where llm/b200/baseline/prep_data.py looks)."""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL, MK = f"{ROOT}/llm/b200/baseline", f"{ROOT}/llm/b200/megakernel"
DATA = os.path.expanduser(os.environ.get("DATA_DIR", "~/data"))
HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FA_PROMPT = "What is a megakernel?"


def sh(cmd, cwd=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, executable="/bin/bash")


def main():
    os.makedirs(DATA, exist_ok=True)
    valid, train = f"{DATA}/TinyStories-valid.txt", f"{DATA}/TinyStories-train-prefix.txt"
    if not os.path.exists(valid) or os.path.getsize(valid) < 1e7:
        sh(f"curl -sfL -o {valid} {HF}/TinyStories-valid.txt")
    if not os.path.exists(train) or os.path.getsize(train) < 3e8:
        sh(f"curl -sfL -r 0-314572799 -o {train} {HF}/TinyStories-train.txt")
    if not os.path.exists(f"{DATA}/tinystories_cl100k_train.u32"):
        sh(f"{sys.executable} prep_data.py", cwd=BL)
    ntok = os.path.getsize(f"{DATA}/tinystories_cl100k_train.u32") // 4
    np.random.default_rng(1234).integers(0, ntok - 512 - 1, size=(4000, 8)).astype(np.uint32).tofile(f"{DATA}/starts_4000.u32")
    sh(f"{sys.executable} make_vocab.py {DATA}/cl100k_vocab.bin {DATA}/prompt_ids.txt 'Once upon a time'", cwd=MK)
    sh(f"{sys.executable} make_vocab.py /tmp/vocab_unused.bin {DATA}/prompt_ids_fa.txt '{FA_PROMPT}'", cwd=MK)
    os.environ.update(DATA_DIR=DATA, MODEL="S")
    sys.path.insert(0, BL)
    import common
    import torch

    w = common.init_weights()
    parts = [w["wte"], w["wpe"]]
    for layer in range(common.L):
        p = f"L{layer:02d}_"
        parts += [w[p + "gamma1"], w[p + "Wqkv"], w[p + "Wo"], w[p + "gamma2"], w[p + "Wup"], w[p + "Wdown"]]
    parts.append(w["gamma_f"])
    torch.cat([x.reshape(-1) for x in parts]).numpy().astype("float32").tofile(f"{DATA}/init_flat_S.f32")
    print("prep ok, train tokens", ntok)


if __name__ == "__main__":
    main()
