"""Token rows for the real-model trunk: python3 h100/trunk_prep.py --repo R --revision V --model-dir DIR --tokens N -> DIR/train_tokens.i32."""

import argparse
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from mm_prep import TEXT
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    a = ap.parse_args()
    snap = snapshot_download(a.repo, revision=a.revision, allow_patterns=["*.json", "merges.txt", "vocab.json"])
    ids = AutoTokenizer.from_pretrained(snap)(TEXT)["input_ids"]
    np.resize(np.array(ids, dtype=np.int32), a.tokens).tofile(Path(a.model_dir) / "train_tokens.i32")
    print(f"wrote {a.tokens} tokens ({len(ids)}-token text repeated)")


if __name__ == "__main__":
    main()
