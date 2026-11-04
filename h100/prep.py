"""H100 data prep (same recipe as b200-run/modal_app.py prep()): TinyStories -> cl100k u32, starts, vocab, prompt ids, init weights.
Runs inside the pinned PyTorch image with HOME=/root and DATA_DIR=/root/data (= ~/data, where llm/b200/baseline/prep_data.py looks)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL, MK = f"{ROOT}/llm/b200/baseline", f"{ROOT}/llm/b200/megakernel"
DATA = os.path.expanduser(os.environ.get("DATA_DIR", "~/data"))
HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FA_PROMPT = "What is a megakernel?"
DOWNLOAD_TIMEOUT = 600
DOWNLOAD_RETRIES = 3
DOWNLOAD_CHUNK_SIZE = 1 << 20
TRAIN_PREFIX_BYTES = 314572800
USER_AGENT = "open-flapping-airplanes-h100-prep/1.0"


def sh(cmd, cwd=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, executable="/bin/bash")


def fetch(url, out, byte_range=None):
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes=0-{byte_range - 1}"
    part = f"{out}.part"

    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            print("+ fetch", url, flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                status = response.getcode()
                if byte_range is not None and status not in (200, 206):
                    raise RuntimeError(f"expected HTTP 206 or 200 for ranged download, got {status}")
                if byte_range is None and (status is None or status < 200 or status >= 300):
                    raise RuntimeError(f"expected successful download, got HTTP {status}")
                content_range = response.headers.get("Content-Range")
                if byte_range is not None and status == 206 and content_range:
                    expected_range = f"bytes 0-{byte_range - 1}/"
                    if not content_range.startswith(expected_range):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")

                written = 0
                with open(part, "wb") as output:
                    while byte_range is None or written < byte_range:
                        size = DOWNLOAD_CHUNK_SIZE
                        if byte_range is not None:
                            size = min(size, byte_range - written)
                        block = response.read(size)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)

            if byte_range is not None and written != byte_range:
                raise RuntimeError(f"expected {byte_range} bytes, received {written}")
            os.replace(part, out)
            return
        except Exception as exc:
            try:
                os.remove(part)
            except FileNotFoundError:
                pass
            if attempt == DOWNLOAD_RETRIES:
                raise RuntimeError(f"download failed after {attempt + 1} attempts: {url}") from exc
            time.sleep(2**attempt)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(init_path):
    files = {}
    for name in (
        "tinystories_cl100k_train.u32",
        "starts_4000.u32",
        "cl100k_vocab.bin",
        "prompt_ids.txt",
    ):
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"required prep output is missing: {path}")
        if os.path.getsize(path) == 0:
            raise RuntimeError(f"required prep output is empty: {path}")
        files[name] = {"size_bytes": os.path.getsize(path), "sha256": sha256(path)}

    if not os.path.isfile(init_path):
        raise RuntimeError(f"required prep output is missing: {init_path}")
    if os.path.getsize(init_path) == 0:
        raise RuntimeError(f"required prep output is empty: {init_path}")
    files[os.path.basename(init_path)] = {
        "size_bytes": os.path.getsize(init_path),
        "sha256": sha256(init_path),
    }

    checksum_path = f"{init_path}.sha256"
    with open(checksum_path, "w", encoding="utf-8") as checksum_file:
        checksum_file.write(f"{files[os.path.basename(init_path)]['sha256']}  {os.path.basename(init_path)}\n")

    manifest_path = os.path.join(DATA, "prep_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({"files": files}, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def main():
    os.makedirs(DATA, exist_ok=True)
    valid, train = f"{DATA}/TinyStories-valid.txt", f"{DATA}/TinyStories-train-prefix.txt"
    if not os.path.exists(valid) or os.path.getsize(valid) < 1e7:
        fetch(f"{HF}/TinyStories-valid.txt", valid)
        if os.path.getsize(valid) < 1e7:
            raise RuntimeError(f"downloaded validation data is too small: {valid}")
    if not os.path.exists(train) or os.path.getsize(train) < 3e8:
        fetch(f"{HF}/TinyStories-train.txt", train, byte_range=TRAIN_PREFIX_BYTES)
        if os.path.getsize(train) < 3e8:
            raise RuntimeError(f"downloaded training prefix is too small: {train}")
    train_u32 = f"{DATA}/tinystories_cl100k_train.u32"
    if not os.path.exists(train_u32):
        sh(f"{sys.executable} prep_data.py", cwd=BL)
    if not os.path.isfile(train_u32):
        raise RuntimeError(f"required prep output is missing: {train_u32}")
    if os.path.getsize(train_u32) == 0:
        raise RuntimeError(f"required prep output is empty: {train_u32}")
    ntok = os.path.getsize(train_u32) // 4
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
    flat = torch.cat([x.reshape(-1) for x in parts])
    expected_elements = sum(p.numel() for p in parts)
    assert flat.numel() == expected_elements
    assert torch.isfinite(flat).all().item()
    init_path = f"{DATA}/init_flat_S.f32"
    flat.numpy().astype("float32").tofile(init_path)
    assert os.path.isfile(init_path)
    init_size = os.path.getsize(init_path)
    assert init_size == 4 * flat.numel()
    assert init_size // 4 == expected_elements
    write_manifest(init_path)
    print("prep ok, train tokens", ntok)


if __name__ == "__main__":
    main()
