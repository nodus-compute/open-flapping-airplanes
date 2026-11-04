"""60 s in-run preflight for the real-model OFA chain (trunk_ckpt -> trunk_tail -> codec), run before any arm.

python3 h100/preflight.py --out OUT --gpu G --img PT_IMG [...] -> OUT/preflight.json; exit 0 only if every check passes.
Checks: H100 80GB HBM3, driver >= 580, free MiB, no other compute apps, MPS per device (no nvidia-cuda-mps-server on OUR GPU UUID,
no CUDA_MPS_* in our env other than the private pipe, private --mps-pipe dir exists and is empty, compute mode Default; another
tenant's MPS control daemon on a different GPU is recorded, not failed), pinned image present
locally, a real CUDA matmul + nvcc inside the image on this GPU, PREFIX_RUN config/summary/dedicated-0..J-1,
train.jsonl, Qwen3-4B revision reachable (HF cache snapshot or HTTP), and free disk under DATA and OUT.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.monotonic()


def sh(argv, timeout):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=max(1.0, timeout))
        return p.returncode, (p.stdout + p.stderr).strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return 124, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--sudo", default="")
    ap.add_argument("--deadline-s", type=float, default=60.0)
    ap.add_argument("--min-free-mib", type=int, default=70000)
    ap.add_argument("--min-driver-major", type=int, default=580)
    ap.add_argument("--gpu-name", default="H100 80GB HBM3")
    ap.add_argument("--prefix-run", default="")
    ap.add_argument("--jobs", type=int, default=0, help="number of PREFIX_RUN dedicated-<j> arms required")
    ap.add_argument("--train-data", default="")
    ap.add_argument("--hf-cache", default="")
    ap.add_argument("--repo", default="Qwen/Qwen3-4B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--mps-pipe", default="", help="private empty CUDA_MPS_PIPE_DIRECTORY our processes run with")
    ap.add_argument("--offline", action="store_true", help="GPU run without network: HF snapshot must be complete in the cache, --py-deps staged")
    ap.add_argument("--py-deps", default="", help="pip --target dir from h100/stage_offline.sh (offline prep deps)")
    ap.add_argument("--min-disk-gib", type=float, default=30.0)
    a = ap.parse_args()
    left = lambda: a.deadline_s - (time.monotonic() - T0)  # noqa: E731
    sudo = a.sudo.split()
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)[-400:], "t_s": round(time.monotonic() - T0, 2)}

    rc, o = sh(["nvidia-smi", "-i", a.gpu, "--query-gpu=name,driver_version,memory.free", "--format=csv,noheader,nounits"], 10)
    f = [x.strip() for x in o.split(",")] if rc == 0 else []
    check("gpu_name", len(f) == 3 and a.gpu_name in f[0], o)
    check("driver_major", len(f) == 3 and f[1].split(".")[0].isdigit() and int(f[1].split(".")[0]) >= a.min_driver_major, o)
    check("free_mib", len(f) == 3 and f[2].isdigit() and int(f[2]) >= a.min_free_mib, o)
    rc, o = sh(["nvidia-smi", "-i", a.gpu, "--query-compute-apps=pid", "--format=csv,noheader"], 10)
    check("gpu_idle", rc == 0 and not o, o)
    rc, o = sh(["nvidia-smi", "-i", a.gpu, "--query-gpu=uuid,compute_mode", "--format=csv,noheader"], 10)
    g = [x.strip() for x in o.split(",")] if rc == 0 else []
    uuid = g[0] if len(g) == 2 and g[0].startswith("GPU-") else ""
    check("compute_mode_default", len(g) == 2 and g[1] == "Default", o)
    rc, o = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], 10)
    mine = [ln for ln in o.splitlines() if uuid and ln.split(",")[0].strip() == uuid] if rc == 0 else []
    check("no_mps_server_on_gpu", rc == 0 and bool(uuid) and not any("nvidia-cuda-mps-server" in ln for ln in mine), {"uuid": uuid, "apps_on_gpu": mine})
    pipe = os.path.realpath(a.mps_pipe) if a.mps_pipe else ""
    env_bad = {k: v for k, v in os.environ.items() if k.startswith("CUDA_MPS_") and not (k == "CUDA_MPS_PIPE_DIRECTORY" and pipe and os.path.realpath(v) == pipe)}
    check("no_mps_env", not env_bad, env_bad)
    check("private_mps_pipe_empty", bool(pipe) and os.path.isdir(pipe) and not os.listdir(pipe), pipe)
    rc, o = sh(["pgrep", "-af", "nvidia-cuda-mps-control"], 5)
    host_mps = o if rc == 0 else ""

    rc, o = sh(sudo + ["docker", "image", "inspect", "--format", "{{.Id}}", a.img], 15)
    check("image_local", rc == 0, o)
    if rc == 0:
        py = ("import torch,subprocess;x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16);"
              "torch.cuda.synchronize();y=(x@x).float().abs().sum().item();torch.cuda.synchronize();"
              "print('MATMUL_OK',torch.cuda.get_device_name(0),y==y);"
              "print(subprocess.run(['nvcc','--version'],capture_output=True,text=True).stdout.splitlines()[-1])")
        mps = ["-e", "CUDA_MPS_PIPE_DIRECTORY=/mps_private", "-v", f"{pipe}:/mps_private:ro"] if pipe else []
        if a.offline:
            mps += ["--network", "none"]
        rc, o = sh(sudo + ["docker", "run", "--rm", "--gpus", f'"device={a.gpu}"', *mps, "--entrypoint", "python", a.img, "-c", py],
                   left() - 2)
        check("container_cuda_matmul_nvcc", rc == 0 and "MATMUL_OK" in o and "True" in o and "cuda" in o.lower(), o)
    else:
        check("container_cuda_matmul_nvcc", False, "image not present locally (pull would not fit in the deadline)")

    if a.prefix_run:
        p = Path(a.prefix_run)
        miss = [n for n in ["config.json", "summary.json"] if not (p / n).is_file()]
        miss += [f"dedicated-{j}" for j in range(a.jobs) if not (p / f"dedicated-{j}").is_dir()]
        check("prefix_run", p.is_dir() and not miss, f"{p} missing={miss[:8]}")
    else:
        check("prefix_run", False, "PREFIX_RUN unset")
    if a.train_data:
        t = Path(a.train_data) / "train.jsonl"
        check("train_jsonl", t.is_file() and t.stat().st_size > 0, t)

    if a.revision:
        snap = Path(a.hf_cache) / "hub" / f"models--{a.repo.replace('/', '--')}" / "snapshots" / a.revision / "config.json"
        if a.offline:
            d = snap.parent
            try:
                w = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())
            except (OSError, ValueError, KeyError):
                w = None
            check("hf_revision_offline_complete", bool(w) and snap.is_file() and (d / "tokenizer.json").is_file() and all((d / f).is_file() for f in w), d)
            deps = Path(a.py_deps)
            check("py_deps_offline", all((deps / m / "__init__.py").is_file() for m in ("transformers", "safetensors", "huggingface_hub", "numpy")), deps)
        elif snap.is_file():
            check("hf_revision", True, snap)
        else:
            url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/config.json"
            hdr = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"] if os.environ.get("HF_TOKEN") else []
            rc, o = sh(["curl", "-sfIL", "-m", str(int(max(2, min(10, left() - 1)))), *hdr, "-o", "/dev/null", "-w", "%{http_code}", url], 12)
            check("hf_revision", rc == 0 and o.endswith("200"), f"{url} rc={rc} {o}")

    for name, d in [("disk_data", a.data), ("disk_out", a.out)]:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            g = shutil.disk_usage(d).free / 2**30
            check(name, g >= a.min_disk_gib, f"{d} free={g:.1f}GiB")

    el = time.monotonic() - T0
    check("within_deadline", el <= a.deadline_s, f"{el:.1f}s <= {a.deadline_s}s")
    passed = all(c["ok"] for c in checks.values())
    s = {"passed": passed, "elapsed_s": round(el, 2), "deadline_s": a.deadline_s, "gpu": a.gpu, "gpu_uuid": uuid, "mps_pipe_dir": pipe, "offline": a.offline, "host_mps_control_daemons": host_mps, "image": a.img,
         "failed": [k for k, c in checks.items() if not c["ok"]], "checks": checks}
    Path(a.out, "preflight.json").write_text(json.dumps(s, indent=1))
    print(f"preflight passed={passed} elapsed={el:.1f}s failed={s['failed']}")
    sys.exit(0 if passed else 7)


if __name__ == "__main__":
    main()
