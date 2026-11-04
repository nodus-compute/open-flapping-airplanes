"""Read-only A/B scorer for two codec bundles: python3 h100/ab_verify.py BASE NEW -> JSON on stdout; writes nothing.

Exit 0 only if (1) h100/codec_verify.py passes on NEW and on BASE (real H100 evidence, every in-bundle gate, rates from raw counts and
billed wall seconds) and (2) every loss.f32 / P.f32 / toks.i32 / pf_out.bf16 dump under BASE/cdc/<arm>/ is byte-equal in NEW
(training losses + adapters, greedy decode ids, prefill outputs). Report: NEW/BASE ratios of billed rates per arm (train-alone job-steps,
decode solo tokens, best co U vs dedicated and vs time-slice, co job-steps and decode tokens per paid GPU-hour) per decode batch b.
"""

import json
import subprocess
import sys
from pathlib import Path

NAMES = ("loss.f32", "P.f32", "toks.i32", "pf_out.bf16")


def verify(bundle):
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("codec_verify.py")), str(bundle)], capture_output=True, text=True)
    try:
        return p.returncode, json.loads(p.stdout)
    except ValueError:
        return p.returncode, None


def ratio(a, b):
    return a / b if a and b else None


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: ab_verify.py BASE NEW")
    base, new = Path(sys.argv[1]), Path(sys.argv[2])
    (rb, jb), (rn, jn) = verify(base), verify(new)
    fail = []
    if rb != 0:
        fail.append(f"base codec_verify rc={rb}: {(jb or {}).get('failed', [])[:6]}")
    if rn != 0:
        fail.append(f"new codec_verify rc={rn}: {(jn or {}).get('failed', [])[:6]}")
    files = sorted(str(p.relative_to(base)) for p in (base / "cdc").glob("*/*") if p.name in NAMES) if (base / "cdc").is_dir() else []
    if not files:
        fail.append("no dumps under BASE/cdc")
    diff = [f for f in files if not (new / f).is_file() or (new / f).read_bytes() != (base / f).read_bytes()]
    if diff:
        fail.append(f"{len(diff)} of {len(files)} dumps differ")
    per_b = []
    if jb and jn:
        cb = {c["b"]: c for c in jb.get("comparison", [])}
        for c in jn.get("comparison", []):
            o = cb.get(c["b"])
            if not o:
                continue
            bn = next((r for r in c["co_rows"] if r["arm"] == c["best_co"]), {})
            bo = next((r for r in o["co_rows"] if r["arm"] == o["best_co"]), {})
            per_b.append(dict(b=c["b"], best_co_new=c["best_co"], best_co_base=o["best_co"],
                              co_vs_2_dedicated_new=c["co_vs_2_dedicated"], co_vs_2_dedicated_base=o["co_vs_2_dedicated"],
                              co_vs_time_slice_new=c["co_vs_time_slice"], co_vs_time_slice_base=o["co_vs_time_slice"],
                              U_ratio=ratio(c["co_vs_2_dedicated"], o["co_vs_2_dedicated"]),
                              co_job_steps_ratio=ratio(bn.get("job_steps_per_paid_gpu_hour"), bo.get("job_steps_per_paid_gpu_hour")),
                              co_decode_tokens_ratio=ratio(bn.get("decode_tokens_per_paid_gpu_hour"), bo.get("decode_tokens_per_paid_gpu_hour")),
                              decode_solo_ratio=ratio(c["dedicated"]["decode_tokens_per_paid_gpu_hour"], o["dedicated"]["decode_tokens_per_paid_gpu_hour"])))
    res = dict(base=str(base), new=str(new), compared_dumps=len(files), differing_dumps=diff[:20],
               train_alone_job_steps_ratio=ratio((jn or {}).get("train_alone_job_steps_per_s_billed"), (jb or {}).get("train_alone_job_steps_per_s_billed")),
               per_b=per_b, passed=not fail, failed=fail)
    print(json.dumps(res, indent=1))
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
