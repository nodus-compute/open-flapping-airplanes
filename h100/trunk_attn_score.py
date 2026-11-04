"""Score tensor-core attention (--attn tc) on the real trunk: python3 h100/trunk_attn_score.py OUT -> OUT/trunk_attn_summary.json.

Declared tolerance (not bitwise; the mma.sync path reduces in a different order): the cut hidden state of each path vs HF fp32 must pass
trunk_ckpt_ref's gate (rel <= max(1e-2, 1.5x HF bf16-vs-fp32)). Also reports bitwise equality with the CUDA-core path, trunk steps/s
of both paths, 100 ms DCGM and errors.
"""

import json
import sys
from pathlib import Path

from trunk_ckpt_score import rate


def main():
    out = Path(sys.argv[1])
    arms, errors = {}, []
    for at in ("cuda", "tc"):
        refp = out / "tca" / at / "trunk_ckpt_ref.json"
        ref = json.loads(refp.read_text()) if refp.exists() else {}
        lg = out / f"tca_rate_{at}.log"
        r = rate(lg.read_text(errors="replace")) if lg.exists() else None
        if not ref or r is None:
            errors.append(f"{at}: missing ref or rate")
        arms[at] = dict(ref=ref, steps_s=r, gate_pass=bool(ref.get("gate_pass")), dcgm=(out / f"dcgm_tca_{at}.txt").exists())
    xs = [out / "tca" / at / "Xt.bf16" for at in ("cuda", "tc")]
    bit = all(x.exists() for x in xs) and xs[0].read_bytes() == xs[1].read_bytes()
    gates = dict(tc_within_declared_tolerance=arms["tc"]["gate_pass"], cuda_within_declared_tolerance=arms["cuda"]["gate_pass"],
                 dcgm_100ms=all(a["dcgm"] for a in arms.values()), zero_errors=not errors)
    c, t = arms["cuda"]["steps_s"], arms["tc"]["steps_s"]
    s = dict(arms=arms, bitwise_tc_vs_cuda=bit, tc_speedup_vs_cuda=t / c if t and c else None, gates=gates, passed=all(gates.values()), errors=errors)
    (out / "trunk_attn_summary.json").write_text(json.dumps(s, indent=1))
    print(f"tc x{s['tc_speedup_vs_cuda'] or 0:.2f} vs cuda-core, bitwise {bit}, gates {gates} passed={s['passed']}")


if __name__ == "__main__":
    main()
