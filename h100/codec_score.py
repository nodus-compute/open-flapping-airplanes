"""Score real Qwen3-4B decode + shared-trunk real-tail training in one launch: python3 h100/codec_score.py OUT -> OUT/codec_summary.json.

Gates per (b, reserve) co arm: decode token ids byte-equal to decode solo; each job's first 16 losses byte-equal to that job trained
alone (solo16_j*); p99 TPOT <= 1.10x decode solo; decode still running when training ends (rates overlap); one launch; 100 ms DCGM;
zero errors. Report: U = decode/decode-solo + train/train-alone, i.e. combined useful work per paid GPU-hour vs two dedicated GPUs
(one decode, one shared-trunk training), and the same with training counted against the per-job dedicated arm (N x steps / solo).
"""

import argparse
import json
import re
import struct
from pathlib import Path

from trunk_ckpt_score import log, rate, raw

PRE = "cdc"
DEC = re.compile(r"-> ([\d.]+) tok/s, tpot p50 ([\d.]+) ms p99 ([\d.]+) ms")


def dec(text):
    m = DEC.search(text or "")
    return dict(tok_s=float(m.group(1)), p50_ms=float(m.group(2)), p99_ms=float(m.group(3))) if m else None


def u64s(out, run, name):
    b = raw(out, run, name, PRE)
    return list(struct.unpack(f"{len(b) // 8}Q", b)) if b else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tpot-ratio", type=float, default=1.10)
    a = ap.parse_args()
    out = Path(a.out)
    d = out / PRE
    errors, rows = [], []
    solos = sorted(int(m.group(1)) for p in d.glob("solo16_j*") for m in [re.match(r"solo16_j(\d+)$", p.name)] if m) if d.exists() else []
    t_alone, t_solo = rate(log(out, "train", PRE)), rate(log(out, "solo_j0", PRE))
    if t_alone is None or t_solo is None:
        errors.append("missing train-alone or per-job dedicated rate")
    gates = dict(decode_bitwise=True, train_bitwise=bool(solos), tpot_p99=True, rates_overlap=True, one_launch=True, dcgm_100ms=True, zero_errors=True)
    for p in sorted(d.glob("co_b*_r*")) if d.exists() else []:
        m = re.match(r"co_b(\d+)_r(\d+)$", p.name)
        if not m:
            continue
        b, rs = int(m.group(1)), int(m.group(2))
        t = log(out, p.name, PRE)
        ds, dc, tc = dec(log(out, f"dec_b{b}", PRE)), dec(t), rate(t)
        tok_ok = bool(raw(out, p.name, "toks.i32", PRE)) and raw(out, p.name, "toks.i32", PRE) == raw(out, f"dec_b{b}", "toks.i32", PRE)
        lb = raw(out, p.name, "loss.f32", PRE)
        n = len(solos)
        co = struct.unpack(f"{len(lb) // 4}f", lb) if lb else ()
        tr_ok = bool(solos) and len(co) >= 16 * n
        for j in solos:
            sb = raw(out, f"solo16_j{j}", "loss.f32", PRE)
            ref = struct.unpack(f"{len(sb) // 4}f", sb)[:16] if sb else ()
            tr_ok &= len(ref) == 16 and all(struct.pack("f", co[s * n + j]) == struct.pack("f", ref[s]) for s in range(16)) if len(co) >= 16 * n else False
        sd, td = [x for x in u64s(out, p.name, "step_done.u64") if x], u64s(out, p.name, "tok_done.u64")
        overlap = bool(sd and td) and td[-1] >= sd[-1]
        tp_ok = bool(ds and dc) and dc["p99_ms"] <= a.tpot_ratio * ds["p99_ms"]
        if not (ds and dc and tc):
            errors.append(f"{p.name}: missing decode or train rate")
        fd = dc["tok_s"] / ds["tok_s"] if ds and dc and ds["tok_s"] else None
        ft = tc / t_alone if tc and t_alone else None
        fj = n * tc / t_solo if tc and t_solo and n else None
        rows.append(dict(arm=p.name, b=b, reserve=rs, decode_solo=ds, decode_co=dc, train_steps_s=tc, f_decode=fd, f_train=ft,
                         gain_vs_2_dedicated_gpus=fd + ft if fd is not None and ft is not None else None,
                         gain_vs_dedicated_decode_plus_perjob_train=fd + fj if fd is not None and fj is not None else None,
                         decode_tokens_per_paid_gpu_hour=3600 * dc["tok_s"] if dc else None,
                         train_useful_steps_per_paid_gpu_hour=3600 * n * tc if tc else None,
                         decode_bitwise=tok_ok, train_bitwise=tr_ok, tpot_ok=tp_ok, rates_overlap=overlap))
        gates["decode_bitwise"] &= tok_ok
        gates["train_bitwise"] &= tr_ok
        gates["tpot_p99"] &= tp_ok
        gates["rates_overlap"] &= overlap
        gates["one_launch"] &= "kernel launches 1 for" in t
        gates["dcgm_100ms"] &= (out / f"dcgm_{PRE}_{p.name}.txt").exists()
    gates["zero_errors"] = not errors and bool(rows)
    s = dict(train_alone_steps_s=t_alone, perjob_dedicated_steps_s=t_solo, jobs=len(solos), rows=rows, gates=gates,
             passed=all(gates.values()), errors=errors)
    (out / "codec_summary.json").write_text(json.dumps(s, indent=1))
    for r in rows:
        print(f"{r['arm']}: f_decode {r['f_decode']} f_train {r['f_train']} gain vs 2 GPUs {r['gain_vs_2_dedicated_gpus']}")
    print(f"gates {gates} passed={s['passed']}")


if __name__ == "__main__":
    main()
