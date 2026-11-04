"""Read-only f_decode / f_train analyzer for a codec bundle, from raw timestamps only: python3 h100/f_split.py OUT [--partial] -> JSON on stdout.

Writes nothing, reads no summaries. Per run dir under OUT/cdc: tok_done.u64 (globaltimer ns per decode token step) and step_done.u64
(per train step), plus toks.i32 / loss.f32 counts. Rates use the steady window: decode = (tokens after the first 4 steps) / span,
train = (steps after the first step) / span; co arms are restricted to the interval where both run (overlap). f_decode = co decode rate /
dec_b<b> solo rate, f_train = co train rate / train-alone rate, U = f_decode + f_train (vs 2 dedicated GPUs). --partial: report what
exists (a run still in progress), exit 0; otherwise exit 0 only when every co arm has both f values.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path


def u64(p):
    b = p.read_bytes() if p.is_file() else b""
    return [x for x in struct.unpack(f"{len(b) // 8}Q", b) if x] if b else []


def rate(ts, per, lo=None, hi=None, skip=1):
    t = [x for x in ts if (lo is None or x >= lo) and (hi is None or x <= hi)][skip:]
    return per * (len(t) - 1) / ((t[-1] - t[0]) * 1e-9) if len(t) >= 2 and t[-1] > t[0] else None


def tile_report(p, step_s):
    """Report-only: per-kind max train-tile ns (tmax.u64, indexed by Ph kind) vs the solo decode token-step interval."""
    b = (p / "tmax.u64").read_bytes() if (p / "tmax.u64").is_file() else b""
    tm = {k: v * 1e-3 for k, v in enumerate(struct.unpack(f"{len(b) // 8}Q", b)) if v} if b else {}
    worst = max(tm.items(), key=lambda kv: kv[1], default=None)
    return dict(max_tile_us_by_kind=tm or None, worst_tile_kind=worst and worst[0], worst_tile_us=worst and worst[1],
                worst_tile_vs_solo_token_step=worst[1] * 1e-6 / step_s if worst and step_s else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--min-u", type=float, default=0.0, help="gate: some co arm with U >= this (and f_decode >= --min-f-decode)")
    ap.add_argument("--min-f-decode", type=float, default=0.0)
    a = ap.parse_args()
    d = Path(a.out) / "cdc"
    tr = u64(d / "train" / "step_done.u64")
    t_alone = rate(tr, 1)
    rows, missing = [], []
    for p in sorted(d.glob("co_b*_r*")) if d.is_dir() else []:
        m = re.match(r"co_b(\d+)_r(\d+)(_bt)?$", p.name)
        if not m:
            continue
        b = int(m.group(1))
        sd, td = u64(p / "step_done.u64"), u64(p / "tok_done.u64")
        ds = rate(u64(d / f"dec_b{b}" / "tok_done.u64"), b, skip=4)
        lo, hi = (max(sd[0], td[0]), min(sd[-1], td[-1])) if sd and td else (None, None)
        dc, tc = (rate(td, b, lo, hi, 4), rate(sd, 1, lo, hi)) if lo is not None and hi > lo else (None, None)
        fd, ft = (dc / ds if dc and ds else None), (tc / t_alone if tc and t_alone else None)
        if fd is None or ft is None:
            missing.append(p.name)
        rows.append(dict(arm=p.name, b=b, overlap_s=(hi - lo) * 1e-9 if lo is not None and hi > lo else None, decode_solo_tok_s=ds, decode_co_tok_s=dc,
                         train_alone_steps_s=t_alone, train_co_steps_s=tc, f_decode=fd, f_train=ft,
                         U=fd + ft if fd is not None and ft is not None else None,
                         **tile_report(p, b / ds if ds else None)))
    ok = [r for r in rows if r["U"] is not None and r["U"] >= a.min_u and r["f_decode"] >= a.min_f_decode]
    best = max((r for r in rows if r["U"] is not None), key=lambda r: r["U"], default=None)
    res = dict(bundle=str(a.out), arms=rows, missing=missing, complete=bool(rows) and not missing, best_arm=best and best["arm"],
               gate=dict(min_u=a.min_u, min_f_decode=a.min_f_decode, passing_arms=[r["arm"] for r in ok]))
    print(json.dumps(res, indent=1))
    sys.exit(0 if a.partial or (res["complete"] and ok) else 1)


if __name__ == "__main__":
    main()
