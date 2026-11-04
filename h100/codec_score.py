"""Score real Qwen3-4B decode + shared-trunk real-tail training in one launch: python3 h100/codec_score.py OUT -> OUT/codec_summary.json.

Gates per (b, reserve) co arm: decode token ids byte-equal to decode solo; each job's first 16 losses byte-equal to that job trained
alone (solo16_j*); p99 TPOT <= 1.10x decode solo; decode still running when training ends (rates overlap); one launch; 100 ms DCGM;
zero errors. Each gate holds per attention-backward arm (cuda runs unsuffixed, --attn-bwd tc runs suffixed _bt, each against its own
solo/train-alone/solo16 references; decode solo is shared). Declared tc-vs-cuda tolerance: each co arm's first 16 per-job losses within
1e-3 relative of the matching cuda co arm (bitwise not expected). Report: U = decode/decode-solo + train/train-alone, i.e. combined useful work per paid GPU-hour vs two dedicated GPUs
(one decode, one shared-trunk training), and the same with training counted against the per-job dedicated arm (N x steps / solo).
Stock baseline (ts_dec_b*/ts_train_b*): the same decode and N-job training as 2 processes time-sliced on one GPU (no MPS), gated the same
way (token ids, solo16 losses bitwise, DCGM); host start/end give the overlap fraction (any non-overlap time favours the baseline).
CODEC_PREFILL=1 (pf solo, prefill in every co arm, ts_pf_b* third process): prefill outputs (final-norm hidden, pf_out.bf16) byte-equal to
pf solo, f_prefill = co prefill tok/s / solo, U adds f_prefill, and the dedicated baseline is 3 GPUs.
Per b: work per paid GPU-hour as job steps + decode tokens, and U for co (1 GPU), ts (1 GPU) and 2 dedicated GPUs (U = 2 over 2 GPU-h).
"""

import argparse
import json
import re
import struct
from pathlib import Path

from trunk_ckpt_score import log, rate, raw

PRE = "cdc"
PFR = re.compile(r"prefill: (\d+) of (\d+) chunks done in ([\d.]+) s -> ([\d.]+) tok/s, chunk p50 ([\d.]+) ms p99 ([\d.]+) ms")
DEC = re.compile(r"-> ([\d.]+) tok/s, tpot p50 ([\d.]+) ms p99 ([\d.]+) ms")


def dec(text):
    m = DEC.search(text or "")
    return dict(tok_s=float(m.group(1)), p50_ms=float(m.group(2)), p99_ms=float(m.group(3))) if m else None


def pfr(text):
    m = PFR.search(text or "")
    return dict(chunks=int(m.group(1)), of=int(m.group(2)), secs=float(m.group(3)), tok_s=float(m.group(4)), p50_ms=float(m.group(5)), p99_ms=float(m.group(6))) if m else None


def prefill(out, run, gates, key):
    """(f_prefill, row fields) for a run with the prefill tenant vs pf solo; (0.0, {}) when CODEC_PREFILL was off."""
    solo = pfr(log(out, "pf", PRE))
    if solo is None:
        return 0.0, {}
    r = pfr(log(out, run, PRE))
    ok = bool(raw(out, run, "pf_out.bf16", PRE)) and raw(out, run, "pf_out.bf16", PRE) == raw(out, "pf", "pf_out.bf16", PRE) and bool(r) and r["chunks"] == r["of"]
    gates[key] = gates.get(key, True) and ok
    f = r["tok_s"] / solo["tok_s"] if r and solo["tok_s"] else None
    return f, dict(prefill_solo=solo, prefill=r, f_prefill=f, prefill_bitwise=ok, prefill_tokens_per_paid_gpu_hour=3600 * r["tok_s"] if r else None)


def u64s(out, run, name):
    b = raw(out, run, name, PRE)
    return list(struct.unpack(f"{len(b) // 8}Q", b)) if b else []


def losses_ok(out, run, solos, sx):
    lb = raw(out, run, "loss.f32", PRE)
    n = len(solos)
    co = struct.unpack(f"{len(lb) // 4}f", lb) if lb else ()
    if not solos or len(co) < 16 * n:
        return False
    for j in solos:
        sb = raw(out, f"solo16_j{j}{sx}", "loss.f32", PRE)
        ref = struct.unpack(f"{len(sb) // 4}f", sb)[:16] if sb else ()
        if len(ref) != 16 or any(struct.pack("f", co[s * n + j]) != struct.pack("f", ref[s]) for s in range(16)):
            return False
    return True


def times(d, name):
    p = d / f"{name}.times"
    t = dict(ln.split()[:2] for ln in p.read_text().splitlines() if ln.startswith(("start ", "end "))) if p.exists() else {}
    return (float(t["start"]), float(t["end"])) if "start" in t and "end" in t else None


def ts_rows(out, d, solos, t_alone, errors, gates):
    rows, n = [], len(solos)
    for p in sorted(d.glob("ts_dec_b*")):
        m = re.match(r"ts_dec_b(\d+)$", p.name)
        if not m:
            continue
        b, tn = int(m.group(1)), f"ts_train_b{m.group(1)}"
        ds, dt, tt = dec(log(out, f"dec_b{b}", PRE)), dec(log(out, p.name, PRE)), rate(log(out, tn, PRE))
        tok_ok = bool(raw(out, p.name, "toks.i32", PRE)) and raw(out, p.name, "toks.i32", PRE) == raw(out, f"dec_b{b}", "toks.i32", PRE)
        tr_ok = losses_ok(out, tn, solos, "")
        fp, pfx = prefill(out, f"ts_pf_b{b}", gates, "ts_prefill_bitwise")
        tw = [times(d, p.name), times(d, tn)] + ([times(d, f"ts_pf_b{b}")] if pfx else [])
        ov = (min(t[1] for t in tw) - max(t[0] for t in tw)) / max(t[1] - t[0] for t in tw) if all(tw) else None
        if not (ds and dt and tt and ov is not None):
            errors.append(f"ts_b{b}: missing decode/train rate or host times")
        fd = dt["tok_s"] / ds["tok_s"] if ds and dt and ds["tok_s"] else None
        ft = tt / t_alone if tt and t_alone else None
        if fp is None:
            fd = None
        rows.append(dict(arm=f"ts_b{b}", b=b, decode_ts=dt, train_steps_s=tt, f_decode=fd, f_train=ft, overlap_frac=ov, **pfx,
                         gain_vs_2_dedicated_gpus=fd + ft + fp if fd is not None and ft is not None else None,
                         decode_tokens_per_paid_gpu_hour=3600 * dt["tok_s"] if dt else None,
                         train_useful_steps_per_paid_gpu_hour=3600 * n * tt if tt else None, decode_bitwise=tok_ok, train_bitwise=tr_ok))
        gates["ts_decode_bitwise"] &= tok_ok
        gates["ts_train_bitwise"] &= tr_ok
        gates["dcgm_100ms"] &= (out / f"dcgm_{PRE}_ts_b{b}.txt").exists()
    return rows


def score_arm(out, d, sx, tpot_ratio, errors, gates):
    rows = []
    solos = sorted(int(m.group(1)) for p in d.glob("solo16_j*") for m in [re.match(r"solo16_j(\d+)" + sx + "$", p.name)] if m) if d.exists() else []
    t_alone, t_solo = rate(log(out, "train" + sx, PRE)), rate(log(out, "solo_j0" + sx, PRE))
    if t_alone is None or t_solo is None:
        errors.append(f"missing train-alone or per-job dedicated rate{sx}")
    gates["train_bitwise"] &= bool(solos)
    for p in sorted(d.glob("co_b*_r*")) if d.exists() else []:
        m = re.match(r"co_b(\d+)_r(\d+)" + sx + "$", p.name)
        if not m:
            continue
        b, rs = int(m.group(1)), int(m.group(2))
        t = log(out, p.name, PRE)
        ds, dc, tc = dec(log(out, f"dec_b{b}", PRE)), dec(t), rate(t)
        tok_ok = bool(raw(out, p.name, "toks.i32", PRE)) and raw(out, p.name, "toks.i32", PRE) == raw(out, f"dec_b{b}", "toks.i32", PRE)
        n = len(solos)
        tr_ok = losses_ok(out, p.name, solos, sx)
        sd, td = [x for x in u64s(out, p.name, "step_done.u64") if x], u64s(out, p.name, "tok_done.u64")
        overlap = bool(sd and td) and td[-1] >= sd[-1]
        tp_ok = bool(ds and dc) and dc["p99_ms"] <= tpot_ratio * ds["p99_ms"]
        if not (ds and dc and tc):
            errors.append(f"{p.name}: missing decode or train rate")
        fd = dc["tok_s"] / ds["tok_s"] if ds and dc and ds["tok_s"] else None
        ft = tc / t_alone if tc and t_alone else None
        fj = n * tc / t_solo if tc and t_solo and n else None
        fp, pfx = prefill(out, p.name, gates, "prefill_bitwise")
        if fd is not None and fp is None:
            fd = None
        rows.append(dict(arm=p.name, b=b, reserve=rs, decode_solo=ds, decode_co=dc, train_steps_s=tc, f_decode=fd, f_train=ft,
                         gain_vs_2_dedicated_gpus=fd + ft + fp if fd is not None and ft is not None else None,
                         gain_vs_dedicated_decode_plus_perjob_train=fd + fj + fp if fd is not None and fj is not None else None,
                         decode_tokens_per_paid_gpu_hour=3600 * dc["tok_s"] if dc else None,
                         train_useful_steps_per_paid_gpu_hour=3600 * n * tc if tc else None,
                         decode_bitwise=tok_ok, train_bitwise=tr_ok, tpot_ok=tp_ok, rates_overlap=overlap, **pfx))
        gates["decode_bitwise"] &= tok_ok
        gates["train_bitwise"] &= tr_ok
        gates["tpot_p99"] &= tp_ok
        gates["rates_overlap"] &= overlap
        gates["one_launch"] &= "kernel launches 1 for" in t
        gates["dcgm_100ms"] &= (out / f"dcgm_{PRE}_{p.name}.txt").exists()
    ts = ts_rows(out, d, solos, t_alone, errors, gates) if not sx else []
    return dict(train_alone_steps_s=t_alone, perjob_dedicated_steps_s=t_solo, jobs=len(solos), rows=rows, ts_rows=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tpot-ratio", type=float, default=1.10)
    ap.add_argument("--loss-tol", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out)
    d = out / PRE
    errors, rows, groups = [], [], {}
    gates = dict(decode_bitwise=True, train_bitwise=True, tpot_p99=True, rates_overlap=True, one_launch=True, dcgm_100ms=True,
                 ts_decode_bitwise=True, ts_train_bitwise=True, zero_errors=True)
    present = {m.group(1) or "" for p in (d.glob("co_b*_r*") if d.exists() else []) for m in [re.match(r"co_b\d+_r\d+(_bt)?$", p.name)] if m}
    for sx in sorted(present):
        g = score_arm(out, d, sx, a.tpot_ratio, errors, gates)
        groups["tc" if sx else "cuda"] = g
        rows += g["rows"]
    if "cuda" in groups and "tc" in groups:
        cu = {r["arm"]: r for r in groups["cuda"]["rows"]}
        tol = []
        for r in groups["tc"]["rows"]:
            c = cu.get(r["arm"][: -len("_bt")])
            lt, lc = raw(out, r["arm"], "loss.f32", PRE), raw(out, c["arm"], "loss.f32", PRE) if c else None
            n = groups["tc"]["jobs"]
            if lt and lc and n:
                ft, fc = struct.unpack(f"{len(lt) // 4}f", lt)[:16 * n], struct.unpack(f"{len(lc) // 4}f", lc)[:16 * n]
                tol.append(max((abs(x - y) / max(abs(y), 1e-12) for x, y in zip(ft, fc)), default=None) if len(ft) == len(fc) == 16 * n else None)
            else:
                tol.append(None)
            r["loss_rel_vs_cuda"] = tol[-1]
            r["train_rate_vs_cuda"] = r["train_steps_s"] / c["train_steps_s"] if c and r["train_steps_s"] and c["train_steps_s"] else None
            r["gain_vs_cuda_arm"] = r["gain_vs_2_dedicated_gpus"] / c["gain_vs_2_dedicated_gpus"] if c and r["gain_vs_2_dedicated_gpus"] and c["gain_vs_2_dedicated_gpus"] else None
        gates["tc_bwd_loss_tol"] = bool(tol) and all(x is not None and x <= a.loss_tol for x in tol)
    ts = groups.get("cuda", {}).get("ts_rows", [])
    gates["ts_present"] = bool(ts)
    cmp = []
    for t in ts:
        cos = [r for r in groups["cuda"]["rows"] if r["b"] == t["b"] and r["gain_vs_2_dedicated_gpus"] is not None]
        best = max(cos, key=lambda r: r["gain_vs_2_dedicated_gpus"], default=None)
        for r in cos:
            r["gain_vs_time_slice"] = r["gain_vs_2_dedicated_gpus"] / t["gain_vs_2_dedicated_gpus"] if t["gain_vs_2_dedicated_gpus"] else None
        cmp.append(dict(b=t["b"], best_co_arm=best and best["arm"],
                        work_per_paid_gpu_hour={
                            "co_1gpu": best and dict(U=best["gain_vs_2_dedicated_gpus"], job_steps=best["train_useful_steps_per_paid_gpu_hour"],
                                                     decode_tokens=best["decode_tokens_per_paid_gpu_hour"]),
                            "time_slice_1gpu": dict(U=t["gain_vs_2_dedicated_gpus"], job_steps=t["train_useful_steps_per_paid_gpu_hour"],
                                                    decode_tokens=t["decode_tokens_per_paid_gpu_hour"]),
                            "prefill_tokens": dict(co_1gpu=best and best.get("prefill_tokens_per_paid_gpu_hour"),
                                                   time_slice_1gpu=t.get("prefill_tokens_per_paid_gpu_hour"),
                                                   dedicated=t["prefill_solo"]["tok_s"] * 3600 / 3 if t.get("prefill_solo") else None),
                            "dedicated_gpus": 3 if t.get("prefill_solo") else 2,
                            "dedicated_2gpu": dict(U=1.0, job_steps=1800 * groups["cuda"]["jobs"] * groups["cuda"]["train_alone_steps_s"]
                                                   if groups["cuda"]["train_alone_steps_s"] else None,
                                                   decode_tokens=1800 * t["decode_ts"]["tok_s"] / t["f_decode"] if t["decode_ts"] and t["f_decode"] else None)},
                        co_vs_time_slice=best and best.get("gain_vs_time_slice"), co_vs_2_dedicated=best and best["gain_vs_2_dedicated_gpus"]))
    gates["zero_errors"] = not errors and bool(rows)
    s = dict(arms={k: {x: v for x, v in g.items() if x not in ("rows", "ts_rows")} for k, g in groups.items()}, rows=rows, ts_rows=ts,
             comparison=cmp, gates=gates, passed=all(gates.values()), errors=errors)
    (out / "codec_summary.json").write_text(json.dumps(s, indent=1))
    for r in rows:
        print(f"{r['arm']}: f_decode {r['f_decode']} f_train {r['f_train']} gain vs 2 GPUs {r['gain_vs_2_dedicated_gpus']}")
    for c in cmp:
        print(f"b{c['b']}: co vs time-slice {c['co_vs_time_slice']} co vs 2 dedicated GPUs {c['co_vs_2_dedicated']}")
    print(f"gates {gates} passed={s['passed']}")


if __name__ == "__main__":
    main()
