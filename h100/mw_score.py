"""Scores the mw.cu multi-workload arm from run_h100.sh dumps (OUT/mw/*) -> OUT/mw_summary.json. Stdlib only.
Rates are taken over the window where both workloads were running (co-run and MPS pair), from GPU globaltimer stamps.
Gates: co-run training loss/params/grads (16 steps) byte-equal to solo training; co-run decode logits (token 0), greedy tokens and
per-token logit sums byte-equal to solo decode (offline forward); decode TPOT p99 <= 1.10x solo full-GPU p99; PyTorch tolerance check.
  python3 h100/mw_score.py OUT
"""
import array
import json
import pathlib
import sys

P99_MAX, WARM = 1.10, 3


def arr(path, code):
    a = array.array(code)
    if path.exists():
        a.frombytes(path.read_bytes())
    return list(a)


def p99(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * 0.99))] if xs else None


def train_rate(sd, lo=None, hi=None):
    sd = sd[WARM:]
    lo = sd[0] if lo is None else max(lo, sd[0]) if sd else lo
    hi = sd[-1] if hi is None else min(hi, sd[-1]) if sd else hi
    n = sum(1 for t in sd if lo < t <= hi)
    return n / ((hi - lo) * 1e-9) if sd and hi > lo else None


def dec_stats(td, b, lo=None, hi=None):
    lat = [(td[i] - td[i - 1]) * 1e-6 for i in range(1, len(td)) if (lo is None or td[i - 1] >= lo) and (hi is None or td[i] <= hi)]
    span = sum(lat) * 1e-3
    return {"tok_s": len(lat) * b / span if span else None, "tpot_p99_ms": p99(lat), "tokens": len(lat)}


def same_bytes(a, b, names):
    return all(a.joinpath(n).exists() and a.joinpath(n).read_bytes() == b.joinpath(n).read_bytes() for n in names)


def same_prefix(a, b, name, width):
    x, y = a.joinpath(name).read_bytes() if a.joinpath(name).exists() else b"", b.joinpath(name).read_bytes() if b.joinpath(name).exists() else b""
    n = min(len(x), len(y))
    return n >= width and x[:n] == y[:n]


def overlap(sd, td):
    if len(sd) <= WARM or len(td) < 2:
        return None, None
    return max(sd[WARM], td[0]), min(sd[-1], td[-1])


def main(out):
    out = pathlib.Path(out)
    mw = out / "mw"
    ref = json.loads((out / "mw_ref.json").read_text()) if (out / "mw_ref.json").exists() else {}
    solo_train = train_rate(arr(mw / "train" / "step_done.u64", "Q"))
    res = {"solo_train_steps_s": solo_train, "pytorch_ref": ref, "batches": {}}
    ok_all = bool(ref.get("pass")) and solo_train is not None
    for bdir in sorted(mw.glob("dec_b*")):
        b = int(bdir.name[5:])
        solo = dec_stats(arr(bdir / "tok_done.u64", "Q"), b)
        part = dec_stats(arr(mw / f"dec_part_b{b}" / "tok_done.u64", "Q"), b)
        row = {"solo_dec_full": solo, "solo_dec_partition": part}
        for arm, tdir, ddir in (("co", f"co_b{b}", f"co_b{b}"), ("lend", f"lend_b{b}", f"lend_b{b}"), ("mps", f"mps_train_b{b}", f"mps_dec_b{b}")):
            sd, td = arr(mw / tdir / "step_done.u64", "Q"), arr(mw / ddir / "tok_done.u64", "Q")
            lo, hi = overlap(sd, td)
            tr = train_rate(sd, lo, hi) if lo is not None else None
            ds = dec_stats(td, b, lo, hi) if lo is not None else {"tok_s": None, "tpot_p99_ms": None, "tokens": 0}
            u = (tr / solo_train + ds["tok_s"] / solo["tok_s"]) if tr and ds["tok_s"] and solo_train and solo["tok_s"] else None
            row[arm] = {"train_steps_s": tr, "train_vs_solo": tr / solo_train if tr and solo_train else None, **ds,
                        "dec_vs_solo": ds["tok_s"] / solo["tok_s"] if ds["tok_s"] and solo["tok_s"] else None, "combined_U": u}
        co16 = mw / f"co16_b{b}"
        gates = {
            "train_bitwise_vs_solo_16": same_bytes(mw / "train16", co16, ["loss.f32", "P.f32", "G.f32"]),
            "lend_train_bitwise_vs_solo_16": same_bytes(mw / "train16", mw / f"lend16_b{b}", ["loss.f32", "P.f32", "G.f32"]),
            "lend_decode_bitwise_vs_offline": same_bytes(bdir, mw / f"lend_b{b}", ["logits0.f32"]) and same_prefix(bdir, mw / f"lend_b{b}", "toks.i32", 4),
            "decode_logits_bitwise_vs_offline": same_bytes(bdir, co16, ["logits0.f32"]) and same_bytes(bdir, mw / f"co_b{b}", ["logits0.f32"]),
            "decode_tokens_bitwise_vs_offline": all(same_prefix(bdir, mw / d, n, 4) for d in (f"co16_b{b}", f"co_b{b}") for n in ("toks.i32", "lsum.f32")),
            "pytorch_tolerance": bool(ref.get("pass")),
        }
        cp, sp = row["co"]["tpot_p99_ms"], solo["tpot_p99_ms"]
        row["p99_vs_solo"] = cp / sp if cp and sp else None
        gates["decode_p99_within_1.10x_solo"] = row["p99_vs_solo"] is not None and row["p99_vs_solo"] <= P99_MAX
        lp = row["lend"]["tpot_p99_ms"]
        row["lend_p99_vs_solo"] = lp / sp if lp and sp else None
        lu, mu2 = row["lend"]["combined_U"], row["mps"]["combined_U"]
        row["lend_vs_mps_U"] = lu / mu2 if lu and mu2 else None
        cu, mu = row["co"]["combined_U"], row["mps"]["combined_U"]
        row["co_vs_mps_U"] = cu / mu if cu and mu else None
        row["gates"] = gates
        ok_all = ok_all and all(gates.values())
        res["batches"][b] = row
        print(f"b={b}: train steps/s solo {solo_train} co {row['co']['train_steps_s']} mps {row['mps']['train_steps_s']} | "
              f"decode tok/s solo {solo['tok_s']} co {row['co']['tok_s']} mps {row['mps']['tok_s']} | p99 co/solo {row['p99_vs_solo']} | "
              f"U co {cu} mps {mu} | gates {gates}")
    res["all_gates_pass"] = ok_all and bool(res["batches"])
    (out / "mw_summary.json").write_text(json.dumps(res, indent=1))
    print("all_gates_pass", res["all_gates_pass"])


if __name__ == "__main__":
    main(sys.argv[1])
