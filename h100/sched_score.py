"""Score the sched arm: python3 h100/sched_score.py OUT  ->  OUT/sched_summary.json (+ table on stdout)."""
import json
import sys
from pathlib import Path

import numpy as np

P99_MAX = 1.10


def rd(d, name, dt):
    p = d / name
    return np.fromfile(p, dtype=dt) if p.is_file() else None


def res(d):
    p = d / "result.json"
    return json.loads(p.read_text()) if p.is_file() else None


def same_bytes(a, b, names):
    return all((a / n).is_file() and (b / n).is_file() and (a / n).read_bytes() == (b / n).read_bytes() for n in names)


def loss_prefix_equal(ref, d):
    a, b = rd(ref, "loss.f32", np.float32), rd(d, "loss.f32", np.float32)
    if a is None or b is None or len(b) == 0:
        return False, 0
    n = min(len(a), len(b))
    return a[:n].tobytes() == b[:n].tobytes(), n


def dec_window(d):
    t0, arr, td = rd(d, "t0.u64", np.uint64), rd(d, "arr.u64", np.uint64), rd(d, "tok_done.u64", np.uint64)
    if t0 is None or arr is None or td is None or len(td) == 0:
        return None
    return int(t0[0] + arr[0]), int(td[-1])


def train_in(d, win):
    sd = rd(d, "step_done.u64", np.uint64)
    if sd is None or win is None or win[1] <= win[0]:
        return None
    n = int(((sd >= win[0]) & (sd <= win[1])).sum())
    return n / ((win[1] - win[0]) * 1e-9)


def dcgm(out, name):
    p = out / f"dcgm_{name}.txt"
    if not p.is_file():
        return None
    cols = {}
    for line in p.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if not parts or parts[0] in ("#Entity", "ID", "timestamp") or parts[0].startswith("#"):
            continue
        vals = []
        for x in parts:
            try:
                vals.append(float(x.rstrip("%")))
            except ValueError:
                pass
        for i, v in enumerate(vals):
            cols.setdefault(i, []).append(v)
    return {"file": p.name, "samples": len(cols.get(0, [])), "column_means": [round(sum(v) / len(v), 4) for _, v in sorted(cols.items())]}


def main(out):
    out = Path(out)
    s = out / "sched"
    ref = s / "train_ref"
    solo_train = res(s / "train_solo") or {}
    rates = sorted({float(p.name.split("_r")[1]) for p in s.glob("solo_r*")})
    rows, gates = [], {}
    for r in rates:
        tag = f"r{r:g}"
        solo = res(s / f"solo_{tag}")
        row = {"rate_req_s": r, "train_alone_steps_s": solo_train.get("train_steps_s"), "decode_solo_req_p99_ms": solo and solo["req_p99_ms"]}
        arms = {"fifo": s / f"fifo_{tag}", "prio": s / f"prio_{tag}", "mps": s / f"mps_dec_{tag}"}
        for arm, d in arms.items():
            rr = res(d)
            tdir = s / f"mps_train_{tag}" if arm == "mps" else d
            win = dec_window(d)
            sps = (rr or {}).get("train_steps_s") if arm != "mps" else train_in(tdir, win)
            bit_dec = solo is not None and same_bytes(s / f"solo_{tag}", d, ["toks.i32", "lsum.f32", "logits0.f32"])
            bit_tr, n = loss_prefix_equal(ref, tdir)
            p99 = rr and rr["req_p99_ms"]
            row.update({
                f"{arm}_train_steps_s": sps, f"{arm}_train_steps_per_gpu_hour": sps and sps * 3600,
                f"{arm}_train_vs_alone": sps / solo_train["train_steps_s"] if sps and solo_train.get("train_steps_s") else None,
                f"{arm}_req_p99_ms": p99, f"{arm}_p99_vs_solo": p99 / solo["req_p99_ms"] if p99 and solo and solo["req_p99_ms"] else None,
                f"{arm}_tpot_p99_ms": rr and rr["tpot_p99_ms"], f"{arm}_decode_bitwise_vs_solo": bit_dec,
                f"{arm}_train_loss_bitwise_vs_alone": bit_tr, f"{arm}_loss_floats_compared": n,
                f"{arm}_all_requests_done": bool(rr) and rr["requests_done"] == rr["nreq"],
                f"{arm}_dcgm": dcgm(out, f"{arm}_{tag}"),
            })
            gates[f"{tag}_{arm}_decode_bitwise_vs_solo"] = bit_dec
            gates[f"{tag}_{arm}_train_loss_bitwise_vs_alone"] = bit_tr
            gates[f"{tag}_{arm}_zero_errors"] = row[f"{arm}_all_requests_done"]
        row["solo_dcgm"] = dcgm(out, f"solo_{tag}")
        row["prio_vs_fifo_train"] = row["prio_train_steps_s"] / row["fifo_train_steps_s"] if row["prio_train_steps_s"] and row["fifo_train_steps_s"] else None
        row["prio_vs_mps_train"] = row["prio_train_steps_s"] / row["mps_train_steps_s"] if row["prio_train_steps_s"] and row["mps_train_steps_s"] else None
        gates[f"{tag}_prio_p99_within_{P99_MAX}x_solo"] = row["prio_p99_vs_solo"] is not None and row["prio_p99_vs_solo"] <= P99_MAX
        rows.append(row)
    summary = {"train_solo_dcgm": dcgm(out, "train_solo"), "rows": rows, "gates": gates, "all_gates_pass": bool(gates) and all(gates.values())}
    summary["gemm_backend"] = "mma_sync"
    (out / "sched_summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["rate_req_s", "train_alone_steps_s", "fifo_train_steps_s", "prio_train_steps_s", "mps_train_steps_s", "decode_solo_req_p99_ms",
            "fifo_p99_vs_solo", "prio_p99_vs_solo", "mps_p99_vs_solo", "prio_vs_mps_train"]
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(f"{row[k]:.4g}" if isinstance(row[k], float) else str(row[k]) for k in keys))
    print("gates:", json.dumps(gates))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
