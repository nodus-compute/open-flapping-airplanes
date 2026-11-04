"""Scores an h100/run_h100.sh output dir -> summary.json with a pre-registered boolean verdict.

Combined useful work U (per GPU-second, normalized so time-slicing the two jobs = 1.0):
  U = train tok/s / best solo full-GPU train tok/s  +  serve tok/s / solo full-GPU vLLM tok/s
Arms: megakernel split (one launch: train + decode), MPS pair (PyTorch CUDA-graph train + vLLM, MPS split, measured
concurrently), serial (1.0 by definition). CoSched U is read from --cosched-summary if given.
"""
import argparse
import json
import math
import pathlib
import re

GATES = {
    "vs_serial_min": 3.0,
    "vs_mps_min": 1.5,
    "tpot_slo_ms": 50.0,
    "slo_attainment_min": 0.99,
    "loss_delta_max": 0.02,
    "errors_max": 0,
    "dcgm_coverage_min": 0.9,
}


def parse_mk_log(text):
    out = {}
    m = re.search(r"steady state: ([\d.]+) ms/step \(median ([\d.]+)\) -> (\d+) tok/s;\s+4,000 steps = ([\d.]+) s", text)
    if m:
        out.update(ms_step=float(m[1]), ms_step_median=float(m[2]), train_tok_s=float(m[3]), train_4000_s=float(m[4]))
    m = re.search(r"decode: (\d+) tokens generated during ([\d.]+) s of training -> (\d+) tok/s", text)
    if m:
        out.update(decode_tokens=int(m[1]), decode_window_s=float(m[2]), serve_tok_s=float(m[3]))
    m = re.search(r"^loss: (.*)$", text, re.M)
    if m:
        pairs = dict(re.findall(r"(step0|s\d+) ([\d.]+)", m[1]))
        if pairs:
            last = max((k for k in pairs if k != "step0"), key=lambda k: int(k[1:]), default="step0")
            out.update(loss_initial=float(pairs.get("step0", "nan")), loss_final=float(pairs[last]), loss_final_step=last)
    m = re.search(r"serve: demand ([\d.]+) tok/s \((open-loop|closed-loop)\), (\d+) tokens in the training window ([\d.]+) s -> ([\d.]+) tok/s, "
                  r"latency p50 ([\d.]+) ms p99 ([\d.]+) ms, slo<=50ms ([\d.]+), lent train tasks (\d+)", text)
    if m:
        out.update(demand_tok_s=float(m[1]) if m[2] == "open-loop" else None, served_tok_s=float(m[5]), latency_p50_ms=float(m[6]),
                   latency_p99_ms=float(m[7]), slo_attainment=float(m[8]), lent_tasks=int(m[9]))
    m = re.search(r"SM partition \((.*?)\)", text)
    if m:
        out["partition"] = m[1]
    out["errors"] = 0 if ("train_tok_s" in out) else 1
    return out


def load_json(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size else None


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def attach_dcgm_metrics(out, arms):
    marks = {}
    marks_path = out / "phase_marks.jsonl"
    if marks_path.exists():
        for line in marks_path.read_text(errors="replace").splitlines():
            try:
                mark = json.loads(line)
            except json.JSONDecodeError:
                continue
            marks.setdefault(mark.get("phase"), {})[mark.get("event")] = mark
    samples = []
    for line in (out / "dcgm.jsonl").read_text(errors="replace").splitlines() if (out / "dcgm.jsonl").exists() else ():
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        if all(sample.get(key) is not None for key in ("sm_active", "pipe_tensor_active", "dram_active")):
            samples.append(sample)
    phase_for_arm = {
        "pt_mps_pair": "mps_pair",
        "serve_mps_pair": "mps_pair",
    }
    for name, arm in arms.items():
        phase = phase_for_arm.get(name, name)
        window = marks.get(phase, {})
        start = window.get("start", {}).get("wall_time")
        end = window.get("end", {}).get("wall_time")
        if start is None or end is None or end <= start:
            arm["dcgm"] = {"coverage": 0.0, "valid_rows": 0, "reason": "phase marks unavailable"}
            continue
        rows = [
            sample for sample in samples
            if start <= sample.get("wall_time", -1) <= end
        ]
        expected = max(1, math.ceil((end - start) * 1000 / 100))
        metrics = {
            "coverage": min(1.0, len(rows) / expected),
            "valid_rows": len(rows),
            "expected_rows": expected,
            "mode": json.loads((out / "dcgm_state.json").read_text()).get("mode")
            if (out / "dcgm_state.json").exists() else None,
        }
        for key in ("sm_active", "pipe_tensor_active", "dram_active"):
            metrics[f"{key}_mean"] = mean([row.get(key) for row in rows])
        arm["dcgm"] = metrics


def collect(out):
    out = pathlib.Path(out)
    arms = {}
    for log in sorted(out.glob("mk_*.log")):
        if log.stem != "mk_loss20":
            arms[log.stem] = parse_mk_log(log.read_text(errors="replace"))
    for name in ("pt_full", "pt_graph", "te_graph", "pt_mps_pair", "mg_solo"):
        r = load_json(out / f"{name}.json")
        if r:
            arms[name] = {"train_tok_s": r.get("tok_s"), "train_s": r.get("seconds"), "loss_final": r.get("loss_final"),
                          "errors": 0}
    for name in ("serve_full", "serve_mps", "serve_mps_pair"):
        r = load_json(out / f"{name}.json")
        if r:
            arms[name] = {"serve_tok_s": mean(r.get("tok_s", [])), "errors": 0}
    attach_dcgm_metrics(out, arms)
    return arms


def summarize(arms, gates=GATES, cosched=None, wall_budget=None):
    s = {"gates": gates, "arms": arms, "errors": [], "verdict": False}
    solo_train = {k: arms[k]["train_tok_s"] for k in ("pt_full", "pt_graph", "mg_solo")
                  if k in arms and arms[k].get("train_tok_s") and arms[k]["train_tok_s"] > 0}
    solo_serve = arms.get("serve_full", {}).get("serve_tok_s")
    if not solo_train:
        s["errors"].append("no solo training baseline")
    if not solo_serve:
        s["errors"].append("no solo vLLM baseline (serve_full)")
    dcgm_coverage_ok = bool(arms) and all(
        arm.get("dcgm", {}).get("coverage", 0.0) >= gates["dcgm_coverage_min"]
        for arm in arms.values()
    )
    s["dcgm_coverage_ok"] = dcgm_coverage_ok
    low_coverage = [
        f"{name}={arm.get('dcgm', {}).get('coverage', 0.0):.3f}"
        for name, arm in arms.items()
        if arm.get("dcgm", {}).get("coverage", 0.0) < gates["dcgm_coverage_min"]
    ]
    if low_coverage:
        s["errors"].append(
            "DCGM coverage below 90%: " + ", ".join(low_coverage)
        )
    budget = wall_budget
    if budget:
        s["errors"].append(budget.get("reason", "maximum wall budget exceeded"))
    if s["errors"]:
        s["verdict"] = False
        return s
    best_train = max(solo_train.values())
    s.update(best_solo_train=max(solo_train, key=solo_train.get), best_solo_train_tok_s=best_train, solo_vllm_tok_s=solo_serve)
    ref_loss = mean([arms[k].get("loss_final") for k in ("pt_full", "pt_graph", "te_graph") if k in arms])

    def score(train_tok_s, serve_tok_s):
        return train_tok_s / best_train + serve_tok_s / solo_serve

    pair = None
    if "pt_mps_pair" in arms and "serve_mps_pair" in arms:
        pair = (arms["pt_mps_pair"]["train_tok_s"], arms["serve_mps_pair"]["serve_tok_s"])
        s["mps_useful_work"] = score(*pair)
    else:
        s["errors"].append("MPS pair baseline missing")
    s["cosched_useful_work"] = (cosched or {}).get("useful_work")
    rows = {}
    for name, a in arms.items():
        if not name.startswith(("mk_split", "mk_paced", "mk_dyn", "mk_wg_split", "mk_wg_dyn")):
            continue
        demand = a.get("demand_tok_s")
        served = a.get("served_tok_s") if demand else a.get("serve_tok_s")
        if not (a.get("train_tok_s") and served):
            rows[name] = {"pass": False, "errors": 1, "why": "missing train or decode numbers"}
            continue
        credit = min(served, demand) if demand else served     # open-loop arms get credit for demand met, not idle capacity
        u = score(a["train_tok_s"], credit)
        mps_u = score(pair[0], min(pair[1], demand) if demand else pair[1]) if pair else None
        slo = a.get("slo_attainment")
        tpot = 1e3 / served
        slo_ok = (slo >= gates["slo_attainment_min"]) if slo is not None else (tpot <= gates["tpot_slo_ms"])
        dloss = abs(a["loss_final"] - ref_loss) if (ref_loss is not None and a.get("loss_final") is not None) else None
        row = {"useful_work": u, "vs_serial": u, "train_frac": a["train_tok_s"] / best_train, "serve_frac": credit / solo_serve,
               "demand_tok_s": demand, "served_tok_s": served, "slo_attainment": slo, "latency_p99_ms": a.get("latency_p99_ms"),
               "lent_tasks": a.get("lent_tasks"), "mean_tpot_ms": tpot, "loss_delta": dloss, "mps_useful_work_same_demand": mps_u,
               "vs_mps": u / mps_u if mps_u else None,
               "vs_cosched": u / s["cosched_useful_work"] if s["cosched_useful_work"] else None, "errors": a["errors"]}
        row["pass"] = bool(row["vs_serial"] >= gates["vs_serial_min"] and row["vs_mps"] is not None
                           and row["vs_mps"] >= gates["vs_mps_min"] and slo_ok
                           and dloss is not None and dloss <= gates["loss_delta_max"] and row["errors"] <= gates["errors_max"])
        rows[name] = row
    for name, row in rows.items():
        twin = "mk_paced_" + name[len("mk_dyn_"):]
        if name.startswith("mk_dyn_") and rows.get(twin, {}).get("useful_work"):
            row["vs_static_same_demand"] = row["useful_work"] / rows[twin]["useful_work"]
    s["fused"] = rows
    s["verdict"] = (
        any(r.get("pass") for r in rows.values())
        and dcgm_coverage_ok
        and budget is None
    )
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cosched-summary")
    a = ap.parse_args(argv)
    cos = load_json(a.cosched_summary) if a.cosched_summary else None
    s = summarize(
        collect(a.out),
        GATES,
        cos,
        load_json(pathlib.Path(a.out) / "wall_budget.json"),
    )
    pathlib.Path(a.out, "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s.get(k) for k in ("verdict", "fused", "mps_useful_work", "best_solo_train", "errors")}, indent=1))
    return s


if __name__ == "__main__":
    main()
