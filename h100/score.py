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

GATES = {"vs_serial_min": 3.0, "vs_mps_min": 1.5, "tpot_slo_ms": 50.0, "loss_delta_max": 0.02, "errors_max": 0}


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


def collect(out):
    out = pathlib.Path(out)
    arms = {}
    for name in ("mk_train", "mk_split", "mk_split_dyn"):
        log = out / f"{name}.log"
        if log.exists():
            arms[name] = parse_mk_log(log.read_text(errors="replace"))
    for name in ("pt_full", "pt_graph", "te_graph", "pt_mps_pair"):
        r = load_json(out / f"{name}.json")
        if r:
            arms[name] = {"train_tok_s": r.get("tok_s"), "train_s": r.get("seconds"), "loss_final": r.get("loss_final"),
                          "errors": 0}
    for name in ("serve_full", "serve_mps", "serve_mps_pair"):
        r = load_json(out / f"{name}.json")
        if r:
            arms[name] = {"serve_tok_s": mean(r.get("tok_s", [])), "errors": 0}
    return arms


def summarize(arms, gates=GATES, cosched=None):
    s = {"gates": gates, "arms": arms, "errors": [], "verdict": False}
    solo_train = {k: arms[k]["train_tok_s"] for k in ("mk_train", "pt_full", "pt_graph", "te_graph")
                  if k in arms and arms[k].get("train_tok_s")}
    solo_serve = arms.get("serve_full", {}).get("serve_tok_s")
    if not solo_train:
        s["errors"].append("no solo training baseline")
    if not solo_serve:
        s["errors"].append("no solo vLLM baseline (serve_full)")
    if s["errors"]:
        return s
    best_train = max(solo_train.values())
    s.update(best_solo_train=max(solo_train, key=solo_train.get), best_solo_train_tok_s=best_train, solo_vllm_tok_s=solo_serve)
    ref_loss = mean([arms[k].get("loss_final") for k in ("pt_full", "pt_graph", "te_graph") if k in arms])

    def score(train_tok_s, serve_tok_s):
        return train_tok_s / best_train + serve_tok_s / solo_serve

    if "pt_mps_pair" in arms and "serve_mps_pair" in arms:
        s["mps_useful_work"] = score(arms["pt_mps_pair"]["train_tok_s"], arms["serve_mps_pair"]["serve_tok_s"])
    else:
        s["errors"].append("MPS pair baseline missing")
    s["cosched_useful_work"] = (cosched or {}).get("useful_work")
    rows = {}
    for name in ("mk_split", "mk_split_dyn"):
        a = arms.get(name)
        if not a:
            continue
        if not (a.get("train_tok_s") and a.get("serve_tok_s")):
            rows[name] = {"pass": False, "errors": 1, "why": "missing train or decode numbers"}
            continue
        u = score(a["train_tok_s"], a["serve_tok_s"])
        tpot = 1e3 / a["serve_tok_s"]
        dloss = abs(a["loss_final"] - ref_loss) if (ref_loss is not None and a.get("loss_final") is not None) else None
        row = {"useful_work": u, "vs_serial": u, "train_frac": a["train_tok_s"] / best_train,
               "serve_frac": a["serve_tok_s"] / solo_serve, "mean_tpot_ms": tpot, "loss_delta": dloss,
               "vs_mps": u / s["mps_useful_work"] if s.get("mps_useful_work") else None,
               "vs_cosched": u / s["cosched_useful_work"] if s["cosched_useful_work"] else None, "errors": a["errors"]}
        row["pass"] = bool(row["vs_serial"] >= gates["vs_serial_min"] and row["vs_mps"] is not None
                           and row["vs_mps"] >= gates["vs_mps_min"] and tpot <= gates["tpot_slo_ms"]
                           and dloss is not None and dloss <= gates["loss_delta_max"] and row["errors"] <= gates["errors_max"])
        rows[name] = row
    s["fused"] = rows
    s["verdict"] = any(r.get("pass") for r in rows.values())
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cosched-summary")
    a = ap.parse_args(argv)
    cos = load_json(a.cosched_summary) if a.cosched_summary else None
    s = summarize(collect(a.out), GATES, cos)
    pathlib.Path(a.out, "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s.get(k) for k in ("verdict", "fused", "mps_useful_work", "best_solo_train", "errors")}, indent=1))
    return s


if __name__ == "__main__":
    main()
