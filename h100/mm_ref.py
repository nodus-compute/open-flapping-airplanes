"""Tolerance check of an mm.cu solo run against HF transformers (fp32, same pinned snapshot): greedy tokens and last-prompt logits.

python h100/mm_ref.py --model-dir /root/data/mm/qwen3_4b --dump /out/mm/solo1_b1 --index 1 --out /out/mm_ref_m1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MAXB, NTOK, MIN_PREFIX, LOGITS_REL = 32, 16, 8, 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md, dd = Path(a.model_dir), Path(a.dump)
    prep = json.loads((md / "prep.json").read_text())
    res = json.loads((dd / "result.json").read_text())
    b, P = res["b"], prep["prompt_len"]
    prompt = np.fromfile(md / "prompt.i32", dtype=np.int32).reshape(MAXB, P)
    toks = np.fromfile(dd / f"toks{a.index}.i32", dtype=np.int32).reshape(-1, MAXB)
    logits = torch.from_numpy(np.fromfile(dd / f"logitsP{a.index}.f32", dtype=np.float32).reshape(b, -1))
    model = AutoModelForCausalLM.from_pretrained(prep["snapshot"], torch_dtype=torch.float32).cuda().eval()
    rows = []
    for i in range(min(a.rows, b)):
        ids = torch.tensor(prompt[i:i + 1].astype(np.int64)).cuda()
        with torch.no_grad():
            ref_logits = model(ids).logits[0, -1].float().cpu()
            gen = model.generate(ids, max_new_tokens=NTOK, do_sample=False, temperature=None, top_p=None, top_k=None)[0, P:].tolist()
        ours = toks[P - 1:P - 1 + NTOK, i].tolist()
        pre = next((k for k in range(min(len(gen), len(ours))) if gen[k] != ours[k]), min(len(gen), len(ours)))
        rel = ((logits[i] - ref_logits).abs().max() / ref_logits.abs().max()).item()
        rows.append({"row": i, "match_prefix": pre, "ours": ours, "hf": gen, "logits_rel_err": rel})
    ok = all(r["match_prefix"] >= MIN_PREFIX and r["logits_rel_err"] <= LOGITS_REL for r in rows)
    r = {"repo": prep["repo"], "revision": prep["revision"], "index": a.index, "b": b, "rows": rows,
         "tol": {"min_greedy_prefix": MIN_PREFIX, "logits_rel": LOGITS_REL}, "pass": ok}
    print(json.dumps(r, indent=1))
    Path(a.out).write_text(json.dumps(r, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
