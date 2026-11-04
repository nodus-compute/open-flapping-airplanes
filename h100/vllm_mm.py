"""Stock vLLM decode client for the mm arm (stdlib only): per model, b concurrent streaming greedy completions of fixed prompt ids
(ignore_eos), per-token arrival stamps -> tok/s, TPOT p50/p99; with two servers the models run concurrently and the summary also
keeps every token stamp's span so the scorer can rate both models inside their overlap window.

python3 h100/vllm_mm.py --server m0:8101:1024 --server m1:8100:256 --prompt /root/data/mm/qwen3_4b/prompt.i32 --b 1 8 32 --out x.json
"""
import argparse
import http.client
import json
import sys
import threading
import time
from pathlib import Path

MAXB = 32
BLOCKED_PORTS = {8000}


def stream(port, name, ids, n, stamps, errs):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        body = json.dumps({"model": name, "prompt": ids, "max_tokens": n, "temperature": 0.0, "ignore_eos": True, "stream": True})
        c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            errs.append(f"{name}: HTTP {r.status}")
            return
        got = 0
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            stamps.append(time.perf_counter())
            got += 1
        if got < n:
            errs.append(f"{name}: {got}/{n} chunks")
    except Exception as e:  # noqa: BLE001
        errs.append(f"{name}: {e!r}")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="append", required=True, help="name:port:tokens")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--b", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import array
    pr = array.array("i")
    pr.frombytes(Path(a.prompt).read_bytes())
    prompts = [pr[i * a.prompt_len:(i + 1) * a.prompt_len].tolist() for i in range(MAXB)]
    servers = []
    for s in a.server:
        name, port, n = s.split(":")
        if int(port) in BLOCKED_PORTS:
            sys.exit(f"refusing port {port}")
        servers.append((name, int(port), int(n)))
    rows = []
    for b in a.b:
        for rep in range(a.reps + 1):                          # rep 0 = warm-up (graphs, prefix cache)
            per, threads = {}, []
            for name, port, n in servers:
                per[name] = {"stamps": [[] for _ in range(b)], "errs": [], "tokens": n}
                for i in range(b):
                    threads.append(threading.Thread(target=stream, args=(port, name, prompts[i], n, per[name]["stamps"][i], per[name]["errs"])))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if rep == 0:
                continue
            for name, d in per.items():
                st = d["stamps"]
                firsts = [s[0] for s in st if s]
                lasts = [s[-1] for s in st if s]
                gaps = [s[k] - s[k - 1] for s in st for k in range(1, len(s))]
                start, end = (max(firsts), min(lasts)) if firsts else (0, 0)
                ntok = sum(1 for s in st for x in s if start <= x <= end)
                rows.append({"system": "vllm_stock", "model": name, "b": b, "rep": rep, "concurrent_models": len(servers),
                             "tokens_per_row": d["tokens"], "tok_s": ntok / (end - start) if end > start else None,
                             "tpot_p50_ms": pct(gaps, 0.5) and pct(gaps, 0.5) * 1e3, "tpot_p99_ms": pct(gaps, 0.99) and pct(gaps, 0.99) * 1e3,
                             "start_s": min(firsts) if firsts else None, "end_s": max(lasts) if lasts else None,
                             "all_rows_start_s": start, "all_rows_end_s": end, "errors": d["errs"]})
                print(json.dumps({k: v for k, v in rows[-1].items() if k != "errors"} | {"errors": len(d["errs"])}), flush=True)
    Path(a.out).write_text(json.dumps({"rows": rows}, indent=1))
    return 0 if all(not r["errors"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
