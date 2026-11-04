"""Independent stdlib verifier for the Moeslim H100 PCIe tilecap sweep:
python3 h100/tilecap_pcie_verify.py --base R16:S1:DIR --bundle R:S:DIR [--bundle ...] -> JSON on stdout; exit 0 only if every gate passes.

Shares no code with codec_score / codec_verify / f_split and reads no summary JSON: raw artifacts only.
Per bundle: preflight.json passed, nvidia-smi name exactly 'NVIDIA H100 PCIe', driver >= 580, MATMUL_OK; every cdc_<run>.wall rc 0;
32 solo16_j refs; for each co_b<b>_r<r> arm and the ts_b<b> time-slice pair: decode ids bitwise vs dec_b<b> solo, per-job first-16 losses
bitwise vs solo16_j<j>, all losses finite, p99 TPOT <= 1.10x solo, decode outlasts training (co), one kernel launch (co), >= 50% 100 ms DCGM
coverage; f_decode, f_train, U over billed wall time; worst tile from tmax.u64 (report). No fatal CUDA/Python text in any cdc log.
Against BASE (same box, flag defaults = the 9873b79 config): S == 1 bundles must match every common run byte for byte (gate);
S > 1 bundles report it only (split-K changes the fp32 summation order). Hardware class H100 PCIe (moeslim): never pooled with SXM5.
"""

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

PRE = "cdc"
GPU = "NVIDIA H100 PCIe"
STEPS = re.compile(r"decode: b=(\d+), (\d+) token steps")
TPOT = re.compile(r"tpot p50 ([\d.]+) ms p99 ([\d.]+) ms")
FATAL = re.compile(r"CUDA error|an illegal memory access|out of memory|Traceback \(most recent call last\)|Segmentation fault|Aborted")
ARM = re.compile(r"co_b(\d+)_r(\d+)$")


def rd(p):
    return p.read_bytes() if p.is_file() else None


def txt(p):
    return p.read_text(errors="replace") if p.is_file() else ""


def u64(p):
    b = rd(p) or b""
    return [x for x in struct.unpack(f"{len(b) // 8}Q", b) if x]


def f32(p):
    b = rd(p) or b""
    return struct.unpack(f"{len(b) // 4}f", b)


def spec(s):
    r, k, d = s.split(":", 2)
    return int(r.lstrip("R")), int(k.lstrip("S")), Path(d)


def verify(out, jobs, batches, tpot_ratio):
    fail, d = [], out / PRE

    def need(k, ok, detail=None):
        if not ok:
            fail.append(k if detail is None else f"{k}: {detail}")
        return bool(ok)

    try:
        pf = json.loads(txt(out / "preflight.json"))
    except ValueError:
        pf = None
    ck = (pf or {}).get("checks", {})
    g = ck.get("gpu_name", {})
    f = [x.strip() for x in str(g.get("detail", "")).strip().split(",")]
    need("preflight_passed", bool(pf and pf.get("passed")))
    need("gpu_name_exact", bool(g.get("ok")) and f[0] == GPU, g.get("detail"))
    dm = f[1].split(".")[0] if len(f) >= 2 else ""
    need("driver_ge_580", dm.isdigit() and int(dm) >= 580, f[1:2])
    need("matmul_ok", "MATMUL_OK" in str(ck.get("container_cuda_matmul_nvcc", {}).get("detail", "")))

    walls = {}
    for p in sorted(out.glob(f"{PRE}_*.wall")):
        t = dict(ln.split()[:2] for ln in txt(p).splitlines() if len(ln.split()) >= 2)
        try:
            walls[p.name[len(PRE) + 1:-5]] = (float(t["start"]), float(t["end"]), int(t["rc"]))
        except (KeyError, ValueError):
            walls[p.name[len(PRE) + 1:-5]] = None
    need("walls", bool(walls))
    for k, w in walls.items():
        need(f"rc0_{k}", w is not None and w[2] == 0, w)
    for p in sorted(out.glob(f"{PRE}_*.log")):
        m = FATAL.search(txt(p))
        need(f"no_fatal_{p.name}", not m, m and m.group(0))

    def billed(*names):
        w = [walls.get(x) for x in names]
        if not need(f"wall_{'+'.join(names)}", all(w)):
            return None
        return max(t[1] for t in w) - min(t[0] for t in w)

    def dcgm(name, secs):
        rows = [ln for ln in txt(out / f"dcgm_{PRE}_{name}.txt").splitlines() if re.search(r"\d", ln) and not ln.startswith(("#", "timestamp"))]
        need(f"dcgm_100ms_{name}", bool(secs) and len(rows) >= 0.5 * secs / 0.1, len(rows))

    def tokens(name):
        m = STEPS.search(txt(out / f"{PRE}_{name}.log"))
        need(f"toks_{name}", m and rd(d / name / "toks.i32"))
        return int(m.group(1)) * int(m.group(2)) if m else None

    def steps(name):
        ls = f32(d / name / "loss.f32")
        need(f"losses_{name}", ls)
        need(f"finite_losses_{name}", ls and all(math.isfinite(x) for x in ls))
        return len(ls) or None

    def tpot(name):
        m = TPOT.search(txt(out / f"{PRE}_{name}.log"))
        return float(m.group(2)) if m else None

    refs = [f32(d / f"solo16_j{j}" / "loss.f32")[:16] for j in range(jobs)]
    need("solo16_refs", all(len(r) == 16 for r in refs) and not (d / f"solo16_j{jobs}").exists(), [len(r) for r in refs[:4]])

    def losses_bitwise(name):
        co = f32(d / name / "loss.f32")
        return len(co) >= 16 * jobs and all(len(refs[j]) == 16 and struct.pack("f", co[s * jobs + j]) == struct.pack("f", refs[j][s])
                                             for j in range(jobs) for s in range(16))

    def frac(rate, ref):
        return rate / ref if rate and ref else None

    t_s = billed("train")
    r_train = (steps("train") or 0) / t_s if t_s else None
    dcgm("train", t_s)
    for j in range(jobs):
        billed(f"solo16_j{j}")
    arms = {}
    for p in sorted(d.glob("co_b*_r*")) if d.is_dir() else []:
        m = ARM.match(p.name)
        if m:
            arms.setdefault(int(m.group(1)), []).append((int(m.group(2)), p.name))
    need("batches", sorted(arms) == sorted(batches), sorted(arms))
    rows = []
    for b in sorted(arms):
        sd = billed(f"dec_b{b}")
        r_dec = (tokens(f"dec_b{b}") or 0) / sd if sd else None
        dcgm(f"dec_b{b}", sd)
        solo_ids, solo_p99 = rd(d / f"dec_b{b}" / "toks.i32"), tpot(f"dec_b{b}")
        pairs = [(r, n, n, n, (n,)) for r, n in sorted(arms[b])] + [(None, f"ts_b{b}", f"ts_dec_b{b}", f"ts_train_b{b}", (f"ts_dec_b{b}", f"ts_train_b{b}"))]
        for r, n, dn, tn, wn in pairs:
            s = billed(*wn)
            dcgm(n, s)
            fd = frac((tokens(dn) or 0) / s if s else None, r_dec)
            ft = frac((steps(tn) or 0) / s if s else None, r_train)
            ids_ok = need(f"bitwise_tokens_{dn}", solo_ids and rd(d / dn / "toks.i32") == solo_ids)
            ls_ok = need(f"bitwise_losses_{tn}", losses_bitwise(tn))
            p99 = tpot(dn)
            need(f"tpot_{dn}", p99 is not None and solo_p99 and p99 <= tpot_ratio * solo_p99, (p99, solo_p99))
            row = dict(arm=n, b=b, reserve=r, billed_s=s, f_decode=fd, f_train=ft, U=fd + ft if fd is not None and ft is not None else None,
                       tpot_p99_ms=p99, solo_tpot_p99_ms=solo_p99, bitwise_vs_same_bundle_refs=ids_ok and ls_ok)
            if r is not None:
                sdn, tdn = u64(d / n / "step_done.u64"), u64(d / n / "tok_done.u64")
                need(f"overlap_{n}", sdn and tdn and tdn[-1] >= sdn[-1])
                need(f"one_launch_{n}", "kernel launches 1 for" in txt(out / f"{PRE}_{n}.log"))
                tb = rd(d / n / "tmax.u64") or b""
                tm = {k: v * 1e-3 for k, v in enumerate(struct.unpack(f"{len(tb) // 8}Q", tb)) if v}
                w = max(tm.items(), key=lambda kv: kv[1], default=(None, None))
                row.update(worst_tile_kind=w[0], worst_tile_us=w[1])
            rows.append(row)
    return dict(bundle=str(out), passed=not fail, failed=fail, arms=rows)


def same_bytes(a, b):
    da, db = a / PRE, b / PRE
    runs = sorted({p.parent.name for p in da.glob("*/*") if p.suffix in (".i32", ".f32") and p.name in ("toks.i32", "loss.f32")}
                  & {p.parent.name for p in db.glob("*/*") if p.name in ("toks.i32", "loss.f32")}) if da.is_dir() and db.is_dir() else []
    diff = [f"{r}/{n}" for r in runs for n in ("toks.i32", "loss.f32") if rd(da / r / n) != rd(db / r / n)]
    return dict(common_runs=len(runs), identical=bool(runs) and not diff, differing=diff[:64])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="R16:S1:DIR, the same-box flag-default bundle")
    ap.add_argument("--bundle", action="append", default=[], help="R:S:DIR")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--tpot-ratio", type=float, default=1.10)
    a = ap.parse_args()
    bs = [int(x) for x in a.batches.split(",")]
    br, bk, bd = spec(a.base)
    base = verify(bd, a.jobs, bs, a.tpot_ratio)
    fail = [] if base["passed"] and (br, bk) == (16, 1) else [f"base R{br}/S{bk}"]
    out = []
    for s in a.bundle:
        r, k, p = spec(s)
        v = verify(p, a.jobs, bs, a.tpot_ratio)
        ab = same_bytes(bd, p)
        ab["role"] = "gate" if k == 1 else "report-only (split-K fp32 order)"
        v.update(ce_rows_per_task=r, gnn_voc_ksplit=k, vs_base=ab)
        if not v["passed"] or (k == 1 and not ab["identical"]):
            fail.append(f"R{r}/S{k}")
        out.append(v)
    res = dict(hardware_class="H100 PCIe (moeslim); never pooled with SXM5", gpu_name=GPU, base=dict(base, ce_rows_per_task=br, gnn_voc_ksplit=bk),
               bundles=out, passed=not fail, failed=fail)
    print(json.dumps(res, indent=1))
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
