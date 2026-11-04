# h100/: open-flapping-airplanes on one H100 (Nodus fork)

Fork of https://github.com/Bobliuuu/open-flapping-airplanes, upstream SHA `9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1`. Upstream ships A100 (`mma.sync`),
B200 and B300 (`tcgen05`) builds and has no H100 build. This directory runs the SM-count-parameterised megakernel (`llm/b200/megakernel/ofa_r2.cu`,
`mma.sync` GEMM backend forced with `-DOFA_GEMM_BACKEND=0` (the r2 header defaults to 2, which falls back to mma.sync off sm_100), built for `sm_90a`, Doom compiled out with `NODOOM=1`) on a 132-SM H100. It compares that build with solo and MPS-split
baselines and scores combined useful work per GPU-second. Nothing here has run on an H100 yet, so there are no measured numbers.

## Command (1x H100; needs docker + nvidia-container-toolkit; the vLLM 0.30 image needs driver >= 580)
```bash
git clone -b <branch> https://github.com/nodus-compute/open-flapping-airplanes && cd open-flapping-airplanes
GPU=0 ALLOW_GPU0=1 ./h100/run_h100.sh                      # everything (GPU 0 needs ALLOW_GPU0=1)
GPU=0 ALLOW_GPU0=1 PHASES="prep build mk_loss20 mk_train mk_split score" ./h100/run_h100.sh   # megakernel repro only
```
Images (pinned): `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd...` builds and runs the megakernel and the PyTorch baselines.
`vllm/vllm-openai:v0.30.0-x86_64@sha256:5f5e5352...` runs the vLLM baseline. Data (about 1.5 GB) is cached in `$DATA` (default `~/ofa_data`).

## Arms (model S: 162.4M GPT, 4,000 steps x 4,096 tokens; split 111 train / 21 serve SMs, FA's 24/148 proportion)
| phase | what |
|---|---|
| `mk_train` | megakernel, training only, all 132 SMs (solo megakernel training) |
| `mk_split` | megakernel, ONE launch: training on 111 SMs plus decode on 21 SMs from the live weights |
| `mk_paced` | `mk_split` with an open-loop serving demand of R tok/s (`--serve-rate R`, R in `SERVE_RATES`, default 100 and 400), static split |
| `mk_dyn` | IMPROVED arm: `mk_paced` plus `--lend`. A serving block whose next token is not yet due claims one training task |
| `pt_full`, `pt_graph` | PyTorch eager / one CUDA graph per step, full GPU, solo. `pt_full` saves the checkpoint vLLM serves |
| `serve_full`, `serve_mps` | vLLM 0.30 batch-1 decode, 256 tokens, full GPU / MPS 16% |
| `mps_pair` | FA's baseline style, but MEASURED concurrently: PyTorch CUDA-graph training on MPS 84% plus vLLM on MPS 16%, one host MPS daemon |
| `build_wg`, `gemm_wg`, `mk_wg` | sm90 wgmma backend build, its GEMM correctness/speed check vs cuBLAS, and the megakernel arms on it (see below) |
| `score` | `h100/score.py OUT` writes `summary.json` |

Megatron + Transformer Engine needs the TE source build in `llm/b200/baseline/setup_b200.sh` (use `NVTE_CUDA_ARCHS=90`). It is not in the default
phases. If you run it, write its JSON as `OUT/te_graph.json`; `score.py` then uses it as a solo training baseline.

## Improved arm: dynamic SM split driven by the serving queue (`--serve-rate R --lend`, `ofa_r2.cu`)
Upstream decodes closed loop: the serve SMs generate tokens back to back, whatever the demand. With `--serve-rate R`, token s is released at
`t(token 0) + s/R`, which is an open-loop demand. A serve block that holds a decode ticket whose token is not yet due has an empty queue.
With `--lend` that block claims ONE task from the training ticket and runs it, using the same claim, dependency wait, completion counters and
step accounting as a training block. It then returns to its held decode ticket. The 21 serve SMs therefore train whenever the serving queue
is empty, and serve when tokens are due. A decode token waits at most one training task, which takes tens of µs. Upstream's ticket-order
argument keeps this deadlock-free: a lending block's training ticket only depends on lower tickets. The decode ticket it holds blocks only
later decode tokens.
The run prints `serve: demand R tok/s (open-loop), N tokens ... latency p50/p99, slo<=50ms <fraction>, lent train tasks <n>`. Latency is
measured from release to completion of each token, using per-token `%globaltimer` stamps. This code is compiled only into `ofa_h100_dyn` (`-DOFA_NODUS_DYN=1`).
`ofa_h100` (mk_train, mk_split) keeps upstream's device code; only `Ctx` has 40 more bytes. Checked with nvcc 12.8 for sm_90a: 128 registers and 236/264 B
spill stores/loads, the same as unmodified upstream (stack 1496 vs 1456 B). `ofa_h100_dyn`: 128 registers, 308/400 B spills. Lending needs the `mma.sync` backend (`OFA_GEMM_BACKEND=0`, the H100 build).
Scoring credits open-loop arms with min(served, demand) / solo vLLM. The MPS pair gets the same demand cap: its serving term becomes
min(vLLM-under-MPS rate, R) / solo. `vs_static_same_demand` = mk_dyn / mk_paced at the same R.

## Improved arm 2: Hopper wgmma + TMA GEMMs (`llm/h100/gemm_sm90.cuh`, `-DOFA_GEMM_BACKEND=3`)
Upstream has no Hopper GEMM. Its H100 build runs A100-era `mma.sync` GEMMs, while upstream's tcgen05 backend made B200 training about 2x faster
(95.8 s to 48.1 s). `gemm_sm90.cuh` keeps the task contract: one 128x128 tile per task, all 256 threads, the same layouts and epilogues.
- Loads: TMA 2D loads with 128B swizzle into a 3-stage mbarrier pipeline (99 KB smem, 2 blocks/SM fit).
- Math: `wgmma.mma_async m64n128k16` bf16 to fp32, warpgroup 0 on rows 0-63 and warpgroup 1 on rows 64-127.
- Descriptor strides: copied from upstream's B300-verified tcgen05 kit.
- Gate: phase `gemm_wg` runs upstream's `test_gemm_r2.cu` (all 15 model GEMM shapes vs an fp32 reference and cuBLAS, plus timing vs cuBLAS) on the
  sm90 backend. The `mk_wg_*` arms (training only, static split, dynamic split) run only if every shape PASSes.
- Variant `ofa_h100_wg1` / arm `mk_wg1_train`: 1 block per SM with 255 registers (208 used, 0 spills, vs 128 regs at 2 blocks/SM).
- Status: compiled with nvcc 12.8 for sm_90a but never executed, so expect the first GPU run to find descriptor or layout bugs.

## Score and gates (pre-registered, `score.py GATES`)
U = training tok/s / best solo full-GPU training tok/s (max of mk_train, pt_full, pt_graph, te_graph), plus serving tok/s / solo full-GPU vLLM tok/s.
Running the two jobs one after the other gives U = 1.0 (serial). Verdict is true only if a megakernel arm has:
- U >= 3.0 vs serial
- U >= 1.5 x the MPS pair's U
- mean TPOT <= 50 ms
- final loss within 0.02 of the PyTorch solo runs
- 0 errors

With two jobs, U cannot exceed 2.0 unless the megakernel beats both solo baselines. The run measures where it lands. CoSched: pass
`COSCHED_SUMMARY=<summary.json with useful_work>` to add `vs_cosched`.

## Artifacts
`OUT/meta.txt` holds the GPU, driver, fork and upstream SHAs and the split. Each phase writes `OUT/<phase>.log`. Baselines write `*.json`. The megakernel
writes `mk_*_loss.txt` and `mk_split_tokens.txt`. `ofa_h100.buildlog` has the ptxas registers, stack and spills. The scored result is `summary.json`.

## Runtime and failure modes
- Runtime: prep about 10 min (download plus tokenization); build about 2 min; each megakernel arm about 2-4 min; each PyTorch run about 3-5 min.
  vLLM startup is about 1-2 min per phase. A full run takes about 40-60 min.
- `refusing: GPU busy`: pick a free GPU. Never kill other jobs.
- Build: `-DGEMM_STAGES=4` (64 KB/block) x 2 blocks fits H100's 228 KB/SM. If the launch fails with too many resources, rebuild with
  `-DOFA_BLOCKS_PER_SM=1`.
- vLLM image on driver < 580 fails at CUDA init. Use a driver-580 box.
- MPS: the daemon runs on the host as root (`$SUDO`), and the containers share `/tmp/ofa_mps_$GPU/pipe` with `--ipc=host`.
- Loss mismatch between `mk_*` and `pt_*`: the bf16 GEMM paths differ, and upstream's A100 runs agree to about 0.005.
