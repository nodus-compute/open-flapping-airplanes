// ofa_r2.cu : round-2 copy of ofa.cu (B300). Only changes: includes gemm_backend_r2.cuh (backend 2 = gemm_sm100_v2.cuh) and GEMM task count uses tg::TILE_N (task tile 128 x TILE_N).
// B200 PORT (UNTESTED ON B200; sm_80 path regression-checked on A100, see README): SM partition derived from the device, see "SM PARTITION" below.
// ofa.cu : model "S" (GPT, 162M params) TRAINING + live DECODE + real DOOM inside ONE persistent kernel launch (A100, sm_80),
// with an embedded HTTP dashboard server (the three-panel open-flapping-airplanes page). Derived from llm/megakernel/ofa_llm.cu.
//
//  * (SMs x OFA_BLOCKS_PER_SM) persistent blocks x 256 threads (2 per SM by default: 216 on A100, 296 on B200). Each block reads %smid and takes a role by static SM partition:
//      smid <  INFER_SMS                    -> serves tokens (decode job)
//      INFER_SMS <= smid < INFER+DOOM_SMS   -> runs real Doom (1 tick task + 5 copy tasks per frame, paced at 35 Hz)
//      otherwise                            -> trains
//  * A job is a table of PHASES (a phase = N independent tasks of one kind). Tasks are claimed in order from a ticket; a phase is
//    claimable once all phases it depends on completed (per-phase completion counters, release/acquire). Tasks of different layers
//    overlap whenever their true data dependencies allow.
//  * All task bodies are the library tasks from llm/{gemm,attention,elementwise,decode} (unchanged).
//
//
// SM PARTITION (new in the b200 port; nothing is hard-coded any more):
//   --split auto | a100 | b200 | none   auto = by cudaDeviceProp.multiProcessorCount: 108 -> infer 17 / doom 6 / train 85 (A100, FA proportions),
//                                       148 -> infer 24 / doom 8 / train 116 (B200, FA's split), any other count -> FA's B200 proportions
//                                       (24/148, 8/148, Doom >= 1). a100 / b200 force that preset (useful with MPS-limited or other GPUs).
//   --infer-sms N --doom-sms N [--train-sms N]   explicit (override --split; as before). train defaults to the remainder; if smaller, extra SMs idle.
//   env OFA_SPLIT, OFA_INFER_SMS, OFA_DOOM_SMS, OFA_TRAIN_SMS are read when the flag is absent. Neither flag nor env => train only (old default).
//   Compile-time knobs: -DOFA_BLOCKS_PER_SM=2 (launch bounds + persistent blocks per SM), -DGEMM_STAGES=n (mma.sync backend smem = n*16 KB),
//   -DOFA_GEMM_BACKEND=0|1 (0 mma.sync, 1 tcgen05, see ../gemm_tcgen05), -DGEMM100_STAGES, -DGEMM100_LAYOUT_MASK, -DCLAIM_FETCHADD.
//   --probe-smid (default on, OFA_PROBE_SMID=0 to skip): a probe launch maps %smid -> dense rank 0..SMs-1, so partitions stay correct even if %smid is
//   not contiguous on a part with disabled SMs.
// Build:  ./build.sh sm_80|sm_100a [extra nvcc flags]   (nvcc -arch=sm_XX -O3 -std=c++17 --extended-lambda -rdc=true -maxrregcount=128 ... see build.sh)
// Run:    ./ofa_llm --steps 100 --data ~/data/tinystories_cl100k_train.u32 --init ~/data/init_flat_S.f32 --starts ~/data/starts_4000.u32
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../gemm_tcgen05/gemm_backend_r2.cuh"     // tg:: = GEMM backend (mma.sync default, tcgen05 with -DOFA_GEMM_BACKEND=1); includes llm/gemm/gemm.cuh
#include "../../attention/attention.cuh"
#include "../../elementwise/ew_all.cuh"
#include "decode_wide.cuh"

#include "doom_job.cuh"        // Doom job (device + host setup), see that header
#include "dashboard.cuh"       // HTTP dashboard server + sampler (host only)

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

// ------------------------------------------------------------------------------------------------ model config
#if defined(MODEL_F)
constexpr int L = 4, D = 2048, H = 32;
using DC = decode::ModelF;
#elif defined(MODEL_M)
constexpr int L = 16, D = 1024, H = 16;
using DC = decode::ModelM;
#else
constexpr int L = 12, D = 768, H = 12;
using DC = decode::ModelS;
#endif
constexpr int T = 512, V = 100352, VALID = 100277, B = 8, N = B * T, FF = 4 * D;
#ifndef CE_CACHE
#define CE_CACHE 8                      // 16-B vectors per thread of the logits row kept in smem by cross-entropy (8 = 32 KB; 48 = 192 KB caches ~98% of a 100352-wide row: only with the 1-block/SM tcgen05 build)
#endif
constexpr int GEMM_SMEM = tg::smem_bytes();           // 64 KB (mma.sync, 4 stages) / 98 KB (tcgen05, 3 stages x 32 KB + barriers)
constexpr int DYN_SMEM = GEMM_SMEM;
static_assert(attn::SMEM_DKDV <= DYN_SMEM && ew::ce_smem_bytes<CE_CACHE>() <= DYN_SMEM && ew::ln_bwd_smem_bytes<D>() <= DYN_SMEM, "smem budget");
static_assert(ew::embed_bwd_wte_smem_bytes(N) <= DYN_SMEM, "smem budget");
static_assert(decode::smem_bytes<DC>() <= DYN_SMEM, "smem budget");
#ifndef OFA_BLOCKS_PER_SM
#define OFA_BLOCKS_PER_SM 2
#endif
constexpr int BPS = OFA_BLOCKS_PER_SM;
constexpr int MAXPH = 1024;
#ifndef CLAIM_FETCHADD
#define CLAIM_FETCHADD 1
#endif

// ------------------------------------------------------------------------------------------------ phase tables
enum Kind { K_DATA, K_EMBED, K_LN_FWD, K_GEMM, K_ATT_FWD, K_ATT_PREP, K_ATT_BWD, K_CE, K_CE_MEAN, K_LN_BWD, K_LN_DG, K_EMB_WTE, K_EMB_WPE, K_ADAMW, K_DECODE };

struct PhaseDesc {
  int kind, ntasks, layout, epi;
  int tilesM, i0, i1, i2;
  tg::Args ga;
  const void* p[10];
  size_t n;
};

struct Job {
  int nph, tps, limit, last_ph;                 // phases, tasks per step, #steps, phase whose completion ends a step
  const PhaseDesc* ph;
  const int *prefix, *ntasks, *dep_off, *dep_cnt;
  const int2* deps;                             // (phase, step delta): need done[phase] >= ntasks[phase] * (step + 1 + delta)
  int* done;
  unsigned* ticket;
};

struct Ctx {
  Job train, dec, dec2;
  const uint32_t* data; u64 ntok; const uint32_t* starts; int nstarts; u64 seed;
  int* tok_in; int* tok_tgt;
  float lr_max, lr_min, beta1, beta2, eps, wd; int warmup, sched_steps;
  u64 *phase_busy, *phase_first, *phase_last; int prof_step;
  u64* step_done; float* loss_hist; u64* blk_busy; int* finished; u64* t_end;
  int infer_sms, infer2_sms, doom_sms, train_sms, use_decode;
  u64 run_ns; u64* run_start;
  u64 serve_period_ns; int lend, tok_cap;             // Nodus: open-loop serving demand (0 = closed loop) and idle-serve-block lending to training
  u64 *tok_done, *serve_base; unsigned* lent_tasks;
  u64 serve2_period_ns; u64 *tok_done2, *serve_base2;
  const int* smid_rank;                                 // %smid -> dense SM rank (identity unless the probe found gaps)
  decode::Params<DC> dp;
  decode::Params<DC> dp2;
  DoomCtx dm; Live* live;                               // Doom job (runs only on the Doom partition) + pinned live counters
};

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_vol(const unsigned* p) { return *reinterpret_cast<const volatile unsigned*>(p); }
__device__ __forceinline__ u64 ld_vol64(const u64* p) { return *reinterpret_cast<const volatile u64*>(p); }
__device__ __forceinline__ bool stopped(const Ctx& C) {
  if (C.run_ns && gtimer() >= ld_vol64(C.run_start) + C.run_ns) atomicExch(C.finished, 1);
  return *reinterpret_cast<volatile int*>(C.finished) != 0;
}
__global__ void clock_probe(u64* out) { *out = gtimer(); }
__device__ __forceinline__ int ld_vol(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ u64 splitmix(u64 z) { z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }

__device__ float lr_at(const Ctx& C, int s) {
  if (s < C.warmup) return C.lr_max * (float)(s + 1) / (float)C.warmup;
  const double p = (double)(s - C.warmup) / (double)(C.sched_steps - C.warmup);
  const double q = p < 1.0 ? p : 1.0;
  return (float)(C.lr_min + 0.5 * (C.lr_max - C.lr_min) * (1.0 + cos(3.14159265358979323846 * q)));
}
__device__ __forceinline__ ew::AdamHP adam_hp(const Ctx& C, int step) {
  ew::AdamHP hp; hp.lr = lr_at(C, step); hp.beta1 = C.beta1; hp.beta2 = C.beta2; hp.eps = C.eps; hp.weight_decay = C.wd; hp.step = step + 1; return hp;
}

template <int LAYOUT, int EPI>
__device__ __forceinline__ void gtile(const PhaseDesc& d, int tm, int tn, char* smem) { tg::tile<LAYOUT, EPI>(d.ga, tm, tn, smem); }

__device__ __noinline__ void run_gemm(const PhaseDesc& d, int local, char* smem) {
  const int tm = tg::RAW_TASK ? local : local % d.tilesM, tn = tg::RAW_TASK ? 0 : local / d.tilesM;   // backend 2: tile() takes the task index (a run of d.ga.nsub tiles)
  switch (d.layout * 8 + d.epi) {
    case gemm::NN * 8 + gemm::EPI_BF16:  gtile<gemm::NN, gemm::EPI_BF16>(d, tm, tn, smem); break;
    case gemm::NN * 8 + gemm::EPI_RES:   gtile<gemm::NN, gemm::EPI_RES>(d, tm, tn, smem); break;
    case gemm::NN * 8 + gemm::EPI_GELU:  gtile<gemm::NN, gemm::EPI_GELU>(d, tm, tn, smem); break;
    case gemm::NT * 8 + gemm::EPI_BF16:  gtile<gemm::NT, gemm::EPI_BF16>(d, tm, tn, smem); break;
    case gemm::NT * 8 + gemm::EPI_DGELU: gtile<gemm::NT, gemm::EPI_DGELU>(d, tm, tn, smem); break;
    case gemm::TN * 8 + gemm::EPI_F32:   gtile<gemm::TN, gemm::EPI_F32>(d, tm, tn, smem); break;
    default: break;
  }
}

// one training task
__device__ void run_train_task(const Ctx& C, const PhaseDesc& d, int local, int step, char* smem) {
  constexpr int BH = B * H, QT = T / 64;
  switch (d.kind) {
    case K_DATA: {                                           // sequence `local` of this step: tok_in = data[s..s+T), tok_tgt = data[s+1..s+T+1)
      u64 st;
      if (C.starts) st = C.starts[(size_t)(step % C.nstarts) * B + local];
      else st = splitmix(C.seed ^ ((u64)step * B + local)) % (C.ntok - T - 1);
      for (int t = threadIdx.x; t < T; t += 256) {
        C.tok_in[local * T + t] = (int)C.data[st + t];
        C.tok_tgt[local * T + t] = (int)C.data[st + t + 1];
      }
      break;
    }
    case K_EMBED: ew::embed_fwd_task<D, T>(local, (bf16*)d.p[0], (const bf16*)d.p[1], (const bf16*)d.p[2], (const int*)d.p[3], N); break;
    case K_LN_FWD: ew::ln_fwd_task<D>(local, (bf16*)d.p[0], (float*)d.p[1], (float*)d.p[2], (const bf16*)d.p[3], nullptr, nullptr, (const bf16*)d.p[4], N); break;
    case K_GEMM: run_gemm(d, local, smem); break;
    case K_ATT_FWD: {                                        // heavy (late) query tiles first
      const int qt = QT - 1 - local / BH, bh = local % BH;
      attn::attn_fwd_tile<T, H>((const bf16*)d.p[0], (bf16*)d.p[1], (float*)d.p[2], bh / H, bh % H, qt, smem);
      break;
    }
    case K_ATT_PREP: {
      const int rt = local / BH, bh = local % BH;
      attn::attn_bwd_prep_tile<T, H>((const bf16*)d.p[0], (const bf16*)d.p[1], (float*)d.p[2], bh / H, bh % H, rt);
      break;
    }
    case K_ATT_BWD: {                                        // first BH*QT tasks: dkdv (heavy = low kt first); then dq (heavy = high qt first)
      if (local < BH * QT) {
        const int kt = local / BH, bh = local % BH;
        attn::attn_bwd_dkdv_tile<T, H>((const bf16*)d.p[0], (const bf16*)d.p[1], (const float*)d.p[2], (const float*)d.p[3], (bf16*)d.p[4], bh / H, bh % H, kt, smem);
      } else {
        const int l2 = local - BH * QT, qt = QT - 1 - l2 / BH, bh = l2 % BH;
        attn::attn_bwd_dq_tile<T, H>((const bf16*)d.p[0], (const bf16*)d.p[1], (const float*)d.p[2], (const float*)d.p[3], (bf16*)d.p[4], bh / H, bh % H, qt, smem);
      }
      break;
    }
    case K_CE: ew::ce_task<V, CE_CACHE>(local, (bf16*)d.p[0], (const int*)d.p[1], (float*)d.p[2], N, 1.0f / (float)N, smem); break;
    case K_CE_MEAN: {
      ew::ce_mean_task((float*)d.p[0], (const float*)d.p[1], N, smem);
      __syncthreads();
      if (threadIdx.x == 0) { const float lv = ((const volatile float*)d.p[0])[0]; C.loss_hist[step] = lv; if (C.live) C.live->loss = lv; }
      break;
    }
    case K_LN_BWD: ew::ln_bwd_task<D>(local, (bf16*)d.p[0], (float*)d.p[1], (const bf16*)d.p[2], (const bf16*)d.p[3], (const float*)d.p[4], (const float*)d.p[5], (const bf16*)d.p[6], (const bf16*)d.p[7], N, smem); break;
    case K_LN_DG: ew::ln_dgamma_reduce_task<D>(local, (float*)d.p[0], (const float*)d.p[1], (N + 15) / 16); break;
    case K_EMB_WTE: {
      ew::embed_bwd_wte_task<D>(local, (float*)d.p[0], (const bf16*)d.p[1], (const int*)d.p[2], N, smem);
      if (d.i0) {                                            // fused AdamW of this task's 128 vocab rows (rows are final here); wte decays, so no nodecay ranges
        constexpr size_t RN = (size_t)ew::EMB_VCHUNK * D;
        const size_t off = (size_t)local * RN;
        const ew::AdamHP hp = adam_hp(C, step);
        for (int j = 0; j < (int)((RN + ew::OPT_CHUNK - 1) / ew::OPT_CHUNK); j++)
          ew::adamw_task<true>(j, (float*)d.p[3] + off, (float*)d.p[0] + off, (float*)d.p[4] + off, (float*)d.p[5] + off, (bf16*)d.p[6] + off, RN, hp, (const int2*)d.p[7], d.i1, nullptr);   // p[7]/i1: empty nodecay list, passed at run time so the compiler keeps the exact FMA pattern of the stand-alone AdamW (bit-identical results)
      }
      break;
    }
    case K_EMB_WPE: ew::embed_bwd_wpe_task<D, T>(local, (float*)d.p[0], (const bf16*)d.p[1], N); break;
    case K_ADAMW: {
      const ew::AdamHP hp = adam_hp(C, step);
      ew::adamw_task<true>(local, (float*)d.p[0], (float*)d.p[1], (float*)d.p[2], (float*)d.p[3], (bf16*)d.p[4], d.n, hp, (const int2*)d.p[5], d.i0, nullptr);
      break;
    }
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------ scheduler
// Nodus dynamic split: a serving block whose next token is not yet due (open-loop demand, queue empty) runs ONE training task
// (claimed from the training ticket exactly like a training block) and then returns to its held decode ticket.
#ifndef OFA_NODUS_DYN
#define OFA_NODUS_DYN 0
#endif
#if OFA_NODUS_DYN
__device__ __noinline__ bool lend_train_task(const Ctx& C, char* smem, int* lc) {
  __shared__ volatile int s_l[4];
  const Job& J = C.train;
  const int tid = threadIdx.x;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  if (tid == 0) {
    int res = 0, step = 0, ph = 0, local = 0;
    if (stopped(C) || !C.train_sms || ld_vol(J.ticket) >= total) res = -2;
    else {
      const unsigned g = ld_vol(J.ticket);
      if (g >= total) res = -2;
      else {
        step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
        int cph = lc[0];
        if (r < __ldg(J.prefix + cph)) cph = 0;
        while (r >= __ldg(J.prefix + cph + 1)) cph++;
        lc[0] = cph; ph = cph; local = r - __ldg(J.prefix + ph);
        if (!(step == lc[1] && ph == lc[2])) {
          const int off = __ldg(J.dep_off + ph), cnt = __ldg(J.dep_cnt + ph);
          bool ready = true;
          for (int k = 0; k < cnt && ready; k++) {
            const int2 e = __ldg(J.deps + off + k);
            const int need = __ldg(J.ntasks + e.x) * (step + 1 + e.y);
            if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;
          }
          if (ready) { lc[1] = step; lc[2] = ph; } else res = -2;
        }
        if (!res && atomicCAS(const_cast<unsigned*>(J.ticket), g, g + 1) != g) res = -2;
      }
    }
    s_l[0] = res; s_l[1] = step; s_l[2] = ph; s_l[3] = local;
  }
  __syncthreads();
  const int res = s_l[0], step = s_l[1], ph = s_l[2], local = s_l[3];
  __syncthreads();
  if (res < 0) return false;
  const PhaseDesc& d = J.ph[ph];
  u64 t0 = 0;
  if (tid == 0) t0 = gtimer();
  run_train_task(C, d, local, step, smem);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const u64 t1 = gtimer();
    atomicAdd(C.phase_busy + ph, t1 - t0);
    atomicAdd(C.lent_tasks, 1u);
    const int old = atomicAdd(J.done + ph, 1);
    if (ph == J.last_ph && old == __ldg(J.ntasks + ph) * (step + 1) - 1) {
      C.step_done[step] = t1;
      if (C.live) C.live->steps = (unsigned)(step + 1);
      if (step == J.limit - 1 && !C.run_ns) { *C.t_end = t1; __threadfence(); *reinterpret_cast<volatile int*>(C.finished) = 1; }
    }
  }
  __syncthreads();
  return true;
}
#endif

__device__ void job_loop(const Ctx& C, const Job& J, int job_id, char* smem) {
  __shared__ volatile int s_sel[4];
  const int tid = threadIdx.x;
  const u64 period = job_id == 2 ? C.serve2_period_ns : C.serve_period_ns;
  u64* const serve_base = job_id == 2 ? C.serve_base2 : C.serve_base;
  u64* const tok_done = job_id == 2 ? C.tok_done2 : C.tok_done;
  int cph = 0, rstep = -1, rph = -1;                    // thread 0: phase cache and "this (step, phase) is known ready"
#if OFA_NODUS_DYN
  int pend = 0; unsigned pend_g = 0; int lcache[3] = {0, -1, -1};   // thread 0: held decode ticket while lending; lending phase cache
#endif
  u64 busy = 0;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  if (job_id == 0) tg::block_begin();                   // tcgen05 backend with -DG2_PERSIST_TMEM: allocate TMEM once per block (no-op otherwise)
  for (;;) {
    if (tid == 0) {
      int res = 0, step = 0, ph = 0, local = 0;
#if CLAIM_FETCHADD
      // Unconditional fetch-add claim: never retries (the CAS version serialised all blocks at ~2 claims/us). Dependencies always point to
      // LOWER tickets (phases are topologically ordered, step s+1 depends on step s), so a block that holds a claim while its inputs are
      // not finished cannot deadlock: the lowest unfinished ticket always has all its dependencies finished.
      if ((job_id > 0 || C.run_ns) && stopped(C)) { res = -2; }
      else {
#if OFA_NODUS_DYN
        unsigned g;
        if (pend) { g = pend_g; pend = 0; } else g = atomicAdd(const_cast<unsigned*>(J.ticket), 1u);
#else
        const unsigned g = atomicAdd(const_cast<unsigned*>(J.ticket), 1u);
#endif
        if (g >= total) res = -2;
        else {
          step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
          if (r < __ldg(J.prefix + cph)) cph = 0;
          while (r >= __ldg(J.prefix + cph + 1)) cph++;
          ph = cph; local = r - __ldg(J.prefix + ph);
          if (!(step == rstep && ph == rph)) {
            const int off = __ldg(J.dep_off + ph), cnt = __ldg(J.dep_cnt + ph);
            for (;;) {
              if (C.run_ns && stopped(C)) { res = -2; break; }
              bool ready = true;
              for (int k = 0; k < cnt && ready; k++) {
                const int2 e = __ldg(J.deps + off + k);
                const int need = __ldg(J.ntasks + e.x) * (step + 1 + e.y);
                if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;
              }
#if OFA_NODUS_DYN
              if (ready && job_id > 0 && period && ph == 0 && (step > 0 || C.run_ns)) {
                const u64 base = ld_vol64(C.run_ns ? C.run_start : serve_base);
                if (base == 0) ready = false;
                else if (gtimer() < base + (u64)step * period) {
                  ready = false;
                  if (C.lend && !ld_vol(C.finished)) { pend = 1; pend_g = g; res = -3; break; }
                }
              }
#endif
              if (ready) { rstep = step; rph = ph; break; }
              if (job_id > 0 && stopped(C)) { res = -2; break; }
              __nanosleep(job_id > 0 ? 60 : 100);
            }
          }
        }
      }
#else
      for (;;) {
        if ((job_id > 0 || C.run_ns) && stopped(C)) { res = -2; break; }
        const unsigned g = ld_vol(J.ticket);
        if (g >= total) { res = -2; break; }
        step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
        if (r < __ldg(J.prefix + cph)) cph = 0;
        while (r >= __ldg(J.prefix + cph + 1)) cph++;
        ph = cph;
        bool ready = (step == rstep && ph == rph);
        if (!ready) {
          ready = true;
          const int off = __ldg(J.dep_off + ph), cnt = __ldg(J.dep_cnt + ph);
          for (int k = 0; k < cnt && ready; k++) {
            const int2 e = __ldg(J.deps + off + k);
            const int need = __ldg(J.ntasks + e.x) * (step + 1 + e.y);
            if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;
          }
          if (ready) { rstep = step; rph = ph; }
        }
        if (!ready) { __nanosleep(job_id == 1 ? 100 : 200); continue; }
        if (atomicCAS(const_cast<unsigned*>(J.ticket), g, g + 1) == g) { local = r - __ldg(J.prefix + ph); break; }
      }
#endif
      s_sel[0] = res; s_sel[1] = step; s_sel[2] = ph; s_sel[3] = local;
    }
    __syncthreads();
    const int res = s_sel[0];
#if OFA_NODUS_DYN
    if (res == -3) { __syncthreads(); lend_train_task(C, smem, lcache); continue; }
#endif
    if (res < 0) break;
    const int step = s_sel[1], ph = s_sel[2], local = s_sel[3];
    __syncthreads();
    const PhaseDesc& d = J.ph[ph];
    u64 t0 = 0;
    if (tid == 0) t0 = gtimer();
    if (job_id == 0) run_train_task(C, d, local, step, smem);
    else decode::run_task<DC>(job_id == 2 ? C.dp2 : C.dp, d.i0, local, smem);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const u64 t1 = gtimer();
      busy += t1 - t0;
      if (job_id == 0) {
        atomicAdd(C.phase_busy + ph, t1 - t0);
        if (step == C.prof_step) { atomicMin(C.phase_first + ph, t0); atomicMax(C.phase_last + ph, t1); }
      }
      const int old = atomicAdd(J.done + ph, 1);
#if OFA_NODUS_DYN
      if (job_id > 0 && ph == J.last_ph && old == __ldg(J.ntasks + ph) * (step + 1) - 1) {
        if (step < C.tok_cap) tok_done[step] = t1;
        if (step == 0) { __threadfence(); *reinterpret_cast<volatile u64*>(serve_base) = t1; }
      }
#endif
      if (job_id == 0 && ph == J.last_ph && old == __ldg(J.ntasks + ph) * (step + 1) - 1) {     // last task of the step
        C.step_done[step] = t1;
        if (C.live) C.live->steps = (unsigned)(step + 1);
        if (step == J.limit - 1 && !C.run_ns) { *C.t_end = t1; __threadfence(); *reinterpret_cast<volatile int*>(C.finished) = 1; }
      }
    }
  }
  if (job_id == 0) tg::block_end();
  if (tid == 0) C.blk_busy[blockIdx.x * (C.infer2_sms ? 3 : 2) + job_id] = busy;
}

__global__ void __launch_bounds__(256, OFA_BLOCKS_PER_SM) ofa_kernel(const Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  unsigned hw; asm volatile("mov.u32 %0, %%smid;" : "=r"(hw));
  const int smid = C.smid_rank ? C.smid_rank[hw] : (int)hw;                          // dense rank of this SM
  char* smem = reinterpret_cast<char*>(dsm);
  const int isms = C.use_decode ? C.infer_sms : 0, dsms = C.use_decode ? C.doom_sms : 0;
  if (smid < isms) job_loop(C, C.dec, 1, smem);
  else if (smid < isms + C.infer2_sms) job_loop(C, C.dec2, 2, smem);
  else if (smid < isms + C.infer2_sms + dsms) { if (C.dm.on) doom_loop(C.dm); else return; }
  else if (smid < isms + C.infer2_sms + dsms + C.train_sms) job_loop(C, C.train, 0, smem);
  else return;                                                                       // SMs beyond the requested train share stay idle
}

// Probe: one resident block per slot records %smid, so the host can build the dense rank table (same launch shape as ofa_kernel).
__global__ void __launch_bounds__(256, OFA_BLOCKS_PER_SM) smid_probe(unsigned* seen) {
  if (threadIdx.x == 0) {
    unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    atomicOr(&seen[id >> 5], 1u << (id & 31));
    const long long t0 = clock64(); while (clock64() - t0 < 4000000ll) {}                // ~2 ms: keeps every block resident so each SM gets its share
  }
}

// ------------------------------------------------------------------------------------------------ host: model layout
struct Layout {
  size_t wte, wpe, gf, total;
  size_t g1[L], wqkv[L], wo[L], g2[L], wup[L], wdn[L];
  Layout() {
    size_t o = 0;
    wte = o; o += (size_t)V * D; wpe = o; o += (size_t)T * D;
    for (int l = 0; l < L; l++) {
      g1[l] = o; o += D; wqkv[l] = o; o += (size_t)D * 3 * D; wo[l] = o; o += (size_t)D * D;
      g2[l] = o; o += D; wup[l] = o; o += (size_t)D * FF; wdn[l] = o; o += (size_t)FF * D;
    }
    gf = o; o += D; total = o;
  }
};

struct Builder {
  std::vector<PhaseDesc> ph; std::vector<int> dep_off, dep_cnt, ntasks; std::vector<int2> deps; std::vector<std::string> names;
  int add(const std::string& name, PhaseDesc d, std::vector<int> after, std::vector<int2> prev = {}) {
    dep_off.push_back((int)deps.size());
    for (int a : after) if (a >= 0) deps.push_back(make_int2(a, 0));
    for (auto p : prev) deps.push_back(p);
    dep_cnt.push_back((int)deps.size() - dep_off.back());
    ph.push_back(d); ntasks.push_back(d.ntasks); names.push_back(name);
    return (int)ph.size() - 1;
  }
};

static int g_slots = 148;                                   // train block slots (SMs x blocks/SM), set in main before the graph is built
static PhaseDesc mk(int kind, int ntasks) { PhaseDesc d; memset(&d, 0, sizeof d); d.kind = kind; d.ntasks = ntasks; return d; }
static PhaseDesc mkgemm(int layout, int epi, const void* A, int lda, const void* Bm, int ldb, void* Cc, int ldc, void* aux, int ldaux, int M, int Nn, int K, int accumulate) {
  // tcgen05 backend: wide/narrow config and tiles-per-task (nsub) chosen from the number of train block slots (OFA_WIDE / OFA_NSUB env override for experiments)
  int wide = 0, nsub = 1;
  static const double slotmult = getenv("OFA_SLOTMULT") ? atof(getenv("OFA_SLOTMULT")) : 1.0;   // nsub = ceil(tiles / (slots*slotmult)): >1 gives more, smaller tasks
  tg::choose(M, Nn, std::max(1, (int)(g_slots * slotmult)), &wide, &nsub, (layout == gemm::TN && getenv("OFA_TN_NARROW")) ? 0 : (getenv("OFA_WIDE") ? atoi(getenv("OFA_WIDE")) : -1), getenv("OFA_NSUB") ? atoi(getenv("OFA_NSUB")) : 0);
  PhaseDesc d = mk(K_GEMM, 1);
  d.layout = layout; d.epi = epi; d.tilesM = M / 128;
  d.ga.A = (const bf16*)A; d.ga.B = (const bf16*)Bm; d.ga.C = Cc; d.ga.aux = (bf16*)aux;
  d.ga.lda = lda; d.ga.ldb = ldb; d.ga.ldc = ldc; d.ga.ldaux = ldaux; d.ga.M = M; d.ga.N = Nn; d.ga.K = K; d.ga.accumulate = accumulate;
  tg::prepare(d.ga, layout, wide, nsub);                    // TMA descriptors + wide/nsub for the tcgen05 backend (no-op for mma.sync)
  if (!tg::args_ok(d.ga)) { fprintf(stderr, "bad GEMM args layout=%d epi=%d M=%d N=%d K=%d\n", layout, epi, M, Nn, K); exit(1); }
  d.ntasks = tg::ntasks(d.ga);
  return d;
}

template <class X> static X* dalloc(size_t n) { X* p; CK(cudaMalloc(&p, n * sizeof(X))); CK(cudaMemset(p, 0, n * sizeof(X))); return p; }

static std::vector<float> read_f32(const char* path, size_t n) {
  std::vector<float> v(n); FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); } fclose(f); return v;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- SM partition (see header comment). FA's published B200 split is 116/24/8 of 148; the A100 proportional split is 85/17/6 of 108.
struct SmSplit { int infer, doom, train; const char* src; };
static SmSplit split_for(const char* mode, int nsm) {
  if (!strcmp(mode, "a100") || (!strcmp(mode, "auto") && nsm == 108)) return {17, 6, 85, "A100 preset 85/17/6"};
  if (!strcmp(mode, "b200") || (!strcmp(mode, "auto") && nsm == 148)) return {24, 8, 116, "B200 preset 116/24/8"};
  int doom = std::max(1, (int)lround(nsm * 8.0 / 148)), infer = std::max(1, (int)lround(nsm * 24.0 / 148));   // FA's B200 proportions
  return {infer, doom, nsm - infer - doom, "proportional to FA's 116/24/8 of 148"};
}
static int env_int(const char* n, int dflt) { const char* v = getenv(n); return v && *v ? atoi(v) : dflt; }

static double serve_rate = 0; static int lend = 0;   // Nodus: --serve-rate R (tok/s, open loop), --lend

int main(int argc, char** argv) {
  int steps = 100, sched_steps = 4000, prof_step = -1, infer_sms = 0, doom_sms = 0, prompt_n = 0, nopt = 0, dump_every = 0, opt_mode = 2, train_sms_arg = -1; std::string split_mode = getenv("OFA_SPLIT") ? getenv("OFA_SPLIT") : "";
  std::string data_path, init_path, starts_path, dump_grads, dump_w, loss_out, text_out, prompt_path; float temp = 0.8f; u64 seed = 1234; int dec_seed_steps = 1 << 20;
  int port = 0; std::string wad_path, vocab_path, prompt_text = "Once upon a time", index_path = "index.html"; bool no_doom = false;
  int infer2_sms = 0; double serve2_rate = 0, duration_s = 0;
  std::string metrics_out, ready_file, go_file;
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--steps")) steps = atoi(argv[++i]); else if (is("--sched-steps")) sched_steps = atoi(argv[++i]);
    else if (is("--data")) data_path = argv[++i]; else if (is("--init")) init_path = argv[++i]; else if (is("--starts")) starts_path = argv[++i];
    else if (is("--dump-grads")) dump_grads = argv[++i]; else if (is("--dump-w")) dump_w = argv[++i]; else if (is("--loss-out")) loss_out = argv[++i];
    else if (is("--infer-sms")) infer_sms = atoi(argv[++i]); else if (is("--doom-sms")) doom_sms = atoi(argv[++i]);
    else if (is("--train-sms")) train_sms_arg = atoi(argv[++i]); else if (is("--split")) split_mode = argv[++i];
    else if (is("--prof-step")) prof_step = atoi(argv[++i]); else if (is("--seed")) seed = atoll(argv[++i]);
    else if (is("--temp")) temp = atof(argv[++i]); else if (is("--text-out")) text_out = argv[++i]; else if (is("--prompt")) prompt_path = argv[++i];
    else if (is("--port")) port = atoi(argv[++i]); else if (is("--wad")) wad_path = argv[++i]; else if (is("--vocab")) vocab_path = argv[++i];
    else if (is("--prompt-text")) prompt_text = argv[++i]; else if (is("--index")) index_path = argv[++i];
    else if (is("--serve-rate")) serve_rate = atof(argv[++i]); else if (!strcmp(argv[i], "--lend")) lend = 1;
    else if (is("--infer2-sms")) infer2_sms = atoi(argv[++i]);
    else if (is("--serve2-rate")) serve2_rate = atof(argv[++i]);
    else if (is("--duration-s")) duration_s = atof(argv[++i]);
    else if (is("--metrics-out")) metrics_out = argv[++i];
    else if (is("--ready-file")) ready_file = argv[++i];
    else if (is("--go-file")) go_file = argv[++i];
    else if (!strcmp(argv[i], "--no-doom")) no_doom = true;
    else if (!strcmp(argv[i], "--no-opt")) nopt = 1; else if (is("--opt-mode")) opt_mode = atoi(argv[++i]);
  }
  if (infer2_sms < 0 || steps < 2 || !std::isfinite(duration_s) || duration_s < 0 ||
      (duration_s > 0 && (metrics_out.empty() || !no_doom || port || !std::isfinite(serve_rate) ||
        !std::isfinite(serve2_rate) || (infer_sms > 0 && serve_rate <= 0) ||
        (infer2_sms > 0 && serve2_rate <= 0))) || (infer2_sms && duration_s <= 0)) {
    fprintf(stderr, "invalid multitask configuration: fixed duration, metrics, positive enabled rates and --no-doom required\n"); return 2;
  }
#if !OFA_NODUS_DYN || !CLAIM_FETCHADD
  if (duration_s > 0) { fprintf(stderr, "fixed-window multitask requires OFA_NODUS_DYN=1 and CLAIM_FETCHADD=1\n"); return 2; }
#endif
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const int nsm = prop.multiProcessorCount;
  {  // resolve the partition: explicit flags > env > --split/OFA_SPLIT > (none: train only)
    if (infer_sms == 0) infer_sms = env_int("OFA_INFER_SMS", 0);
    if (doom_sms == 0) doom_sms = env_int("OFA_DOOM_SMS", 0);
    if (train_sms_arg < 0) train_sms_arg = env_int("OFA_TRAIN_SMS", -1);
    const char* src = "explicit";
    if (infer_sms == 0) doom_sms = 0;                                  // Doom runs only next to the decode job (old behaviour)
    if (infer_sms == 0 && doom_sms == 0 && !split_mode.empty() && split_mode != "none") {
      if (split_mode != "auto" && split_mode != "a100" && split_mode != "b200") { fprintf(stderr, "--split must be auto|a100|b200|none\n"); return 2; }
      SmSplit sp = split_for(split_mode.c_str(), nsm); infer_sms = sp.infer; doom_sms = sp.doom; if (train_sms_arg < 0) train_sms_arg = sp.train; src = sp.src;
    }
    if (infer_sms < 0 || doom_sms < 0 || infer_sms + infer2_sms + doom_sms > nsm) { fprintf(stderr, "serving SM budgets exceed GPU\n"); return 2; }
    if (train_sms_arg < 0 || train_sms_arg > nsm - infer_sms - infer2_sms - doom_sms) train_sms_arg = nsm - infer_sms - infer2_sms - doom_sms;
    if (infer_sms > 0 || doom_sms > 0) printf("SM partition (%s): infer %d, doom %d, train %d, idle %d of %d SMs\n", src, infer_sms, doom_sms, train_sms_arg, nsm - infer_sms - doom_sms - train_sms_arg, nsm);
  }
  const bool use_decode = infer_sms > 0;
  if (!infer_sms && !infer2_sms && !train_sms_arg) { fprintf(stderr, "no active workload\n"); return 2; }
  const bool doom_on = use_decode && doom_sms > 0 && !no_doom && !wad_path.empty();
  DoomCtx dm{}; unsigned char* fb_host = nullptr; Live* live_host = nullptr;
  if (!doom_on) CK(cudaDeviceSetLimit(cudaLimitStackSize, duration_s > 0 ? 8192 : 4096));
  if ((doom_on || port > 0) && !doom_host_setup(dm, wad_path, doom_on, fb_host, live_host)) return 1;
  CK(cudaFuncSetAttribute(ofa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ofa_kernel, 256, DYN_SMEM));
  if (getenv("OFA_OCC_SCAN")) {   // diagnostic: blocks/SM the occupancy API reports for several dynamic-smem sizes
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, ofa_kernel));
    printf("ofa_kernel: regs %d, static smem %zu, local %zu, maxThreads %d, ptx %d bin %d\n", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock, fa.ptxVersion, fa.binaryVersion);
    CK(cudaFuncSetAttribute(ofa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
    for (int kb : {32, 48, 64, 80, 90, 96, 98, 100, 101, 102, 104, 110, 113}) { int o = -1; cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, ofa_kernel, 256, kb * 1024); printf("  occ(dyn %d KB) = %d (%s)\n", kb, o, cudaGetErrorString(e)); }
    cudaGetLastError();
    CK(cudaFuncSetAttribute(ofa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM));
  }
  const int nblocks = nsm * std::min(occ, BPS);
  g_slots = env_int("OFA_GRAPH_SMS", train_sms_arg) * std::min(occ, BPS);
  {  // shared-memory opt-in limits come from the device, not from constants (A100: 164 KB/SM, 163 KB opt-in/block; H100/B200: 228 KB/SM, 227 KB opt-in/block)
    int optin = 0, per_sm = 0, reserved = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0));
    const int fit = per_sm / (DYN_SMEM + reserved);
    printf("GPU %s (sm_%d%d), %d SMs; smem/SM %d B, opt-in/block %d B, reserved/block %d B; kernel needs %d B dyn -> %d blocks/SM fit by smem, %d by occupancy (target %d)\n",
           prop.name, prop.major, prop.minor, nsm, per_sm, optin, reserved, DYN_SMEM, fit, occ, BPS);
    if (DYN_SMEM > optin) { fprintf(stderr, "dyn smem %d exceeds the device opt-in limit %d: lower -DGEMM_STAGES / -DGEMM100_STAGES\n", DYN_SMEM, optin); return 2; }
    printf("  smem headroom per SM at %d blocks/SM: %d B (a larger GEMM_STAGES / GEMM100_STAGES can use it)\n", BPS, per_sm - BPS * (DYN_SMEM + reserved));
    printf("%d persistent blocks (%d per SM)\n", nblocks, std::min(occ, BPS));
  }
  if (occ < BPS) printf("WARNING: only %d block/SM fits (registers/smem); performance will differ from the %d/SM design\n", occ, BPS);
  std::vector<int> rank_h; int* d_rank = nullptr;      // %smid -> dense rank
  if (env_int("OFA_PROBE_SMID", 1)) {
    CK(cudaFuncSetAttribute(smid_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM));
    unsigned* seen; CK(cudaMallocManaged(&seen, 64 * 4)); for (int i = 0; i < 64; i++) seen[i] = 0;
    smid_probe<<<nblocks, 256, DYN_SMEM>>>(seen); CK(cudaDeviceSynchronize());
    rank_h.assign(2048, 0); int n = 0, mx = -1; for (int id = 0; id < 2048; id++) if (seen[id >> 5] >> (id & 31) & 1) { rank_h[id] = n++; mx = id; }
    printf("smid probe: %d distinct SMs seen, max %%smid %d (device reports %d SMs)%s\n", n, mx, nsm, (n == nsm && mx == nsm - 1) ? "" : "  <-- NOT contiguous/complete, using rank table");
    if (n != nsm) { printf("WARNING: probe saw %d of %d SMs (MPS limit or a busy GPU?); partition may be wrong. Re-run on an idle GPU or set OFA_PROBE_SMID=0.\n", n, nsm); }
    else if (mx != nsm - 1) { CK(cudaMalloc(&d_rank, 2048 * 4)); CK(cudaMemcpy(d_rank, rank_h.data(), 2048 * 4, cudaMemcpyHostToDevice)); }
    cudaFree(seen);
  }

  const Layout lay; const size_t NP = lay.total;
  printf("model: L=%d d=%d H=%d T=%d V=%d  params=%.1fM  tokens/step=%d\n", L, D, H, T, V, NP / 1e6, N);

  // ---------------- parameters
  float *W32 = dalloc<float>(NP), *G32 = dalloc<float>(NP), *M32 = dalloc<float>(NP), *V32 = dalloc<float>(NP);
  bf16* Wbf = dalloc<bf16>(NP);
  {
    std::vector<float> h(NP);
    if (!init_path.empty()) { h = read_f32(init_path.c_str(), NP); printf("init weights from %s\n", init_path.c_str()); }
    else {                                                   // N(0,0.02), residual projections scaled by 1/sqrt(2L), gammas 1 (host Box-Muller)
      u64 s = 88172645463325252ull; auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double)((s >> 11) + 1) / 9007199254740993.0; };
      auto fill = [&](size_t off, size_t n, float sc) { for (size_t i = 0; i < n; i += 2) { double a = rnd(), b = rnd(), r = sqrt(-2 * log(a)); h[off + i] = (float)(0.02 * sc * r * cos(6.283185307179586 * b)); if (i + 1 < n) h[off + i + 1] = (float)(0.02 * sc * r * sin(6.283185307179586 * b)); } };
      const float sc = 1.f / sqrtf(2.f * L);
      fill(lay.wte, (size_t)V * D, 1); fill(lay.wpe, (size_t)T * D, 1);
      for (int l = 0; l < L; l++) {
        for (int k = 0; k < D; k++) { h[lay.g1[l] + k] = 1.f; h[lay.g2[l] + k] = 1.f; }
        fill(lay.wqkv[l], (size_t)D * 3 * D, 1); fill(lay.wo[l], (size_t)D * D, sc); fill(lay.wup[l], (size_t)D * FF, 1); fill(lay.wdn[l], (size_t)FF * D, sc);
      }
      for (int k = 0; k < D; k++) h[lay.gf + k] = 1.f;
      printf("init weights: random (host RNG)\n");
    }
    CK(cudaMemcpy(W32, h.data(), NP * 4, cudaMemcpyHostToDevice));
    std::vector<bf16> hb(NP); for (size_t i = 0; i < NP; i++) hb[i] = __float2bfloat16(h[i]);
    CK(cudaMemcpy(Wbf, hb.data(), NP * 2, cudaMemcpyHostToDevice));
  }
  std::vector<int2> nodecay; for (int l = 0; l < L; l++) { nodecay.push_back(make_int2((int)lay.g1[l], (int)lay.g1[l] + D)); nodecay.push_back(make_int2((int)lay.g2[l], (int)lay.g2[l] + D)); }
  nodecay.push_back(make_int2((int)lay.gf, (int)lay.gf + D)); std::sort(nodecay.begin(), nodecay.end(), [](int2 a, int2 b) { return a.x < b.x; });
  int2* d_nodecay = dalloc<int2>(nodecay.size()); CK(cudaMemcpy(d_nodecay, nodecay.data(), nodecay.size() * sizeof(int2), cudaMemcpyHostToDevice));

  // ---------------- data
  std::vector<uint32_t> hdata; uint32_t* d_data; u64 ntok;
  if (!data_path.empty()) {
    FILE* f = fopen(data_path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", data_path.c_str()); return 1; }
    fseek(f, 0, SEEK_END); ntok = ftell(f) / 4; fseek(f, 0, SEEK_SET); hdata.resize(ntok);
    if (fread(hdata.data(), 4, ntok, f) != ntok) return 1; fclose(f); printf("data: %s (%llu tokens)\n", data_path.c_str(), ntok);
  } else {                                                  // synthetic, learnable: next = f(prev)
    ntok = 1 << 22; hdata.resize(ntok); uint64_t x = 12345; for (u64 i = 0; i < ntok; i++) { hdata[i] = (uint32_t)(x % 100000); x = (x * 40503ull + 12345ull) % 100000ull; }
    printf("data: synthetic (%llu tokens)\n", ntok);
  }
  CK(cudaMalloc(&d_data, ntok * 4)); CK(cudaMemcpy(d_data, hdata.data(), ntok * 4, cudaMemcpyHostToDevice));
  uint32_t* d_starts = nullptr; int nstarts = 0;
  if (!starts_path.empty()) {
    FILE* f = fopen(starts_path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", starts_path.c_str()); return 1; }
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / 4; fseek(f, 0, SEEK_SET); std::vector<uint32_t> hs(n); if (fread(hs.data(), 4, n, f) != n) return 1; fclose(f);
    nstarts = (int)(n / B); CK(cudaMalloc(&d_starts, n * 4)); CK(cudaMemcpy(d_starts, hs.data(), n * 4, cudaMemcpyHostToDevice)); printf("batch starts: %s (%d steps)\n", starts_path.c_str(), nstarts);
  }
  int *tok_in = dalloc<int>(N), *tok_tgt = dalloc<int>(N);

  // ---------------- activations / gradients (per-layer buffers: no cross-layer hazards)
  auto act = [&](size_t n) { return dalloc<bf16>(n); };
  bf16 *xs[L + 1], *ln1[L], *qkv[L], *ob[L], *hb_[L], *ln2[L], *upre[L], *uact[L];
  float *mean1[L], *rstd1[L], *mean2[L], *rstd2[L], *lse[L];
  bf16 *dxs[L + 1], *dpre[L], *dln2[L], *dh[L], *dob[L], *dqkv[L], *dln1[L]; float *Dr[L], *dgp1[L], *dgp2[L];
  for (int l = 0; l <= L; l++) { xs[l] = act((size_t)N * D); dxs[l] = act((size_t)N * D); }
  for (int l = 0; l < L; l++) {
    ln1[l] = act((size_t)N * D); qkv[l] = act((size_t)N * 3 * D); ob[l] = act((size_t)N * D); hb_[l] = act((size_t)N * D); ln2[l] = act((size_t)N * D);
    upre[l] = act((size_t)N * FF); uact[l] = act((size_t)N * FF);
    mean1[l] = dalloc<float>(N); rstd1[l] = dalloc<float>(N); mean2[l] = dalloc<float>(N); rstd2[l] = dalloc<float>(N); lse[l] = dalloc<float>((size_t)B * H * T);
    dpre[l] = act((size_t)N * FF); dln2[l] = act((size_t)N * D); dh[l] = act((size_t)N * D); dob[l] = act((size_t)N * D); dqkv[l] = act((size_t)N * 3 * D); dln1[l] = act((size_t)N * D);
    Dr[l] = dalloc<float>((size_t)B * H * T); dgp1[l] = dalloc<float>((size_t)((N + 15) / 16) * D); dgp2[l] = dalloc<float>((size_t)((N + 15) / 16) * D);
  }
  bf16 *lnf = act((size_t)N * D), *logits = act((size_t)N * V), *dxf = act((size_t)N * D);
  float *meanf = dalloc<float>(N), *rstdf = dalloc<float>(N), *lossv = dalloc<float>(N), *meanloss = dalloc<float>(1), *dgpf = dalloc<float>((size_t)((N + 15) / 16) * D);
  size_t free_b, total_b; CK(cudaMemGetInfo(&free_b, &total_b)); printf("device memory used: %.1f GB of %.1f GB\n", (total_b - free_b) / 1e9, total_b / 1e9);

  // ---------------- training phase table
  Builder Bd;
  const bf16* wte_b = Wbf + lay.wte; const bf16* wpe_b = Wbf + lay.wpe;
  int p_data, p_embed, p_prev = -1;
  const bool fuse_opt = !nopt && opt_mode > 0; const bool fuse_wte = fuse_opt && opt_mode > 1;                              // AdamW fused into the embedding backward (wte) and run per layer group as soon as that layer's grads are final
  auto pv = [&](int n) { return std::vector<int2>(fuse_opt ? n : 0, make_int2(-1, -1)); };   // placeholders for previous-step deps, patched once the AdamW phase ids exist
  { PhaseDesc d = mk(K_DATA, B); p_data = Bd.add("data", d, {}, {make_int2(-1, -1)}); }   // dep on the last phase of the previous step: patched below
  { PhaseDesc d = mk(K_EMBED, (N + 15) / 16); d.p[0] = xs[0]; d.p[1] = wte_b; d.p[2] = wpe_b; d.p[3] = tok_in; p_embed = Bd.add("embed", d, {p_data}, pv(2)); }
  int cur = p_embed; int p_ln1f[L];
  for (int l = 0; l < L; l++) {
    const bf16 *g1b = Wbf + lay.g1[l], *g2b = Wbf + lay.g2[l], *wq = Wbf + lay.wqkv[l], *wo = Wbf + lay.wo[l], *wu = Wbf + lay.wup[l], *wd = Wbf + lay.wdn[l];
    char nm[64];
    PhaseDesc d = mk(K_LN_FWD, (N + 15) / 16); d.p[0] = ln1[l]; d.p[1] = mean1[l]; d.p[2] = rstd1[l]; d.p[3] = xs[l]; d.p[4] = g1b; snprintf(nm, 64, "L%d ln1", l); int a = Bd.add(nm, d, {cur}, pv(1)); p_ln1f[l] = a;
    d = mkgemm(gemm::NN, gemm::EPI_BF16, ln1[l], D, wq, 3 * D, qkv[l], 3 * D, nullptr, 0, N, 3 * D, D, 0); snprintf(nm, 64, "L%d qkv", l); a = Bd.add(nm, d, {a});
    d = mk(K_ATT_FWD, B * H * (T / 64)); d.p[0] = qkv[l]; d.p[1] = ob[l]; d.p[2] = lse[l]; snprintf(nm, 64, "L%d attn", l); a = Bd.add(nm, d, {a});
    d = mkgemm(gemm::NN, gemm::EPI_RES, ob[l], D, wo, D, hb_[l], D, xs[l], D, N, D, D, 0); snprintf(nm, 64, "L%d wo+res", l); a = Bd.add(nm, d, {a});
    d = mk(K_LN_FWD, (N + 15) / 16); d.p[0] = ln2[l]; d.p[1] = mean2[l]; d.p[2] = rstd2[l]; d.p[3] = hb_[l]; d.p[4] = g2b; snprintf(nm, 64, "L%d ln2", l); a = Bd.add(nm, d, {a});
    d = mkgemm(gemm::NN, gemm::EPI_GELU, ln2[l], D, wu, FF, uact[l], FF, upre[l], FF, N, FF, D, 0); snprintf(nm, 64, "L%d up+gelu", l); a = Bd.add(nm, d, {a});
    d = mkgemm(gemm::NN, gemm::EPI_RES, uact[l], FF, wd, D, xs[l + 1], D, hb_[l], D, N, D, FF, 0); snprintf(nm, 64, "L%d down+res", l); cur = Bd.add(nm, d, {a});
  }
  int p_lnf, p_logits, p_ce, p_cem;
  { PhaseDesc d = mk(K_LN_FWD, (N + 15) / 16); d.p[0] = lnf; d.p[1] = meanf; d.p[2] = rstdf; d.p[3] = xs[L]; d.p[4] = Wbf + lay.gf; p_lnf = Bd.add("lnf", d, {cur}); }
  { PhaseDesc d = mkgemm(gemm::NT, gemm::EPI_BF16, lnf, D, wte_b, D, logits, V, nullptr, 0, N, V, D, 0); p_logits = Bd.add("logits", d, {p_lnf}); }
  { PhaseDesc d = mk(K_CE, N / ew::CE_ROWS); d.p[0] = logits; d.p[1] = tok_tgt; d.p[2] = lossv; p_ce = Bd.add("cross-entropy", d, {p_logits, p_data}); }
  { PhaseDesc d = mk(K_CE_MEAN, 1); d.p[0] = meanloss; d.p[1] = lossv; p_cem = Bd.add("loss mean", d, {p_ce}); }
  std::vector<int> grad_phases;
  int p_headdx, p_headdw, p_lnfb, p_dgf;
  { PhaseDesc d = mkgemm(gemm::NN, gemm::EPI_BF16, logits, V, wte_b, D, dxf, D, nullptr, 0, N, D, V, 0); p_headdx = Bd.add("head dX", d, {p_ce}); }
  { PhaseDesc d = mkgemm(gemm::TN, gemm::EPI_F32, logits, V, lnf, D, G32 + lay.wte, D, nullptr, 0, V, D, N, 1); p_headdw = Bd.add("head dWte", d, {p_ce}); }
  { PhaseDesc d = mk(K_LN_BWD, (N + 15) / 16); d.p[0] = dxs[L]; d.p[1] = dgpf; d.p[2] = dxf; d.p[3] = xs[L]; d.p[4] = meanf; d.p[5] = rstdf; d.p[6] = Wbf + lay.gf; d.p[7] = nullptr; p_lnfb = Bd.add("lnf bwd", d, {p_headdx}); }
  { PhaseDesc d = mk(K_LN_DG, (D + 255) / 256); d.p[0] = G32 + lay.gf; d.p[1] = dgpf; p_dgf = Bd.add("lnf dgamma", d, {p_lnfb}); }
  grad_phases.push_back(p_dgf);
  int g = p_lnfb;                                            // phase producing dxs[l+1]
  std::vector<int> p_group(L, -1);
  const bool adam_early = getenv("OFA_ADAM_EARLY") != nullptr;   // experiment (neutral: 13.29 vs 13.23 ms/step): create the per-layer AdamW phases right after each layer's backward
  // round 2: the per-layer AdamW phases are created RIGHT AFTER each layer's backward (not after the embedding backward) so that, with tickets claimed in phase order, their memory-bound tasks overlap the
  // compute-bound backward of the earlier layers instead of forming a ~1.2 ms serial tail (numerics unchanged: same tasks, same dependencies)
  auto group = [&](const char* nm, size_t a0, size_t a1, std::vector<int> deps) {
    std::vector<int2> nd; for (auto r : nodecay) if ((size_t)r.x >= a0 && (size_t)r.y <= a1) nd.push_back(make_int2(r.x - (int)a0, r.y - (int)a0));
    int2* dnd = nullptr; if (!nd.empty()) { dnd = dalloc<int2>(nd.size()); CK(cudaMemcpy(dnd, nd.data(), nd.size() * sizeof(int2), cudaMemcpyHostToDevice)); }
    const size_t n = a1 - a0;
    PhaseDesc d = mk(K_ADAMW, (int)((n + ew::OPT_CHUNK - 1) / ew::OPT_CHUNK)); d.p[0] = W32 + a0; d.p[1] = G32 + a0; d.p[2] = M32 + a0; d.p[3] = V32 + a0; d.p[4] = Wbf + a0; d.p[5] = dnd; d.i0 = (int)nd.size(); d.n = n;
    return Bd.add(nm, d, deps);
  };
  std::vector<int> layer_grads[L]; int layer_g[L];
  for (int l = L - 1; l >= 0; l--) {
    const size_t gp0 = grad_phases.size();
    const bf16 *g1b = Wbf + lay.g1[l], *g2b = Wbf + lay.g2[l], *wq = Wbf + lay.wqkv[l], *wo = Wbf + lay.wo[l], *wu = Wbf + lay.wup[l], *wd = Wbf + lay.wdn[l];
    char nm[64];
    PhaseDesc d = mkgemm(gemm::NT, gemm::EPI_DGELU, dxs[l + 1], D, wd, D, dpre[l], FF, upre[l], FF, N, FF, D, 0); snprintf(nm, 64, "L%d dpre", l); int p_dpre = Bd.add(nm, d, {g});
    d = mkgemm(gemm::TN, gemm::EPI_F32, uact[l], FF, dxs[l + 1], D, G32 + lay.wdn[l], D, nullptr, 0, FF, D, N, 1); snprintf(nm, 64, "L%d dWdown", l); grad_phases.push_back(Bd.add(nm, d, {g}));
    d = mkgemm(gemm::NT, gemm::EPI_BF16, dpre[l], FF, wu, FF, dln2[l], D, nullptr, 0, N, D, FF, 0); snprintf(nm, 64, "L%d dln2", l); int p_dln2 = Bd.add(nm, d, {p_dpre});
    d = mkgemm(gemm::TN, gemm::EPI_F32, ln2[l], D, dpre[l], FF, G32 + lay.wup[l], FF, nullptr, 0, D, FF, N, 1); snprintf(nm, 64, "L%d dWup", l); grad_phases.push_back(Bd.add(nm, d, {p_dpre}));
    d = mk(K_LN_BWD, (N + 15) / 16); d.p[0] = dh[l]; d.p[1] = dgp2[l]; d.p[2] = dln2[l]; d.p[3] = hb_[l]; d.p[4] = mean2[l]; d.p[5] = rstd2[l]; d.p[6] = g2b; d.p[7] = dxs[l + 1];
    snprintf(nm, 64, "L%d ln2 bwd", l); int p_ln2b = Bd.add(nm, d, {p_dln2});
    d = mk(K_LN_DG, (D + 255) / 256); d.p[0] = G32 + lay.g2[l]; d.p[1] = dgp2[l]; snprintf(nm, 64, "L%d dgamma2", l); grad_phases.push_back(Bd.add(nm, d, {p_ln2b}));
    d = mkgemm(gemm::NT, gemm::EPI_BF16, dh[l], D, wo, D, dob[l], D, nullptr, 0, N, D, D, 0); snprintf(nm, 64, "L%d dO", l); int p_do = Bd.add(nm, d, {p_ln2b});
    d = mkgemm(gemm::TN, gemm::EPI_F32, ob[l], D, dh[l], D, G32 + lay.wo[l], D, nullptr, 0, D, D, N, 1); snprintf(nm, 64, "L%d dWo", l); grad_phases.push_back(Bd.add(nm, d, {p_ln2b}));
    d = mk(K_ATT_PREP, B * H * (T / 64)); d.p[0] = ob[l]; d.p[1] = dob[l]; d.p[2] = Dr[l]; snprintf(nm, 64, "L%d attn prep", l); int p_prep = Bd.add(nm, d, {p_do});
    d = mk(K_ATT_BWD, 2 * B * H * (T / 64)); d.p[0] = qkv[l]; d.p[1] = dob[l]; d.p[2] = lse[l]; d.p[3] = Dr[l]; d.p[4] = dqkv[l]; snprintf(nm, 64, "L%d attn bwd", l); int p_abwd = Bd.add(nm, d, {p_prep});
    d = mkgemm(gemm::NT, gemm::EPI_BF16, dqkv[l], 3 * D, wq, 3 * D, dln1[l], D, nullptr, 0, N, D, 3 * D, 0); snprintf(nm, 64, "L%d dln1", l); int p_dln1 = Bd.add(nm, d, {p_abwd});
    d = mkgemm(gemm::TN, gemm::EPI_F32, ln1[l], D, dqkv[l], 3 * D, G32 + lay.wqkv[l], 3 * D, nullptr, 0, D, 3 * D, N, 1); snprintf(nm, 64, "L%d dWqkv", l); grad_phases.push_back(Bd.add(nm, d, {p_abwd}));
    d = mk(K_LN_BWD, (N + 15) / 16); d.p[0] = dxs[l]; d.p[1] = dgp1[l]; d.p[2] = dln1[l]; d.p[3] = xs[l]; d.p[4] = mean1[l]; d.p[5] = rstd1[l]; d.p[6] = g1b; d.p[7] = dh[l];
    snprintf(nm, 64, "L%d ln1 bwd", l); g = Bd.add(nm, d, {p_dln1});
    d = mk(K_LN_DG, (D + 255) / 256); d.p[0] = G32 + lay.g1[l]; d.p[1] = dgp1[l]; snprintf(nm, 64, "L%d dgamma1", l); grad_phases.push_back(Bd.add(nm, d, {g}));
    layer_grads[l].assign(grad_phases.begin() + gp0, grad_phases.end()); layer_g[l] = g;
    if (fuse_opt && adam_early) {
      std::vector<int> deps = layer_grads[l]; deps.push_back(layer_g[l]); if (l == L - 1) deps.push_back(p_dgf);
      char nm2[64]; snprintf(nm2, 64, "adamw L%d", l);
      p_group[l] = group(nm2, lay.g1[l], l == L - 1 ? lay.gf + D : lay.g1[l + 1], deps);
    }
  }
  int p_ewte, p_ewpe;
  { PhaseDesc d = mk(K_EMB_WTE, V / ew::EMB_VCHUNK); d.p[0] = G32 + lay.wte; d.p[1] = dxs[0]; d.p[2] = tok_in;
    if (fuse_wte) { d.i0 = 1; d.p[3] = W32 + lay.wte; d.p[4] = M32 + lay.wte; d.p[5] = V32 + lay.wte; d.p[6] = Wbf + lay.wte; }
    p_ewte = Bd.add("embed bwd wte (+adamw)", d, {g, p_headdw, p_cem}); grad_phases.push_back(p_ewte); }
  { PhaseDesc d = mk(K_EMB_WPE, T / 4); d.p[0] = G32 + lay.wpe; d.p[1] = dxs[0]; p_ewpe = Bd.add("embed bwd wpe", d, {g}); grad_phases.push_back(p_ewpe); }
  int last_ph = p_ewpe;
  int p_wpe_opt = -1, p_wte_opt = -1;       // AdamW phases per layer (layer L-1 also covers gf) and for wpe
  if (fuse_opt) {
    p_wpe_opt = group("adamw wpe", lay.wpe, lay.wpe + (size_t)T * D, {p_ewpe});
    if (!fuse_wte) p_wte_opt = group("adamw wte", lay.wte, lay.wte + (size_t)V * D, {p_ewte});
    if (!adam_early) for (int l = L - 1; l >= 0; l--) {
      std::vector<int> deps = layer_grads[l]; deps.push_back(layer_g[l]); if (l == L - 1) deps.push_back(p_dgf);
      char nm[64]; snprintf(nm, 64, "adamw L%d", l);
      p_group[l] = group(nm, lay.g1[l], l == L - 1 ? lay.gf + D : lay.g1[l + 1], deps);
    }
    PhaseDesc d = mk(K_DATA, 0); d.ntasks = 1; d.kind = K_CE_MEAN; d.p[0] = meanloss; d.p[1] = lossv;   // 1-task barrier: ends the step once every update is done
    std::vector<int> deps = {p_ewte, p_wpe_opt, p_cem}; if (!fuse_wte) deps.push_back(p_wte_opt); for (int l = 0; l < L; l++) deps.push_back(p_group[l]);
    last_ph = Bd.add("step barrier", d, deps);
  } else if (!nopt) {                                        // opt-mode 0: one AdamW phase over everything after the whole backward
    PhaseDesc d = mk(K_ADAMW, (int)((NP + ew::OPT_CHUNK - 1) / ew::OPT_CHUNK)); d.p[0] = W32; d.p[1] = G32; d.p[2] = M32; d.p[3] = V32; d.p[4] = Wbf; d.p[5] = d_nodecay; d.i0 = (int)nodecay.size(); d.n = NP;
    std::vector<int> deps = grad_phases; deps.push_back(p_cem);
    last_ph = Bd.add("adamw", d, deps);
  } else {                                                   // gradient check mode: a barrier phase so every gradient is complete
    PhaseDesc d = mk(K_DATA, 0); d.ntasks = 1; d.kind = K_CE_MEAN; d.p[0] = meanloss; d.p[1] = lossv;   // harmless 1-task phase (recomputes the loss mean)
    std::vector<int> deps = grad_phases; deps.push_back(p_cem); last_ph = Bd.add("barrier", d, deps);
  }
  if (fuse_opt) {                                            // step s+1 waits only for what it actually reads/overwrites from step s
    Bd.deps[Bd.dep_off[p_data]] = make_int2(fuse_wte ? p_ewte : p_wte_opt, -1);
    Bd.deps[Bd.dep_off[p_embed] + 1] = make_int2(fuse_wte ? p_ewte : p_wte_opt, -1); Bd.deps[Bd.dep_off[p_embed] + 2] = make_int2(p_wpe_opt, -1);
    for (int l = 0; l < L; l++) Bd.deps[Bd.dep_off[p_ln1f[l]] + 1] = make_int2(p_group[l], -1);
  } else Bd.deps[Bd.dep_off[p_data]] = make_int2(last_ph, -1);   // step s+1 starts after step s finished
  const int nph = (int)Bd.ph.size();
  std::vector<int> prefix(nph + 1, 0); for (int i = 0; i < nph; i++) prefix[i + 1] = prefix[i] + Bd.ntasks[i];
  const int tps = prefix[nph];
  printf("training graph: %d phases, %d tasks per step\n", nph, tps);

  auto upload_job = [&](Builder& b, std::vector<int>& pre, int limit, int last, Job& J) {
    int n = (int)b.ph.size();
    PhaseDesc* dph; CK(cudaMalloc(&dph, n * sizeof(PhaseDesc))); CK(cudaMemcpy(dph, b.ph.data(), n * sizeof(PhaseDesc), cudaMemcpyHostToDevice));
    int *dpre_, *dnt, *doff, *dcnt, *ddone; int2* ddeps; unsigned* dticket;
    CK(cudaMalloc(&dpre_, (n + 1) * 4)); CK(cudaMemcpy(dpre_, pre.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dnt, n * 4)); CK(cudaMemcpy(dnt, b.ntasks.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&doff, n * 4)); CK(cudaMemcpy(doff, b.dep_off.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dcnt, n * 4)); CK(cudaMemcpy(dcnt, b.dep_cnt.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ddeps, std::max<size_t>(1, b.deps.size()) * sizeof(int2))); CK(cudaMemcpy(ddeps, b.deps.data(), b.deps.size() * sizeof(int2), cudaMemcpyHostToDevice));
    ddone = dalloc<int>(n); dticket = dalloc<unsigned>(1);
    J.nph = n; J.tps = pre[n]; J.limit = limit; J.last_ph = last; J.ph = dph; J.prefix = dpre_; J.ntasks = dnt; J.dep_off = doff; J.dep_cnt = dcnt; J.deps = ddeps; J.done = ddone; J.ticket = dticket;
  };
  Ctx C; memset(&C, 0, sizeof C);
  upload_job(Bd, prefix, steps, last_ph, C.train);

  // ---------------- decode job
  decode::State<DC>* dstate = nullptr; Builder Dd; std::vector<int> dprefix; int* h_out_tokens_pinned = nullptr; const int out_cap = 1 << 20;
  std::vector<int> forced;
  decode::State<DC>* dstate2 = nullptr; int* tokens2 = nullptr;
  auto setup_decode = [&](decode::Params<DC>& P, Job& J, decode::State<DC>*& state, int*& tokens, int instance, double rate) {
    bf16* frozen = Wbf;
    if (duration_s > 0) { frozen = dalloc<bf16>(NP); CK(cudaMemcpy(frozen, Wbf, NP * 2, cudaMemcpyDeviceToDevice)); }
    decode::Weights& w = P.w; w.wte = frozen + lay.wte; w.wpe = frozen + lay.wpe; w.gf = frozen + lay.gf;
    for (int l = 0; l < L; l++) { w.wqkv[l] = frozen + lay.wqkv[l]; w.wo[l] = frozen + lay.wo[l]; w.wup[l] = frozen + lay.wup[l]; w.wdn[l] = frozen + lay.wdn[l]; w.g1[l] = frozen + lay.g1[l]; w.g2[l] = frozen + lay.g2[l]; }
    state = decode::alloc_state<DC>(out_cap);
    int* dout_dev; CK(cudaHostAlloc((void**)&tokens, out_cap * 4, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dout_dev, tokens, 0));
    memset(tokens, 0xFF, out_cap * 4); CK(cudaMemcpy(&state->out_tokens, &dout_dev, sizeof(dout_dev), cudaMemcpyHostToDevice));
    P.st = state; P.temperature = temp; P.seed = seed + instance; P.valid_vocab = VALID; P.auto_reset = 1;
    forced.clear(); Dd = Builder{};
    if (!prompt_path.empty()) { FILE* f = fopen(prompt_path.c_str(), "r"); int x; while (f && fscanf(f, "%d", &x) == 1) forced.push_back(x); if (f) fclose(f); }
    if (forced.empty()) forced.push_back(100257);            // <|endoftext|>
    int* dforced = dalloc<int>(forced.size()); CK(cudaMemcpy(dforced, forced.data(), forced.size() * 4, cudaMemcpyHostToDevice));
    P.forced = dforced; P.nforced = (int)forced.size();
    decode::begin_context<DC>(P, forced[0], 0);
    for (int p = 0; p < DC::PHASES; p++) {
      PhaseDesc d = mk(K_DECODE, decode::phase_tasks<DC>(p)); d.i0 = p;
      Dd.add("decode", d, {p - 1}, p == 0 ? std::vector<int2>{make_int2(DC::PHASES - 1, -1)} : std::vector<int2>{});
    }
    dprefix.assign(DC::PHASES + 1, 0); for (int i = 0; i < DC::PHASES; i++) dprefix[i + 1] = dprefix[i] + Dd.ntasks[i];
    const u64 period = rate > 0 ? (u64)(1e9 / rate) : 0;
    const u64 limit = duration_s > 0 && period ? 1 + ((u64)(duration_s * 1e9) - 1) / period : 1 << 22;
    if (duration_s > 0 && (!period || limit > (u64)out_cap)) { fprintf(stderr, "serving demand exceeds token capacity\n"); exit(2); }
    upload_job(Dd, dprefix, (int)limit, DC::PHASES - 1, J);
    printf("serve%d: independent state, seed %llu, %d phases, %llu token requests\n", instance + 1, P.seed, DC::PHASES, limit);
  };
  if (use_decode) setup_decode(C.dp, C.dec, dstate, h_out_tokens_pinned, 0, serve_rate);
  if (infer2_sms) setup_decode(C.dp2, C.dec2, dstate2, tokens2, 1, serve2_rate);

  C.dm = dm; C.live = dm.live;
  if (port > 0) {
    dash::fb_host = fb_host; dash::live = live_host; dash::nsm = nsm; dash::index_path = index_path; dash::prompt = prompt_text; dash::tokens_per_step = N;
    dash::sm_infer = infer_sms; dash::sm_doom = doom_on ? doom_sms : 0; dash::sm_train = train_sms_arg;
    if (use_decode) { dash::tok_host = h_out_tokens_pinned; dash::tok_cap = out_cap; dash::skip_tokens = (int)forced.size() - 1; }
    if (!vocab_path.empty() && !dash::load_vocab(vocab_path)) fprintf(stderr, "warning: cannot read vocab %s (text panel will be empty)\n", vocab_path.c_str());
    dash::note = "one kernel launch: training + inference + Doom"; dash::start(port); printf("dashboard on http://localhost:%d\n", port);
  }
  C.data = d_data; C.ntok = ntok; C.starts = d_starts; C.nstarts = nstarts; C.seed = seed; C.tok_in = tok_in; C.tok_tgt = tok_tgt;
  C.lr_max = 6e-4f; C.lr_min = 6e-5f; C.beta1 = 0.9f; C.beta2 = 0.95f; C.eps = 1e-8f; C.wd = 0.1f; C.warmup = 100; C.sched_steps = sched_steps;
  C.phase_busy = dalloc<u64>(nph); C.phase_first = dalloc<u64>(nph); C.phase_last = dalloc<u64>(nph); C.prof_step = prof_step < 0 ? steps / 2 : prof_step;
  { std::vector<u64> big(nph, ~0ull); CK(cudaMemcpy(C.phase_first, big.data(), nph * 8, cudaMemcpyHostToDevice)); }
  C.step_done = dalloc<u64>(steps + 1); C.loss_hist = dalloc<float>(steps + 1); C.blk_busy = dalloc<u64>(nblocks * (infer2_sms ? 3 : 2)); C.finished = dalloc<int>(1); C.t_end = dalloc<u64>(1);
  C.infer_sms = infer_sms; C.doom_sms = doom_sms; C.train_sms = train_sms_arg; C.use_decode = use_decode; C.smid_rank = d_rank;
  C.dm.finished = C.finished;
  C.serve_period_ns = serve_rate > 0 ? (u64)(1e9 / serve_rate) : 0; C.lend = lend; C.tok_cap = 1 << 20;
  C.tok_done = dalloc<u64>(C.tok_cap); C.serve_base = dalloc<u64>(1); C.lent_tasks = dalloc<unsigned>(1);
  C.infer2_sms = infer2_sms; C.run_ns = (u64)(duration_s * 1e9); C.run_start = dalloc<u64>(1);
  C.serve2_period_ns = serve2_rate > 0 ? (u64)(1e9 / serve2_rate) : 0;
  C.tok_done2 = dalloc<u64>(C.tok_cap); C.serve_base2 = dalloc<u64>(1);
  if ((u64)C.train.tps * steps > 0xffffffffull) { fprintf(stderr, "training ticket overflow: reduce --steps\n"); return 2; }
#if !OFA_NODUS_DYN
  if (serve_rate > 0 || lend) { fprintf(stderr, "--serve-rate/--lend need a -DOFA_NODUS_DYN=1 build\n"); return 2; }
#endif
#if OFA_GEMM_BACKEND != 0 && OFA_GEMM_BACKEND != 3
  if (C.lend) { printf("warning: --lend needs the mma.sync or sm90 GEMM backend (OFA_GEMM_BACKEND=0 or 3); lending disabled\n"); C.lend = 0; }
#endif
  if (serve_rate > 0 || lend) printf("serving demand: %s, lending %s\n", serve_rate > 0 ? "open-loop paced" : "closed loop", C.lend ? "on" : "off");
  CK(cudaDeviceSynchronize());

  auto wall_now = []() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
  double clock_error_s = 0;
  if (duration_s > 0) {
    u64 tick = 0; double calibrated = 0; clock_error_s = 1e30;
    for (int sample = 0; sample < 5; sample++) {
      const double before = wall_now();
      clock_probe<<<1, 1>>>(C.run_start);
      u64 probe; CK(cudaMemcpy(&probe, C.run_start, 8, cudaMemcpyDeviceToHost));
      const double after = wall_now(), error = (after - before) / 2;
      if (error < clock_error_s) { tick = probe; calibrated = (before + after) / 2; clock_error_s = error; }
    }
    if (!ready_file.empty()) { std::ofstream f(ready_file); f << getpid() << "\n"; if (!f) return 2; }
    double target = wall_now() + 0.1;
    if (!go_file.empty()) {
      const double timeout = wall_now() + 600;
      for (;;) {
        std::ifstream f(go_file);
        if (f >> target) break;
        if (wall_now() > timeout) { fprintf(stderr, "start barrier timeout\n"); return 2; }
        usleep(10000);
      }
    }
    const u64 start = tick + (u64)(std::max(0.0, target - calibrated) * 1e9);
    CK(cudaMemcpy(C.run_start, &start, 8, cudaMemcpyHostToDevice));
    while (wall_now() < target) usleep(1000);
  }

  // ---------------- run: ONE launch
  printf("launching ONE persistent kernel (%d blocks)...\n", nblocks);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const double tw0 = now_s();
  dash::t0 = dash::now_s(); dash::phase = 1;
  CK(cudaEventRecord(e0)); ofa_kernel<<<nblocks, 256, DYN_SMEM>>>(C); CK(cudaEventRecord(e1));
  std::vector<int> seen_tokens; size_t shown = 0; std::string textdump;
  while (cudaEventQuery(e1) == cudaErrorNotReady) {
    usleep(200000);
    if (use_decode) { unsigned long long t = 0; (void)t; }
  }
  CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  dash::phase = 3;
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  const double wall = now_s() - tw0;

  if (duration_s > 0) {
    FILE* f = fopen(metrics_out.c_str(), "w");
    if (!f) { perror("metrics-out"); return 2; }
    u64 start; CK(cudaMemcpy(&start, C.run_start, 8, cudaMemcpyDeviceToHost));
    fprintf(f, "kind,workload,index,arrival_s,done_s,latency_s,token,loss\n");
    fprintf(f, "clock,all,0,0,%.9f,%.9f,0,0\n", duration_s, clock_error_s);
    auto emit = [&](int instance, const Job& J, u64* done, u64 period, int* tokens) {
      std::vector<u64> td(J.limit); CK(cudaMemcpy(td.data(), done, J.limit * 8, cudaMemcpyDeviceToHost));
      for (int s = 0; s < J.limit; s++) {
        const double arrival = (double)((u64)s * period) * 1e-9;
        const bool complete = td[s] >= start && td[s] <= start + C.run_ns;
        const double end = complete ? (double)(td[s] - start) * 1e-9 : -1;
        fprintf(f, "serve,serve%d,%d,%.9f,%.9f,%.9f,%d,0\n", instance, s, arrival, end, complete ? end - arrival : -1, complete ? tokens[s] : -1);
      }
    };
    if (use_decode) emit(1, C.dec, C.tok_done, C.serve_period_ns, h_out_tokens_pinned);
    if (infer2_sms) emit(2, C.dec2, C.tok_done2, C.serve2_period_ns, tokens2);
    std::vector<float> loss(steps); std::vector<u64> sd(steps);
    CK(cudaMemcpy(loss.data(), C.loss_hist, steps * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sd.data(), C.step_done, steps * 8, cudaMemcpyDeviceToHost));
    for (int s = 0; s < steps; s++) if (sd[s] >= start && sd[s] <= start + C.run_ns)
      fprintf(f, "train,train,%d,0,%.9f,0,0,%.9g\n", s, (double)(sd[s] - start) * 1e-9, loss[s]);
    if (train_sms_arg && sd.back()) fprintf(f, "error,train,%d,0,0,0,0,0\n", steps);
    fclose(f);
    printf("MEASURED fixed window %.3f s, event %.3f s, clock uncertainty %.6f s, metrics %s\n", duration_s, ms / 1e3, clock_error_s, metrics_out.c_str());
    return 0;
  }

  // ---------------- results
  std::vector<float> loss(steps); CK(cudaMemcpy(loss.data(), C.loss_hist, steps * 4, cudaMemcpyDeviceToHost));
  std::vector<u64> sd(steps); CK(cudaMemcpy(sd.data(), C.step_done, steps * 8, cudaMemcpyDeviceToHost));
  u64 tend; CK(cudaMemcpy(&tend, C.t_end, 8, cudaMemcpyDeviceToHost));
  printf("\nkernel time %.3f s (event) / %.3f s (wall incl. launch)\n", ms / 1e3, wall);
  const int w0 = std::min(steps - 1, 3); const double train_s = (double)(sd[steps - 1] - sd[w0]) * 1e-9; const int nstep_t = steps - 1 - w0;
  std::vector<double> dts; for (int s = w0 + 1; s < steps; s++) dts.push_back((sd[s] - sd[s - 1]) * 1e-6); std::sort(dts.begin(), dts.end());
  if (nstep_t > 0) printf("steady state: %.2f ms/step (median %.2f) -> %.0f tok/s;  4,000 steps = %.1f s\n", train_s * 1e3 / nstep_t, dts[dts.size() / 2], (double)N * nstep_t / train_s, 4000 * train_s / nstep_t);
  printf("loss: step0 %.4f", loss[0]); for (int s : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 3999}) if (s < steps) printf("  s%d %.4f", s, loss[s]); printf("\n");
  if (!loss_out.empty()) { FILE* f = fopen(loss_out.c_str(), "w"); for (int s = 0; s < steps; s++) fprintf(f, "%d %.6f\n", s, loss[s]); fclose(f); }
  { // per-phase breakdown at the profiled step + busy-time shares
    std::vector<u64> pb(nph), pf(nph), pl(nph); CK(cudaMemcpy(pb.data(), C.phase_busy, nph * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(pf.data(), C.phase_first, nph * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(pl.data(), C.phase_last, nph * 8, cudaMemcpyDeviceToHost));
    double tot = 0; for (int i = 0; i < nph; i++) tot += pb[i];
    std::vector<std::pair<std::string, double>> grp; auto addg = [&](const std::string& k, double v) { for (auto& g2 : grp) if (g2.first == k) { g2.second += v; return; } grp.push_back({k, v}); };
    for (int i = 0; i < nph; i++) { std::string nme = Bd.names[i]; size_t sp = nme.find(' '); std::string key = (nme[0] == 'L' && sp != std::string::npos && isdigit(nme[1])) ? nme.substr(sp + 1) : nme; addg(key, (double)pb[i]); }
    printf("\nbusy time by phase kind (all steps, block-seconds; %% of train busy):\n"); for (auto& g2 : grp) printf("  %-18s %8.3f s  %5.1f%%\n", g2.first.c_str(), g2.second * 1e-9, 100 * g2.second / tot);
    double wall_s = (double)(sd[steps - 1] - sd[steps - 2]) * 1e-9; (void)wall_s;
    printf("\nphase wall-clock at step %d (first task start -> last task end, ms from step start):\n", C.prof_step ? C.prof_step : 0);
    u64 t0s = ~0ull; for (int i = 0; i < nph; i++) if (pf[i] != ~0ull) t0s = std::min(t0s, pf[i]);
    for (int i = 0; i < nph && i < 400; i++) if (pf[i] != ~0ull && (Bd.names[i].rfind("L1 ", 0) == 0 || Bd.names[i].rfind("L", 0) != 0)) printf("  %-16s %8.2f -> %8.2f ms  (%6.2f ms)  tasks=%d\n", Bd.names[i].c_str(), (pf[i] - t0s) * 1e-6, (pl[i] - t0s) * 1e-6, (pl[i] - pf[i]) * 1e-6, Bd.ntasks[i]);
  }
  { std::vector<u64> bb(nblocks * 2); CK(cudaMemcpy(bb.data(), C.blk_busy, nblocks * 16, cudaMemcpyDeviceToHost)); double tb = 0, ib = 0; for (int b = 0; b < nblocks; b++) { tb += bb[2 * b]; ib += bb[2 * b + 1]; }
    const double span = ms * 1e-3; printf("\nSM utilisation (task time / wall, %d blocks per SM): training %.1f SM-equivalents, serving %.1f SM-equivalents (of %d SMs)\n", std::min(occ, BPS), tb * 1e-9 / span / std::min(occ, BPS), ib * 1e-9 / span / std::min(occ, BPS), nsm); }
  if (use_decode) {
    int dpos = 0, dstep = 0; CK(cudaMemcpy(&dstep, &dstate->step, 4, cudaMemcpyDeviceToHost)); (void)dpos;
    printf("decode: %d tokens generated during %.2f s of training -> %.0f tok/s\n", dstep, (tend ? (double)(tend - sd[0]) * 1e-9 : ms * 1e-3), dstep / ((double)(sd[steps - 1] - sd[0]) * 1e-9 + 1e-9));
    FILE* f = text_out.empty() ? nullptr : fopen(text_out.c_str(), "w"); int cnt = std::min(dstep, out_cap);
    if (f) { for (int i = 0; i < cnt; i++) fprintf(f, "%d ", h_out_tokens_pinned[i]); fclose(f); }
    printf("first sampled token ids:"); for (int i = 0; i < std::min(cnt, 40); i++) printf(" %d", h_out_tokens_pinned[i]); printf("\n");
    { const int nt = std::min(dstep, C.tok_cap); std::vector<u64> td(std::max(nt, 1)); if (nt) CK(cudaMemcpy(td.data(), C.tok_done, nt * 8, cudaMemcpyDeviceToHost));
      u64 base = 0; unsigned lent = 0; CK(cudaMemcpy(&base, C.serve_base, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lent, C.lent_tasks, 4, cudaMemcpyDeviceToHost));
      const u64 w_lo = sd[w0], w_hi = sd[steps - 1]; std::vector<double> lat; int served = 0;
      for (int s = 1; s < nt; s++) {
        if (!td[s] || td[s] < w_lo || td[s] > w_hi) continue;
        served++;
        const u64 rel = C.serve_period_ns ? base + (u64)s * C.serve_period_ns : td[s - 1];
        lat.push_back(td[s] > rel ? (td[s] - rel) * 1e-6 : 0.0);
      }
      std::sort(lat.begin(), lat.end()); int ok = 0; for (double v : lat) ok += v <= 50.0;
      const double win = (double)(w_hi - w_lo) * 1e-9;
      const double p50 = lat.empty() ? 0 : lat[lat.size() / 2], p99 = lat.empty() ? 0 : lat[std::min(lat.size() - 1, (size_t)(lat.size() * 0.99))];
      printf("serve: demand %.1f tok/s (%s), %d tokens in the training window %.2f s -> %.1f tok/s, latency p50 %.3f ms p99 %.3f ms, slo<=50ms %.4f, lent train tasks %u\n",
             serve_rate, C.serve_period_ns ? "open-loop" : "closed-loop", served, win, served / (win + 1e-12), p50, p99, lat.empty() ? 0.0 : (double)ok / lat.size(), lent); }
  }
  if (dm.on) {
    const unsigned fr = live_host->frames; std::vector<u64> fd(std::min<unsigned>(fr, DOOM_MAX_FRAMES)); CK(cudaMemcpy(fd.data(), dm.frame_done, fd.size() * 8, cudaMemcpyDeviceToHost));
    double fps_all = fd.size() > 2 ? (double)(fd.size() - 1) / ((double)(fd.back() - fd.front()) * 1e-9) : 0; unsigned in_train = 0; for (u64 t : fd) if (t >= sd[w0] && t <= sd[steps - 1]) in_train++;
    printf("doom: %u frames in the run, %.1f fps over the run (frame 0 = boot excluded); %.1f fps while training steps %d..%d\n", fr, fps_all, in_train / ((double)(sd[steps - 1] - sd[w0]) * 1e-9), w0, steps - 1);
  }
  if (!dump_grads.empty()) { std::vector<float> h(NP); CK(cudaMemcpy(h.data(), G32, NP * 4, cudaMemcpyDeviceToHost)); FILE* f = fopen(dump_grads.c_str(), "wb"); fwrite(h.data(), 4, NP, f); fclose(f); printf("wrote grads %s\n", dump_grads.c_str()); }
  if (!dump_w.empty()) { std::vector<float> h(NP); CK(cudaMemcpy(h.data(), W32, NP * 4, cudaMemcpyDeviceToHost)); FILE* f = fopen(dump_w.c_str(), "wb"); fwrite(h.data(), 4, NP, f); fclose(f); printf("wrote weights %s\n", dump_w.c_str()); }
  if (port > 0) { printf("done; the dashboard keeps serving the final screen. Ctrl-C to quit.\n"); for (;;) sleep(60); }
  return 0;
}
