// mw.cu : multi-workload persistent megakernel (H100, sm_90a). ONE launch, SM-partitioned work queues:
//   train queue : N-job shared-trunk LoRA tail at Qwen3-4B q/v shapes (top-1 adapted layer, rank 8): shared frozen base q/v projection
//                 (mma.sync tiles from llm/gemm/gemm.cuh, computed once per step for all jobs), per-job LoRA down/up, MSE loss, backward
//                 to A/B, AdamW (fp32), every job every step.
//   decode queue: batch-b (1..8) decode for ONE LoRA adapter at the same layer shapes: q/k/v GEMV (+LoRA on q/v), GQA attention over a KV
//                 cache (32 q / 8 kv heads, d=128), o-proj + residual, tied 151936-row lm_head, greedy argmax, next-token embedding.
//   --mode train | dec | co.  co: blocks on SM ranks < --infer-sms run decode, the rest train. Weights are hash-initialised (Qwen3-4B shapes,
//   not the checkpoint). Every task uses a fixed reduction order and no float atomics, so outputs are bitwise reproducible across modes,
//   SM splits and co-location: the gates compare co-run dumps with solo dumps byte for byte.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../../gemm/gemm.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

constexpr int D = 2560, NQ = 4096, NKV = 1024, NQV = NQ + NKV, NQKV = NQ + 2 * NKV, HD = 128, NKVH = 8, VOC = 151936, R = 8, R2 = 2 * R;
constexpr int PJ = 2 * R * D + NQV * R;            // per-job fp32 params: Aq[R][D] Av[R][D] Bq[NQ][R] Bv[NKV][R]
constexpr int MAXB = 8, MAXPOS = 4096, THREADS = 256, ROWS = 16;
constexpr int SMEM = gemm::smem_bytes();           // 64 KB
static_assert(4 * MAXPOS * 4 <= SMEM, "attention scores fit the GEMM smem");

enum Kind { T_BASEQ, T_BASEV, T_DOWN, T_UPG, T_DB, T_DU, T_DA, T_ADAM, T_LOSS, D_DOWN, D_QKV, D_ATT, D_O, D_HEAD, D_NEXT };
struct Ph { int kind, ntasks, tilesM; gemm::Args ga; };
struct Job {
  int nph, tps, limit, last_ph;
  const Ph* ph; const int *prefix, *ntasks, *dep_off, *dep_cnt; const int2* deps;
  int* done; unsigned* ticket;
};
struct Ctx {
  Job tr, dc; int mode, infer_sms, lend; unsigned* lent; const int* smid_rank; int* finished;
  // train
  int jobs, tk, nx; float scale, lr, b1, b2, eps, inv;
  const bf16 *X, *Wq, *Wv, *Tq, *Tv; bf16 *Yq, *Yv;
  float *P, *G, *M, *Vm, *U, *dO, *dU, *lpart, *loss; u64* step_done;
  // decode
  int b, ctx0; const bf16 *Wqkv, *Wo, *E; const float* adp;   // adp: one adapter, same layout as P
  float *h, *du, *q, *att, *lmax, *lsum_part, *lsum, *logits0; int *lidx, *toks; bf16 *kc, *vc; u64* tok_done;
};

__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ld_vol(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }

// block-wide sum with a fixed order (thread partials -> smem -> serial sum by thread 0)
__device__ float block_sum(float v, float* red) {
  red[threadIdx.x] = v; __syncthreads();
  float s = 0.f;
  if (threadIdx.x == 0) { for (int i = 0; i < THREADS; i++) s += red[i]; red[0] = s; }
  __syncthreads(); s = red[0]; __syncthreads(); return s;
}
// row-dot GEMV: out[bi][row] for 128 consecutive rows of W[rows][K] (bf16), input xin[b][K] (fp32). Warp w owns 16 rows; lane-strided
// 8-element chunks, fixed-order xor-shuffle reduce. Returns values to lane 0 via callback.
template <class F>
__device__ __forceinline__ void gemv128(const bf16* W, int K, int row0, int nrows, const float* xin, int b, F emit) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int rr = 0; rr < ROWS; rr++) {
    const int row = row0 + warp * ROWS + rr;
    if (row >= nrows) break;
    float acc[MAXB];
#pragma unroll
    for (int i = 0; i < MAXB; i++) acc[i] = 0.f;
    const bf16* w = W + (size_t)row * K;
    for (int k = lane * 8; k < K; k += 256) {
      const uint4 raw = *reinterpret_cast<const uint4*>(w + k);
      const bf16* wv = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
      for (int i = 0; i < MAXB; i++) if (i < b) {
        const float* x = xin + (size_t)i * K + k;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[i] = fmaf(bf(wv[e]), x[e], acc[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < MAXB; i++) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
    }
    if (lane == 0) for (int i = 0; i < b; i++) emit(i, row, acc[i]);
  }
}

// ------------------------------------------------------------------------------------------------ training tasks
__device__ void train_task(const Ctx& C, const Ph& d, int local, int step, char* smem) {
  float* red = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, tk = C.tk, rb_per_job = tk / ROWS;
  const bf16* X = C.X + (size_t)(step % C.nx) * tk * D;
  switch (d.kind) {
    case T_BASEQ: case T_BASEV: {
      gemm::Args a = d.ga; a.A = X;
      gemm::gemm_tile<gemm::NT, gemm::EPI_BF16>(a, local % d.tilesM, local / d.tilesM, smem); break;
    }
    case T_DOWN: {                                             // U[j][t][r16] = sum_d X[t][d] * A_{q|v}[r][d]
      const int j = local / rb_per_job, t = (local % rb_per_job) * ROWS + tid / R2, r = tid % R2;
      const float* A = C.P + (size_t)j * PJ + (size_t)r * D;   // Aq rows 0..7 then Av rows 0..7 are contiguous
      const bf16* x = X + (size_t)t * D; float acc = 0.f;
      for (int k = 0; k < D; k++) acc = fmaf(bf(x[k]), A[k], acc);
      C.U[((size_t)j * tk + t) * R2 + r] = acc; break;
    }
    case T_UPG: {                                              // O = Y + s*U.B^T; dO = (O - T) * inv; loss partial 0.5*(O-T)^2
      const int j = local / rb_per_job, rb = local % rb_per_job;
      const float* Bq = C.P + (size_t)j * PJ + 2 * R * D; const float* Bv = Bq + (size_t)NQ * R;
      float part = 0.f;
      for (int e = tid; e < ROWS * NQV; e += THREADS) {
        const int t = rb * ROWS + e / NQV, o = e % NQV;
        const float* u = C.U + ((size_t)j * tk + t) * R2;
        float y, tg, s = 0.f;
        if (o < NQ) { y = bf(C.Yq[(size_t)t * NQ + o]); tg = bf(C.Tq[(size_t)t * NQ + o]); for (int r = 0; r < R; r++) s = fmaf(u[r], Bq[(size_t)o * R + r], s); }
        else { const int ov = o - NQ; y = bf(C.Yv[(size_t)t * NKV + ov]); tg = bf(C.Tv[(size_t)t * NKV + ov]); for (int r = 0; r < R; r++) s = fmaf(u[R + r], Bv[(size_t)ov * R + r], s); }
        const float diff = fmaf(C.scale, s, y) - tg;
        C.dO[((size_t)j * tk + t) * NQV + o] = diff * C.inv;
        part = fmaf(0.5f * diff, diff, part);
      }
      const float tot = block_sum(part, red);
      if (tid == 0) C.lpart[(size_t)j * rb_per_job + rb] = tot; break;
    }
    case T_DB: {                                               // dB[o][r] = s * sum_t dO[t][o] * U[t][r]
      const int per = NQV / THREADS, j = local / per, o = (local % per) * THREADS + tid, off = o < NQ ? 0 : R;
      float acc[R];
      for (int r = 0; r < R; r++) acc[r] = 0.f;
      for (int t = 0; t < tk; t++) {
        const float g = C.dO[((size_t)j * tk + t) * NQV + o]; const float* u = C.U + ((size_t)j * tk + t) * R2 + off;
        for (int r = 0; r < R; r++) acc[r] = fmaf(g, u[r], acc[r]);
      }
      float* gB = C.G + (size_t)j * PJ + 2 * R * D + (size_t)o * R;   // Bq rows then Bv rows: contiguous in o
      for (int r = 0; r < R; r++) gB[r] = C.scale * acc[r]; break;
    }
    case T_DU: {                                               // dU[t][r16] = s * sum_o dO[t][o] * B[o][r]
      const int j = local / rb_per_job, t = (local % rb_per_job) * ROWS + tid / R2, r16 = tid % R2;
      const float* Bq = C.P + (size_t)j * PJ + 2 * R * D; const float* g = C.dO + ((size_t)j * tk + t) * NQV;
      float acc = 0.f;
      if (r16 < R) { for (int o = 0; o < NQ; o++) acc = fmaf(g[o], Bq[(size_t)o * R + r16], acc); }
      else { const float* Bv = Bq + (size_t)NQ * R; for (int o = 0; o < NKV; o++) acc = fmaf(g[NQ + o], Bv[(size_t)o * R + r16 - R], acc); }
      C.dU[((size_t)j * tk + t) * R2 + r16] = C.scale * acc; break;
    }
    case T_DA: {                                               // dA[r16][d] = sum_t dU[t][r16] * X[t][d]
      const int per = D / THREADS, j = local / per, k = (local % per) * THREADS + tid;
      float acc[R2];
      for (int r = 0; r < R2; r++) acc[r] = 0.f;
      for (int t = 0; t < tk; t++) {
        const float x = bf(X[(size_t)t * D + k]); const float* g = C.dU + ((size_t)j * tk + t) * R2;
        for (int r = 0; r < R2; r++) acc[r] = fmaf(g[r], x, acc[r]);
      }
      float* gA = C.G + (size_t)j * PJ;
      for (int r = 0; r < R2; r++) gA[(size_t)r * D + k] = acc[r]; break;
    }
    case T_ADAM: {                                             // AdamW (no decay) on job j's PJ fp32 params
      const int per = PJ / (THREADS * 4), j = local / per;
      const size_t base = (size_t)j * PJ + (size_t)(local % per) * THREADS * 4;
      const float bc1 = 1.f - powf(C.b1, (float)(step + 1)), bc2 = 1.f - powf(C.b2, (float)(step + 1));
      for (int e = 0; e < 4; e++) {
        const size_t i = base + (size_t)e * THREADS + tid; const float g = C.G[i];
        const float m = C.M[i] = C.b1 * C.M[i] + (1.f - C.b1) * g, v = C.Vm[i] = C.b2 * C.Vm[i] + (1.f - C.b2) * g * g;
        C.P[i] -= C.lr * (m / bc1) / (sqrtf(v / bc2) + C.eps);
      }
      break;
    }
    case T_LOSS: {
      if (tid < C.jobs) { float s = 0.f; for (int i = 0; i < rb_per_job; i++) s += C.lpart[(size_t)tid * rb_per_job + i]; C.loss[(size_t)step * C.jobs + tid] = s * C.inv; }
      break;
    }
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------ decode tasks (one token step)
__device__ void decode_task(const Ctx& C, const Ph& d, int local, int step, char* smem) {
  float* sm = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, b = C.b, pos = C.ctx0 + step;
  const float* Bq = C.adp + 2 * R * D; const float* Bv = Bq + (size_t)NQ * R;
  switch (d.kind) {
    case D_DOWN: {
      if (tid < b * R2) { const int i = tid / R2, r = tid % R2; const float* A = C.adp + (size_t)r * D; const float* x = C.h + (size_t)i * D; float acc = 0.f;
        for (int k = 0; k < D; k++) acc = fmaf(x[k], A[k], acc); C.du[i * R2 + r] = acc; }
      break;
    }
    case D_QKV: {
      gemv128(C.Wqkv, D, local * 128, NQKV, C.h, b, [&](int i, int row, float v) {
        const float* u = C.du + i * R2;
        if (row < NQ) { float s = 0.f; for (int r = 0; r < R; r++) s = fmaf(u[r], Bq[(size_t)row * R + r], s); C.q[(size_t)i * NQ + row] = fmaf(C.scale, s, v); }
        else if (row < NQ + NKV) C.kc[((size_t)i * MAXPOS + pos) * NKV + row - NQ] = __float2bfloat16(v);
        else { const int o = row - NQ - NKV; float s = 0.f; for (int r = 0; r < R; r++) s = fmaf(u[R + r], Bv[(size_t)o * R + r], s);
          C.vc[((size_t)i * MAXPOS + pos) * NKV + o] = __float2bfloat16(fmaf(C.scale, s, v)); }
      });
      break;
    }
    case D_ATT: {                                              // task = (sequence i, kv head g): its 4 q heads over positions 0..pos
      const int i = local / NKVH, g = local % NKVH, n = pos + 1, warp = tid >> 5, lane = tid & 31;
      const float* qh = C.q + (size_t)i * NQ + g * 4 * HD;
      for (int p = warp; p < n; p += 8) {
        const bf16* k = C.kc + ((size_t)i * MAXPOS + p) * NKV + g * HD;
        for (int hh = 0; hh < 4; hh++) {
          float s = 0.f;
          for (int e = lane; e < HD; e += 32) s = fmaf(qh[hh * HD + e], bf(k[e]), s);
          for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
          if (lane == 0) sm[hh * MAXPOS + p] = s * 0.08838834764831845f;
        }
      }
      __syncthreads();
      __shared__ float ssum[4];
      if (tid < 4) { float m = -INFINITY; for (int p = 0; p < n; p++) m = fmaxf(m, sm[tid * MAXPOS + p]); float s = 0.f;
        for (int p = 0; p < n; p++) { const float e = __expf(sm[tid * MAXPOS + p] - m); sm[tid * MAXPOS + p] = e; s += e; } ssum[tid] = s; }
      __syncthreads();
      { const int hh = tid / 64, e = (tid % 64) * 2; float a0 = 0.f, a1 = 0.f;
        for (int p = 0; p < n; p++) { const float w = sm[hh * MAXPOS + p]; const bf16* v = C.vc + ((size_t)i * MAXPOS + p) * NKV + g * HD + e; a0 = fmaf(w, bf(v[0]), a0); a1 = fmaf(w, bf(v[1]), a1); }
        float* o = C.att + (size_t)i * NQ + (g * 4 + hh) * HD + e; o[0] = a0 / ssum[hh]; o[1] = a1 / ssum[hh]; }
      break;
    }
    case D_O: gemv128(C.Wo, NQ, local * 128, D, C.att, b, [&](int i, int row, float v) { C.h[(size_t)i * D + row] += v; }); break;
    case D_HEAD: {                                             // logits for 128 vocab rows; per-task first-max and sum partials (fixed order)
      float* lg = sm;                                          // [b][128]
      gemv128(C.E, D, local * 128, VOC, C.h, b, [&](int i, int row, float v) { lg[i * 128 + row - local * 128] = v; if (step == 0) C.logits0[(size_t)i * VOC + row] = v; });
      __syncthreads();
      if (tid < b) { float m = -INFINITY, s = 0.f; int mi = 0; for (int r = 0; r < 128; r++) { const float v = lg[tid * 128 + r]; s += v; if (v > m) { m = v; mi = local * 128 + r; } }
        C.lmax[(size_t)tid * d.ntasks + local] = m; C.lidx[(size_t)tid * d.ntasks + local] = mi; C.lsum_part[(size_t)tid * d.ntasks + local] = s; }
      break;
    }
    case D_NEXT: {                                             // greedy token, checksum, next input = tied embedding row
      const int nt = VOC / 128;
      __shared__ int stok[MAXB];
      if (tid < b) { float m = -INFINITY, s = 0.f; int mi = 0; for (int t = 0; t < nt; t++) { const float v = C.lmax[(size_t)tid * nt + t]; s += C.lsum_part[(size_t)tid * nt + t]; if (v > m) { m = v; mi = C.lidx[(size_t)tid * nt + t]; } }
        stok[tid] = mi; C.toks[(size_t)step * MAXB + tid] = mi; C.lsum[(size_t)step * MAXB + tid] = s; }
      __syncthreads();
      for (int e = tid; e < b * D; e += THREADS) C.h[e] = bf(C.E[(size_t)stok[e / D] * D + e % D]);
      break;
    }
    default: break;
  }
}

__device__ __forceinline__ void finish_train(const Ctx& C, const Job& J, int ph, int step) {
  const int old = atomicAdd(J.done + ph, 1);
  if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) {
    C.step_done[step] = gtimer(); if (step == J.limit - 1) { __threadfence(); *reinterpret_cast<volatile int*>(C.finished) = 1; } }
}
// --lend: a decode block whose next decode task is not ready runs ONE training task, but only one whose dependencies are already met
// (peek the ticket, check deps, CAS-claim exactly it), so lending never waits on training.
__device__ bool try_lend(const Ctx& C, char* smem) {
  __shared__ volatile int s_l[4];
  const Job& J = C.tr;
  if (threadIdx.x == 0) {
    int ok = 0, step = 0, ph = 0, local = 0;
    const unsigned g = *reinterpret_cast<volatile unsigned*>(J.ticket);
    if (g < (unsigned)J.limit * (unsigned)J.tps) {
      step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
      while (r >= J.prefix[ph + 1]) ph++;
      local = r - J.prefix[ph]; bool ready = true;
      for (int k = 0; k < J.dep_cnt[ph] && ready; k++) { const int2 e = J.deps[J.dep_off[ph] + k]; const int need = J.ntasks[e.x] * (step + 1 + e.y);
        if (need > 0 && ld_acq(J.done + e.x) < need) ready = false; }
      if (ready && atomicCAS(J.ticket, g, g + 1) == g) ok = 1;
    }
    s_l[0] = ok; s_l[1] = step; s_l[2] = ph; s_l[3] = local;
  }
  __syncthreads();
  const int ok = s_l[0], step = s_l[1], ph = s_l[2], local = s_l[3];
  __syncthreads();
  if (!ok) return false;
  train_task(C, J.ph[ph], local, step, smem);
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) { finish_train(C, J, ph, step); atomicAdd(C.lent, 1u); }
  __syncthreads();
  return true;
}

// ------------------------------------------------------------------------------------------------ persistent loop (OFA fetch-add ticket claim)
__device__ void job_loop(const Ctx& C, const Job& J, int job_id, char* smem) {
  __shared__ volatile int s_sel[4];
  const int tid = threadIdx.x; int cph = 0, rstep = -1, rph = -1, pend = 0; unsigned pend_g = 0;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  for (;;) {
    if (tid == 0) {
      int res = 0, step = 0, ph = 0, local = 0;
      if (job_id == 1 && C.mode == 2 && ld_vol(C.finished)) res = -2;
      else {
        unsigned g; if (pend) { g = pend_g; pend = 0; } else g = atomicAdd(J.ticket, 1u);
        if (g >= total) res = -2;
        else {
          step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
          if (r < J.prefix[cph]) cph = 0;
          while (r >= J.prefix[cph + 1]) cph++;
          ph = cph; local = r - J.prefix[ph];
          if (!(step == rstep && ph == rph)) {
            for (;;) {
              bool ready = true;
              for (int k = 0; k < J.dep_cnt[ph] && ready; k++) {
                const int2 e = J.deps[J.dep_off[ph] + k]; const int need = J.ntasks[e.x] * (step + 1 + e.y);
                if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;
              }
              if (ready) { rstep = step; rph = ph; break; }
              if (job_id == 1 && C.mode == 2 && ld_vol(C.finished)) { res = -2; break; }
              if (job_id == 1 && C.lend) { pend = 1; pend_g = g; res = -3; break; }
              __nanosleep(64);
            }
          }
        }
      }
      s_sel[0] = res; s_sel[1] = step; s_sel[2] = ph; s_sel[3] = local;
    }
    __syncthreads();
    const int res = s_sel[0], step = s_sel[1], ph = s_sel[2], local = s_sel[3];
    __syncthreads();
    if (res == -3) { if (!try_lend(C, smem) && tid == 0) __nanosleep(64); continue; }
    if (res < 0) break;
    if (job_id == 0) train_task(C, J.ph[ph], local, step, smem); else decode_task(C, J.ph[ph], local, step, smem);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      if (job_id == 0) finish_train(C, J, ph, step);
      else { const int old = atomicAdd(J.done + ph, 1); if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) C.tok_done[step] = gtimer(); }
    }
  }
}

__global__ void __launch_bounds__(THREADS, 2) mw_kernel(const Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  unsigned hw; asm volatile("mov.u32 %0, %%smid;" : "=r"(hw));
  const int smid = C.smid_rank ? C.smid_rank[hw] : (int)hw;
  char* smem = reinterpret_cast<char*>(dsm);
  if (C.mode == 0) job_loop(C, C.tr, 0, smem);
  else if (C.mode == 1) { if (C.infer_sms <= 0 || smid < C.infer_sms) job_loop(C, C.dc, 1, smem); }
  else if (smid < C.infer_sms) job_loop(C, C.dc, 1, smem);
  else job_loop(C, C.tr, 0, smem);
}
__global__ void __launch_bounds__(THREADS, 2) smid_probe(unsigned* seen) {
  if (threadIdx.x == 0) { unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); atomicOr(&seen[id >> 5], 1u << (id & 31));
    const long long t0 = clock64(); while (clock64() - t0 < 4000000ll) {} }
}
__device__ __forceinline__ float hrand(u64 seed, u64 i) { u64 z = seed * 0x9e3779b97f4a7c15ull + i; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; z ^= z >> 31;
  return (float)((z >> 40) * (1.0 / 16777216.0)) * 2.f - 1.f; }
__global__ void fill_bf16(bf16* p, size_t n, u64 seed, float sc) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(sc * hrand(seed, i)); }
__global__ void fill_f32(float* p, size_t n, u64 seed, float sc) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = sc * hrand(seed, i); }

// ------------------------------------------------------------------------------------------------ host
struct Builder {
  std::vector<Ph> ph; std::vector<int> dep_off, dep_cnt, ntasks; std::vector<int2> deps;
  int add(Ph d, std::vector<int2> after) { dep_off.push_back((int)deps.size()); for (auto a : after) deps.push_back(a); dep_cnt.push_back((int)after.size());
    ph.push_back(d); ntasks.push_back(d.ntasks); return (int)ph.size() - 1; }
};
static Ph mk(int kind, int n) { Ph d; memset(&d, 0, sizeof d); d.kind = kind; d.ntasks = n; return d; }
template <class X> static X* dalloc(size_t n) { X* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(X))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(X))); return p; }
static void upload(Builder& b, int limit, int last, Job& J) {
  const int n = (int)b.ph.size(); std::vector<int> pre(n + 1, 0); for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + b.ntasks[i];
  auto up = [](const void* src, size_t bytes) { void* p; CK(cudaMalloc(&p, std::max<size_t>(bytes, 4))); if (bytes) CK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); return p; };
  J.nph = n; J.tps = pre[n]; J.limit = limit; J.last_ph = last;
  J.ph = (const Ph*)up(b.ph.data(), n * sizeof(Ph)); J.prefix = (const int*)up(pre.data(), (n + 1) * 4); J.ntasks = (const int*)up(b.ntasks.data(), n * 4);
  J.dep_off = (const int*)up(b.dep_off.data(), n * 4); J.dep_cnt = (const int*)up(b.dep_cnt.data(), n * 4); J.deps = (const int2*)up(b.deps.data(), b.deps.size() * sizeof(int2));
  J.done = dalloc<int>(n); J.ticket = dalloc<unsigned>(1);
}
template <class X> static void dump(const std::string& dir, const char* name, const X* dev, size_t n) {
  if (dir.empty()) return; std::vector<X> h(n); CK(cudaMemcpy(h.data(), dev, n * sizeof(X), cudaMemcpyDeviceToHost));
  FILE* f = fopen((dir + "/" + name).c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s/%s\n", dir.c_str(), name); exit(1); } fwrite(h.data(), sizeof(X), n, f); fclose(f);
}
static void pct(std::vector<double> v, double& p50, double& p99) { if (v.empty()) { p50 = p99 = 0; return; } std::sort(v.begin(), v.end()); p50 = v[v.size() / 2]; p99 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))]; }

int main(int argc, char** argv) {
  std::string mode_s = "co", dir; int jobs = 16, tk = 512, steps = 16, b = 1, ctx0 = 1024, dtok = 2048, infer_sms = 33, nx = 4, lend = 0; u64 seed = 1234;
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--mode")) mode_s = argv[++i]; else if (is("--jobs")) jobs = atoi(argv[++i]); else if (is("--tokens")) tk = atoi(argv[++i]);
    else if (is("--steps")) steps = atoi(argv[++i]); else if (is("--b")) b = atoi(argv[++i]); else if (is("--ctx")) ctx0 = atoi(argv[++i]);
    else if (is("--dec-tokens")) dtok = atoi(argv[++i]); else if (is("--infer-sms")) infer_sms = atoi(argv[++i]); else if (is("--dump")) dir = argv[++i];
    else if (!strcmp(argv[i], "--lend")) lend = 1; else if (is("--seed")) seed = atoll(argv[++i]); else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  const int mode = mode_s == "train" ? 0 : mode_s == "dec" ? 1 : mode_s == "co" ? 2 : -1;
  if (mode < 0 || jobs < 1 || jobs > THREADS || tk % 128 || b < 1 || b > MAXB || ctx0 < 1 || steps < 1 || (mode != 0 && ctx0 + dtok > MAXPOS)) {
    fprintf(stderr, "bad args: --mode train|dec|co, --tokens multiple of 128, 1<=b<=8, ctx+dec-tokens<=%d\n", MAXPOS); return 2; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); const int nsm = prop.multiProcessorCount;
  if (mode == 2 && (infer_sms < 1 || infer_sms >= nsm)) { fprintf(stderr, "--infer-sms must be in [1,%d)\n", nsm); return 2; }
  if (mode == 1 && infer_sms >= nsm) infer_sms = 0;
  CK(cudaFuncSetAttribute(mw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(smid_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mw_kernel, THREADS, SMEM)); occ = std::min(occ, 2);
  const int nblocks = nsm * occ;
  int* d_rank = nullptr;
  { unsigned* seen; CK(cudaMallocManaged(&seen, 64 * 4)); memset(seen, 0, 256); smid_probe<<<nblocks, THREADS, SMEM>>>(seen); CK(cudaDeviceSynchronize());
    std::vector<int> rk(2048, 0); int n = 0, mx = -1; for (int id = 0; id < 2048; id++) if (seen[id >> 5] >> (id & 31) & 1) { rk[id] = n++; mx = id; }
    if (n != nsm) printf("WARNING: smid probe saw %d of %d SMs (MPS limit or busy GPU)\n", n, nsm);
    if (mx != n - 1) { CK(cudaMalloc(&d_rank, 2048 * 4)); CK(cudaMemcpy(d_rank, rk.data(), 2048 * 4, cudaMemcpyHostToDevice)); }
    cudaFree(seen); }
  printf("GPU %s, %d SMs, %d blocks (%d/SM); mode %s; train jobs %d x %d tokens, q/v rank %d, %d steps; decode b=%d ctx %d, %d tokens, infer SMs %d\n",
         prop.name, nsm, nblocks, occ, mode_s.c_str(), jobs, tk, R, steps, b, ctx0, dtok, mode == 0 ? 0 : infer_sms);

  Ctx C; memset(&C, 0, sizeof C); C.mode = mode; C.infer_sms = infer_sms; C.lend = mode == 2 && lend; C.lent = dalloc<unsigned>(1); C.smid_rank = d_rank; C.finished = dalloc<int>(1);
  C.jobs = jobs; C.tk = tk; C.nx = nx; C.scale = 2.f; C.lr = 1e-3f; C.b1 = 0.9f; C.b2 = 0.999f; C.eps = 1e-8f; C.inv = 1.f / ((float)tk * NQV); C.b = b; C.ctx0 = ctx0;
  auto fb = [&](size_t n, u64 s, float sc) { bf16* p = dalloc<bf16>(n); fill_bf16<<<1024, 256>>>(p, n, seed * 131 + s, sc); return p; };
  auto ff = [&](size_t n, u64 s, float sc) { float* p = dalloc<float>(n); fill_f32<<<1024, 256>>>(p, n, seed * 131 + s, sc); return p; };
  const float ws = 1.7320508f / sqrtf((float)D);              // uniform(-1,1)*sqrt(3/K): unit-variance outputs
  bf16* Wqkv = fb((size_t)NQKV * D, 1, ws);                  // rows: q 4096 | k 1024 | v 1024, [out][in] like nn.Linear
  C.Wq = Wqkv; C.Wv = Wqkv + (size_t)(NQ + NKV) * D; C.Wqkv = Wqkv;
  C.Wo = fb((size_t)D * NQ, 2, 1.7320508f / sqrtf((float)NQ)); C.E = fb((size_t)VOC * D, 3, 1.7320508f);
  // train state
  C.X = fb((size_t)nx * tk * D, 4, 1.7320508f); C.Tq = fb((size_t)tk * NQ, 5, 1.7320508f); C.Tv = fb((size_t)tk * NKV, 6, 1.7320508f);
  C.Yq = dalloc<bf16>((size_t)tk * NQ); C.Yv = dalloc<bf16>((size_t)tk * NKV);
  C.P = dalloc<float>((size_t)jobs * PJ);
  for (int j = 0; j < jobs; j++) fill_f32<<<256, 256>>>(C.P + (size_t)j * PJ, 2 * R * D, seed * 977 + j, 1.7320508f / sqrtf((float)D));   // A random, B zero (LoRA init)
  C.G = dalloc<float>((size_t)jobs * PJ); C.M = dalloc<float>((size_t)jobs * PJ); C.Vm = dalloc<float>((size_t)jobs * PJ);
  C.U = dalloc<float>((size_t)jobs * tk * R2); C.dU = dalloc<float>((size_t)jobs * tk * R2); C.dO = dalloc<float>((size_t)jobs * tk * NQV);
  C.lpart = dalloc<float>((size_t)jobs * (tk / ROWS)); C.loss = dalloc<float>((size_t)steps * jobs); C.step_done = dalloc<u64>(steps);
  // decode state: one adapter (A and B random), KV cache prefilled for ctx0 positions, inputs = embeddings of tokens 1..b
  C.adp = ff(PJ, 7, 1.7320508f / sqrtf((float)D));
  { float* a = const_cast<float*>(C.adp); fill_f32<<<256, 256>>>(a + 2 * R * D, (size_t)NQV * R, seed * 131 + 8, 0.02f); }
  C.kc = fb((size_t)MAXB * MAXPOS * NKV, 9, 1.7320508f); C.vc = fb((size_t)MAXB * MAXPOS * NKV, 10, 1.7320508f);
  C.h = dalloc<float>((size_t)MAXB * D); C.du = dalloc<float>(MAXB * R2); C.q = dalloc<float>((size_t)MAXB * NQ); C.att = dalloc<float>((size_t)MAXB * NQ);
  const int nhead = VOC / 128;
  C.lmax = dalloc<float>((size_t)MAXB * nhead); C.lidx = dalloc<int>((size_t)MAXB * nhead); C.lsum_part = dalloc<float>((size_t)MAXB * nhead);
  C.toks = dalloc<int>((size_t)dtok * MAXB); C.lsum = dalloc<float>((size_t)dtok * MAXB); C.logits0 = dalloc<float>((size_t)MAXB * VOC); C.tok_done = dalloc<u64>(dtok);
  CK(cudaDeviceSynchronize());
  { std::vector<bf16> e((size_t)MAXB * D); CK(cudaMemcpy(e.data(), C.E + D, e.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> h(e.size()); for (size_t i = 0; i < e.size(); i++) h[i] = __bfloat162float(e[i]); CK(cudaMemcpy(C.h, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  if (!dir.empty() && mode != 1) { dump(dir, "init_P.f32", C.P, (size_t)jobs * PJ); dump(dir, "X.bf16", C.X, (size_t)nx * tk * D); dump(dir, "Wq.bf16", C.Wq, (size_t)NQ * D);
    dump(dir, "Wv.bf16", C.Wv, (size_t)NKV * D); dump(dir, "Tq.bf16", C.Tq, (size_t)tk * NQ); dump(dir, "Tv.bf16", C.Tv, (size_t)tk * NKV); }

  // train graph (per step); (phase, -1) = previous step
  Builder Bt; const int rbt = jobs * (tk / ROWS);
  Ph q = mk(T_BASEQ, 0), v = mk(T_BASEV, 0);
  for (Ph* p : {&q, &v}) { const int n = p == &q ? NQ : NKV; p->ga.A = C.X; p->ga.B = p == &q ? C.Wq : C.Wv; p->ga.C = p == &q ? (void*)C.Yq : (void*)C.Yv;
    p->ga.lda = D; p->ga.ldb = D; p->ga.ldc = n; p->ga.ldaux = 8; p->ga.M = tk; p->ga.N = n; p->ga.K = D; p->tilesM = tk / 128; p->ntasks = (tk / 128) * (n / 128);
    if (!gemm::args_ok(p->ga)) { fprintf(stderr, "bad GEMM args\n"); return 2; } }
  const int ADAM = 7, LOSS = 8;
  Bt.add(q, {make_int2(LOSS, -1)}); Bt.add(v, {make_int2(LOSS, -1)});
  Bt.add(mk(T_DOWN, rbt), {make_int2(ADAM, -1)});
  Bt.add(mk(T_UPG, rbt), {make_int2(0, 0), make_int2(1, 0), make_int2(2, 0)});
  Bt.add(mk(T_DB, jobs * (NQV / THREADS)), {make_int2(3, 0)});
  Bt.add(mk(T_DU, rbt), {make_int2(3, 0)});
  Bt.add(mk(T_DA, jobs * (D / THREADS)), {make_int2(5, 0)});
  Bt.add(mk(T_ADAM, jobs * (PJ / (THREADS * 4))), {make_int2(4, 0), make_int2(6, 0)});
  Bt.add(mk(T_LOSS, 1), {make_int2(3, 0)});
  upload(Bt, mode == 1 ? 0 : steps, ADAM, C.tr);
  // decode graph (per token)
  Builder Bd;
  Bd.add(mk(D_DOWN, 1), {make_int2(5, -1)}); Bd.add(mk(D_QKV, NQKV / 128), {make_int2(0, 0)}); Bd.add(mk(D_ATT, b * NKVH), {make_int2(1, 0)});
  Bd.add(mk(D_O, D / 128), {make_int2(2, 0)}); Bd.add(mk(D_HEAD, nhead), {make_int2(3, 0)}); Bd.add(mk(D_NEXT, 1), {make_int2(4, 0)});
  upload(Bd, mode == 0 ? 0 : dtok, 5, C.dc);
  printf("train: %d phases, %d tasks/step; decode: 6 phases, %d tasks/token\n", (int)Bt.ph.size(), C.tr.tps, C.dc.tps);
  { size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot)); printf("device memory used %.1f of %.1f GB\n", (tot - fr) / 1e9, tot / 1e9); }

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); mw_kernel<<<nblocks, THREADS, SMEM>>>(C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); unsigned lent = 0; CK(cudaMemcpy(&lent, C.lent, 4, cudaMemcpyDeviceToHost)); printf("kernel %.3f s, lending %s, lent train tasks %u\n", ms / 1e3, C.lend ? "on" : "off", lent);

  u64 tr_lo = 0, tr_hi = 0;
  if (mode != 1) {
    std::vector<u64> sd(steps); CK(cudaMemcpy(sd.data(), C.step_done, steps * 8, cudaMemcpyDeviceToHost));
    std::vector<float> loss((size_t)steps * jobs); CK(cudaMemcpy(loss.data(), C.loss, loss.size() * 4, cudaMemcpyDeviceToHost));
    const int w0 = std::min(steps - 1, 3); tr_lo = sd[w0]; tr_hi = sd[steps - 1];
    const double span = (tr_hi - tr_lo) * 1e-9; const int n = steps - 1 - w0;
    printf("loss job0: step0 %.6f last %.6f\n", loss[0], loss[(size_t)(steps - 1) * jobs]);
    if (n > 0) printf("train: steady %.4f ms/step -> %.2f steps/s (%d jobs x %d tokens: %.0f job-tok/s)\n", span * 1e3 / n, n / span, jobs, tk, (double)n * jobs * tk / span);
    dump(dir, "loss.f32", C.loss, (size_t)steps * jobs); dump(dir, "step_done.u64", C.step_done, (size_t)steps); dump(dir, "P.f32", C.P, (size_t)jobs * PJ); dump(dir, "G.f32", C.G, (size_t)jobs * PJ);
  }
  if (mode != 0) {
    std::vector<u64> td(dtok); CK(cudaMemcpy(td.data(), C.tok_done, dtok * 8, cudaMemcpyDeviceToHost));
    int done = 0; while (done < dtok && td[done]) done++;
    std::vector<double> lat; int inwin = 0; u64 lo = 0, hi = 0;
    for (int t = 1; t < done; t++) {
      if (mode == 2 && (td[t] < tr_lo || td[t] > tr_hi)) continue;
      lat.push_back((td[t] - td[t - 1]) * 1e-6); if (!inwin) lo = td[t - 1]; hi = td[t]; inwin++;
    }
    double p50, p99; pct(lat, p50, p99); const double win = (hi - lo) * 1e-9;
    printf("decode: b=%d, %d token steps (%d in window %.3f s) -> %.1f tok/s, tpot p50 %.4f ms p99 %.4f ms\n", b, done, inwin, win, inwin * b / (win + 1e-12), p50, p99);
    if (!dir.empty()) { dump(dir, "toks.i32", C.toks, (size_t)done * MAXB); dump(dir, "tok_done.u64", C.tok_done, (size_t)done); dump(dir, "lsum.f32", C.lsum, (size_t)done * MAXB); dump(dir, "logits0.f32", C.logits0, (size_t)b * VOC); }
  }
  return 0;
}
