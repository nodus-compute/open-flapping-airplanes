// sched.cu : SLO-aware instruction scheduler inside one persistent megakernel (H100, sm_90a), built on the mw arm's task graphs.
//   tenants : decode (batch-b requests of --req-tokens tokens, open-loop Poisson arrivals at --rate req/s; one Qwen3-4B-shaped layer +
//             tied lm_head, greedy) and training (16-job shared-trunk LoRA tail, Qwen3-4B q/v, rank 8, top-1 layer, fwd/bwd/AdamW).
//   --mode dec   : decode alone (all SMs).          --mode train : training alone (--steps cap, optional --max-seconds).
//   --mode mix --sched prio : every SM pulls a READY decode instruction first and a training tile only when no decode instruction is
//             ready; preemption is at instruction granularity (no kernel boundary, no SM partition).
//   --mode mix --sched fifo : same non-blocking queues, no priority (blocks alternate which queue they try first).
//   Training stops when the last decode token completes. Fixed reduction order, no float atomics: decode outputs are byte-equal to
//   decode alone, and training losses byte-equal to training alone, regardless of interleaving.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../../gemm/gemm.cuh"
#ifdef MW_GEMM_WG
#include "../gemm_sm90.cuh"
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

constexpr int D = 2560, NQ = 4096, NKV = 1024, NQV = NQ + NKV, NQKV = NQ + 2 * NKV, HD = 128, NKVH = 8, VOC = 151936, R = 8, R2 = 2 * R;
constexpr int PJ = 2 * R * D + NQV * R;            // per-job fp32 params: Aq[R][D] Av[R][D] Bq[NQ][R] Bv[NKV][R]
constexpr int MAXB = 8, MAXPOS = 4096, THREADS = 256, ROWS = 16;
#ifdef MW_GEMM_WG
constexpr int SMEM = gemm::sm90::smem_bytes() > gemm::smem_bytes() ? gemm::sm90::smem_bytes() : gemm::smem_bytes();
#else
constexpr int SMEM = gemm::smem_bytes();
#endif
//           // 64 KB
static_assert(4 * MAXPOS * 4 <= SMEM, "attention scores fit the GEMM smem");

enum Kind { T_BASEQ, T_BASEV, T_DOWN, T_UPG, T_DB, T_DU, T_DA, T_ADAM, T_LOSS, D_DOWN, D_QKV, D_ATT, D_O, D_HEAD, D_NEXT };
struct Ph { int kind, ntasks, tilesM; gemm::Args ga;
#ifdef MW_GEMM_WG
  const CUtensorMap* tmB; const CUtensorMap* const* tmAx;   // wgmma/TMA backend: B map + one A map per X slot
#endif
};
struct Job {
  int nph, tps, limit, last_ph;
  const Ph* ph; const int *prefix, *ntasks, *dep_off, *dep_cnt; const int2* deps;
  int* done; unsigned* ticket;
};
struct Ctx {
  Job tr, dc; int mode, infer_sms, lend; unsigned* lent; const int* smid_rank; int* finished;
  int sched, L; const u64* arr; u64* t0; u64 max_ns;
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
#ifdef MW_GEMM_WG
      gemm::sm90::ArgsWG a; static_cast<gemm::Args&>(a) = d.ga; a.A = X; a.tmA = d.tmAx[step % C.nx]; a.tmB = d.tmB;
      gemm::sm90::tile<gemm::NT, gemm::EPI_BF16>(a, local % d.tilesM, local / d.tilesM, smem); break;
#else
      gemm::Args a = d.ga; a.A = X;
      gemm::gemm_tile<gemm::NT, gemm::EPI_BF16>(a, local % d.tilesM, local / d.tilesM, smem); break;
#endif
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
  const int tid = threadIdx.x, b = C.b, pos = C.ctx0 + step % C.L;
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

__device__ __forceinline__ u64 t0v(const Ctx& C) { return *reinterpret_cast<const volatile u64*>(C.t0); }
__device__ __forceinline__ bool stop_train(const Ctx& C) { return ld_vol(C.finished) || (C.max_ns && gtimer() - t0v(C) > C.max_ns); }
// phase ready: dependency counters, plus (decode) a request's first token is released only at its arrival time
__device__ __forceinline__ bool ph_ready(const Ctx& C, const Job& J, int dec, int step, int ph) {
  for (int k = 0; k < J.dep_cnt[ph]; k++) { const int2 e = J.deps[J.dep_off[ph] + k]; const int need = J.ntasks[e.x] * (step + 1 + e.y);
    if (need > 0 && ld_acq(J.done + e.x) < need) return false; }
  if (dec && ph == 0 && step % C.L == 0 && gtimer() < t0v(C) + C.arr[step / C.L]) return false;
  return true;
}
__device__ __forceinline__ void finish(const Ctx& C, const Job& J, int dec, int ph, int step) {
  const int old = atomicAdd(J.done + ph, 1);
  if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) {
    if (dec) { C.tok_done[step] = gtimer(); if (step == J.limit - 1) { __threadfence(); *reinterpret_cast<volatile int*>(C.finished) = 1; } }
    else C.step_done[step] = gtimer();
  }
}
__device__ __forceinline__ void run_task(const Ctx& C, const Job& J, int dec, int ph, int local, int step, char* smem) {
  if (dec) decode_task(C, J.ph[ph], local, step, smem); else train_task(C, J.ph[ph], local, step, smem);
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) finish(C, J, dec, ph, step);
  __syncthreads();
}
// non-blocking claim: peek the queue head, run it only if it is ready (CAS-claim exactly it). 1 = ran, 0 = head not ready, 2 = queue done.
__device__ int try_claim(const Ctx& C, const Job& J, int dec, char* smem) {
  __shared__ volatile int s_l[4];
  if (threadIdx.x == 0) {
    int st = 0, step = 0, ph = 0, local = 0;
    const unsigned g = *reinterpret_cast<volatile unsigned*>(J.ticket);
    if (g >= (unsigned)J.limit * (unsigned)J.tps || (!dec && stop_train(C))) st = 2;
    else {
      step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
      while (r >= J.prefix[ph + 1]) ph++;
      local = r - J.prefix[ph];
      if (ph_ready(C, J, dec, step, ph) && atomicCAS(J.ticket, g, g + 1) == g) st = 1;
    }
    s_l[0] = st; s_l[1] = step; s_l[2] = ph; s_l[3] = local;
  }
  __syncthreads();
  const int st = s_l[0], step = s_l[1], ph = s_l[2], local = s_l[3];
  __syncthreads();
  if (st == 1) run_task(C, J, dec, ph, local, step, smem);
  return st;
}
// one tenant alone (OFA fetch-add ticket claim, blocking dependency wait)
__device__ void job_loop(const Ctx& C, const Job& J, int dec, char* smem) {
  __shared__ volatile int s_sel[4];
  const int tid = threadIdx.x; int cph = 0;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  for (;;) {
    if (tid == 0) {
      int res = 0, step = 0, ph = 0, local = 0;
      if (!dec && stop_train(C)) res = -2;
      else {
        const unsigned g = atomicAdd(J.ticket, 1u);
        if (g >= total) res = -2;
        else {
          step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
          if (r < J.prefix[cph]) cph = 0;
          while (r >= J.prefix[cph + 1]) cph++;
          ph = cph; local = r - J.prefix[ph];
          while (!ph_ready(C, J, dec, step, ph)) { if (!dec && stop_train(C)) { res = -2; break; } __nanosleep(64); }
        }
      }
      s_sel[0] = res; s_sel[1] = step; s_sel[2] = ph; s_sel[3] = local;
    }
    __syncthreads();
    const int res = s_sel[0], step = s_sel[1], ph = s_sel[2], local = s_sel[3];
    __syncthreads();
    if (res < 0) break;
    run_task(C, J, dec, ph, local, step, smem);
  }
}
// both tenants on every SM: prio = decode queue first, fifo = alternate the first queue each pull
__device__ void mix_loop(const Ctx& C, char* smem) {
  for (unsigned it = 0;; it++) {
    const int dfirst = C.sched == 1 || (it & 1);
    const int a = try_claim(C, dfirst ? C.dc : C.tr, dfirst, smem);
    if (a == 1) continue;
    const int b = try_claim(C, dfirst ? C.tr : C.dc, !dfirst, smem);
    if (b == 1) continue;
    if (a == 2 && b == 2) break;
    if (threadIdx.x == 0) __nanosleep(64);
    __syncthreads();
  }
}

__global__ void __launch_bounds__(THREADS, 2) sched_kernel(const Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  char* smem = reinterpret_cast<char*>(dsm);
  if (threadIdx.x == 0) atomicCAS(C.t0, 0ull, gtimer());
  __syncthreads();
  if (C.mode == 0) job_loop(C, C.tr, 0, smem);
  else if (C.mode == 1) job_loop(C, C.dc, 1, smem);
  else mix_loop(C, smem);
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
  std::string mode_s = "mix", sched_s = "prio", dir; int jobs = 16, tk = 512, steps = 200000, b = 8, ctx0 = 1024, nreq = 64, L = 64, nx = 4;
  double rate = 4.0, max_s = 0.0; u64 seed = 1234;
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--mode")) mode_s = argv[++i]; else if (is("--sched")) sched_s = argv[++i]; else if (is("--jobs")) jobs = atoi(argv[++i]);
    else if (is("--tokens")) tk = atoi(argv[++i]); else if (is("--steps")) steps = atoi(argv[++i]); else if (is("--b")) b = atoi(argv[++i]);
    else if (is("--ctx")) ctx0 = atoi(argv[++i]); else if (is("--rate")) rate = atof(argv[++i]); else if (is("--nreq")) nreq = atoi(argv[++i]);
    else if (is("--req-tokens")) L = atoi(argv[++i]); else if (is("--max-seconds")) max_s = atof(argv[++i]); else if (is("--dump")) dir = argv[++i];
    else if (is("--seed")) seed = atoll(argv[++i]); else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  const int mode = mode_s == "train" ? 0 : mode_s == "dec" ? 1 : mode_s == "mix" ? 2 : -1, sched = sched_s == "fifo" ? 0 : sched_s == "prio" ? 1 : -1;
  const int dtok = nreq * L;
  if (mode < 0 || sched < 0 || jobs < 1 || jobs > THREADS || tk % 128 || b < 1 || b > MAXB || ctx0 < 1 || steps < 1 || nreq < 1 || L < 2 || rate <= 0 ||
      (mode != 0 && ctx0 + L > MAXPOS)) {
    fprintf(stderr, "bad args: --mode train|dec|mix --sched fifo|prio, --tokens multiple of 128, 1<=b<=8, ctx+req-tokens<=%d, rate>0\n", MAXPOS); return 2; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); const int nsm = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(sched_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sched_kernel, THREADS, SMEM)); occ = std::min(occ, 2);
  const int nblocks = nsm * occ;
  printf("GPU %s, %d SMs, %d blocks (%d/SM); mode %s sched %s; train jobs %d x %d tokens, q/v rank %d, step cap %d, max %.1f s; "
         "decode b=%d ctx %d, %d requests x %d tokens at %.2f req/s (Poisson)\n",
         prop.name, nsm, nblocks, occ, mode_s.c_str(), sched_s.c_str(), jobs, tk, R, steps, max_s, b, ctx0, nreq, L, rate);

  Ctx C; memset(&C, 0, sizeof C); C.mode = mode; C.sched = sched; C.L = L; C.finished = dalloc<int>(1); C.t0 = dalloc<u64>(1);
  C.max_ns = (u64)(max_s * 1e9); C.lent = dalloc<unsigned>(1);
  C.jobs = jobs; C.tk = tk; C.nx = nx; C.scale = 2.f; C.lr = 1e-3f; C.b1 = 0.9f; C.b2 = 0.999f; C.eps = 1e-8f; C.inv = 1.f / ((float)tk * NQV); C.b = b; C.ctx0 = ctx0;
  std::vector<u64> arr(nreq);
  { std::mt19937_64 g(seed * 7 + 11); std::exponential_distribution<double> ex(rate); double t = 0.02; for (int k = 0; k < nreq; k++) { t += ex(g); arr[k] = (u64)(t * 1e9); } }
  { u64* a = dalloc<u64>(nreq); CK(cudaMemcpy(a, arr.data(), nreq * 8, cudaMemcpyHostToDevice)); C.arr = a; }
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

  // train graph (per step); (phase, -1) = previous step
  Builder Bt; const int rbt = jobs * (tk / ROWS);
  Ph q = mk(T_BASEQ, 0), v = mk(T_BASEV, 0);
  for (Ph* p : {&q, &v}) { const int n = p == &q ? NQ : NKV; p->ga.A = C.X; p->ga.B = p == &q ? C.Wq : C.Wv; p->ga.C = p == &q ? (void*)C.Yq : (void*)C.Yv;
    p->ga.lda = D; p->ga.ldb = D; p->ga.ldc = n; p->ga.ldaux = 8; p->ga.M = tk; p->ga.N = n; p->ga.K = D; p->tilesM = tk / 128; p->ntasks = (tk / 128) * (n / 128);
    if (!gemm::args_ok(p->ga)) { fprintf(stderr, "bad GEMM args\n"); return 2; } }
#ifdef MW_GEMM_WG
  for (Ph* p : {&q, &v}) { std::vector<const CUtensorMap*> maps(nx);
    for (int x = 0; x < nx; x++) maps[x] = gemm::sm90::make_map(C.X + (size_t)x * tk * D, D, tk, D, 64, 128);
    const CUtensorMap** dm; CK(cudaMalloc(&dm, nx * sizeof(void*))); CK(cudaMemcpy(dm, maps.data(), nx * sizeof(void*), cudaMemcpyHostToDevice));
    p->tmAx = dm; p->tmB = gemm::sm90::make_map(p->ga.B, D, p->ga.N, D, 64, 128); }
  printf("gemm_backend wgmma\n");
#else
  printf("gemm_backend mma_sync\n");
#endif
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
  CK(cudaEventRecord(e0)); sched_kernel<<<nblocks, THREADS, SMEM>>>(C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); u64 t0 = 0; CK(cudaMemcpy(&t0, C.t0, 8, cudaMemcpyDeviceToHost));
  printf("kernel %.3f s\n", ms / 1e3);
  int done = 0, sdone = 0; std::vector<u64> td(std::max(dtok, 1)), sd(steps);
  u64 wlo = 0, whi = 0;
  if (mode != 0) {
    CK(cudaMemcpy(td.data(), C.tok_done, dtok * 8, cudaMemcpyDeviceToHost)); while (done < dtok && td[done]) done++;
    wlo = t0 + arr[0]; whi = done ? td[done - 1] : wlo;
  }
  if (mode != 1) { CK(cudaMemcpy(sd.data(), C.step_done, steps * 8, cudaMemcpyDeviceToHost)); while (sdone < steps && sd[sdone]) sdone++; }
  double rp50 = 0, rp99 = 0, tp50 = 0, tp99 = 0; int nfull = 0;
  if (mode != 0) {
    std::vector<double> rl, tp;
    for (int k = 0; k < nreq; k++) { const int last = k * L + L - 1; if (last >= done) break; nfull++; rl.push_back((td[last] - (t0 + arr[k])) * 1e-6);
      for (int t = k * L + 1; t <= last; t++) tp.push_back((td[t] - td[t - 1]) * 1e-6); }
    pct(rl, rp50, rp99); pct(tp, tp50, tp99);
    printf("decode: %d/%d requests complete, request latency p50 %.3f ms p99 %.3f ms, tpot p50 %.4f ms p99 %.4f ms, window %.3f s\n",
           nfull, nreq, rp50, rp99, tp50, tp99, (whi - wlo) * 1e-9);
  }
  int win_steps = 0; double win_s = 0, sps = 0;
  if (mode != 1) {
    if (mode == 2) { for (int s2 = 0; s2 < sdone; s2++) if (sd[s2] >= wlo && sd[s2] <= whi) win_steps++; win_s = (whi - wlo) * 1e-9; }
    else { const int w0 = std::min(std::max(sdone - 1, 0), 3); win_steps = sdone - 1 - w0; win_s = sdone > w0 + 1 ? (sd[sdone - 1] - sd[w0]) * 1e-9 : 0; }
    sps = win_s > 0 ? win_steps / win_s : 0;
    printf("train: %d steps done, %d in window %.3f s -> %.2f steps/s (%.0f steps per GPU-hour)\n", sdone, win_steps, win_s, sps, sps * 3600);
  }
  if (!dir.empty()) {
    dump(dir, "t0.u64", C.t0, 1); dump(dir, "arr.u64", C.arr, (size_t)nreq);
    if (mode != 0) { dump(dir, "toks.i32", C.toks, (size_t)done * MAXB); dump(dir, "tok_done.u64", C.tok_done, (size_t)done); dump(dir, "lsum.f32", C.lsum, (size_t)done * MAXB);
      dump(dir, "logits0.f32", C.logits0, (size_t)b * VOC); }
    if (mode != 1) { dump(dir, "loss.f32", C.loss, (size_t)sdone * jobs); dump(dir, "step_done.u64", C.step_done, (size_t)sdone); }
    FILE* f = fopen((dir + "/result.json").c_str(), "w"); if (!f) { fprintf(stderr, "cannot write result.json\n"); return 1; }
    fprintf(f, "{\"mode\": \"%s\", \"sched\": \"%s\", \"rate\": %.4f, \"nreq\": %d, \"req_tokens\": %d, \"b\": %d, \"dec_tokens_done\": %d, \"requests_done\": %d, "
               "\"req_p50_ms\": %.6f, \"req_p99_ms\": %.6f, \"tpot_p50_ms\": %.6f, \"tpot_p99_ms\": %.6f, \"train_steps_done\": %d, \"train_steps_window\": %d, "
               "\"window_s\": %.6f, \"train_steps_s\": %.6f, \"train_steps_per_gpu_hour\": %.1f, \"kernel_s\": %.6f, \"gpu\": \"%s\"}\n",
            mode_s.c_str(), sched_s.c_str(), rate, nreq, L, b, done, nfull, rp50, rp99, tp50, tp99, sdone, win_steps, win_s, sps, sps * 3600, ms / 1e3, prop.name);
    fclose(f);
  }
  return (mode != 0 && done != dtok) ? 1 : 0;
}
