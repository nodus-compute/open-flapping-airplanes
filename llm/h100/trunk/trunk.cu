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
#include "../mm/mm_decode.cuh"
#include <random>
#ifdef MW_GEMM_WG
#include "../gemm_sm90.cuh"
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

constexpr int D = 2560, NQ = 4096, NKV = 1024, NQV = NQ + NKV, NQKV = NQ + 2 * NKV, HD = 128, NKVH = 8, VOC = 151936, R = 8, R2 = 2 * R;
constexpr int PJ = 2 * R * D + NQV * R;            // per-job fp32 params: Aq[R][D] Av[R][D] Bq[NQ][R] Bv[NKV][R]
constexpr int TI = 9728;                            // Qwen3-4B MLP width (trunk up/down; real gate/up/down)
constexpr int MAXB = 8, MAXPOS = 4096, THREADS = 256, ROWS = 16;
#ifdef MW_GEMM_WG
constexpr int SMEM = gemm::sm90::smem_bytes() > gemm::smem_bytes() ? gemm::sm90::smem_bytes() : gemm::smem_bytes();
#else
constexpr int SMEM = gemm::smem_bytes();
#endif
//           // 64 KB
static_assert(4 * MAXPOS * 4 <= SMEM, "attention scores fit the GEMM smem");

enum Kind { T_BASEQ, T_BASEV, T_DOWN, T_UPG, T_DB, T_DU, T_DA, T_ADAM, T_LOSS, D_DOWN, D_QKV, D_ATT, D_O, D_HEAD, D_NEXT, T_TUP, T_TDN, T_TCV, T_REMB, T_RRMS, T_RQKN, T_RATT, T_RACT,
  T_RATTC, T_RQV, T_RCP, T_CE, T_RNB, T_RACTB, T_RADQ, T_RADV, T_GNN, D_REAL, T_RADQC, T_RADVC, T_PFCP };
struct Ph { int kind, ntasks, tilesM, reps, layer, f32, koff; gemm::Args ga; float *fx, *fx2; const bf16* w2;
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
  const Ctx* pf;   // --prefill-chunks: device copy of the prefill tenant's context (own activations, own queue in pf->tr)
  Job tr, dc; int mode, infer_sms, lend; unsigned* lent; const int* smid_rank; int* finished;
  // train
  int jobs, tk, nx; float scale, lr, b1, b2, eps, inv;
  const bf16 *X, *Wq, *Wv, *Tq, *Tv; bf16 *Yq, *Yv;
  const bf16* Remb; const int* rtok; float* rh; bf16 *rqkv, *ratt, *rgu, *ract, *rxn; int rseq; float rms_eps, rtheta;   // --trunk-ckpt real layers
  const int* rlab; const float *tinv, *lrj; float wd;   // --tail real: shifted labels per slot, 1/n_valid per slot, per-job AdamW lr, weight decay
  int L; const u64* arr; u64 *t0, *tmax; u64 max_ns;   // request decode: L tokens/request, Poisson arrivals (ns after t0); per-kind max train-task ns
  int lt; bf16 *Xt[2], *Ht, *junk;   // frozen shared trunk: lt layers, ping-pong outputs, MLP hidden, per-job-arm scratch
  float *P, *G, *M, *Vm, *U, *dO, *dU, *lpart, *loss; u64* step_done;
  // decode
  int b, ctx0; const bf16 *Wqkv, *Wo, *E; const float* adp;   // adp: one adapter, same layout as P
  float *h, *du, *q, *att, *lmax, *lsum_part, *lsum, *logits0; int *lidx, *toks; bf16 *kc, *vc; u64* tok_done;
  mmd::Model RM; int dres;   // --dec real: real-weight Qwen3-4B decode (mm_decode.cuh); --dec-reserve: blocks [0, dres) claim decode only
};

__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ld_vol(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ unsigned ld2(const bf16* p) { return *reinterpret_cast<const unsigned*>(p); }
__device__ __forceinline__ unsigned pk(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<unsigned*>(&v); }
__device__ __forceinline__ unsigned pk2(bf16 lo, bf16 hi) { __nv_bfloat162 v; v.x = lo; v.y = hi; return *reinterpret_cast<unsigned*>(&v); }
__device__ __forceinline__ void mma16816(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

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
__device__ __noinline__ void radq_tc(const Ctx& C, const Ph& d, int local, int tk) {   // T_RADQC (--attn-bwd tc), kept out of line so its mma fragments don't raise train_task's register pressure
  const int tid = threadIdx.x;                                            // tensor-core attention backward for dq (--attn-bwd tc): warp = 16 queries x 1 head (T_RATTC layout), mma.sync
      //   m16n8k16: S = Q K^T and dP = dO V^T, dS = P (dP - D) packed to bf16, dq += dS K over fixed key blocks; then RoPE^T + q-norm backward as T_RADQ
      const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3, h = local % 32, kvg = h / 4;
      const int q0 = (local / 32) * 128 + warp * 16, s0 = q0 - q0 % C.rseq;
      const bf16* QKV = d.ga.A; const bf16* GO = d.ga.aux; const bf16* OO = d.ga.B;
      const float sl2 = 0.08838834764831845f * 1.4426950408889634f;
      unsigned qa[8][4]; float Dt[2] = {0.f, 0.f}, L2[2];
      for (int ks = 0; ks < 8; ks++) {
        const bf16* r0 = QKV + (size_t)(q0 + g) * NQKV + h * HD + ks * 16 + 2 * c; const bf16* r1 = r0 + (size_t)8 * NQKV;
        qa[ks][0] = ld2(r0); qa[ks][1] = ld2(r1); qa[ks][2] = ld2(r0 + 8); qa[ks][3] = ld2(r1 + 8);
        for (int i = 0; i < 2; i++) { const size_t ro = (size_t)(q0 + g + 8 * i) * NQ + h * HD + ks * 16 + 2 * c;
          for (int e = 0; e < 2; e++) Dt[i] = fmaf(bf(GO[ro + 8 + e]), bf(OO[ro + 8 + e]), fmaf(bf(GO[ro + e]), bf(OO[ro + e]), Dt[i])); }
      }
      for (int i = 0; i < 2; i++) { Dt[i] += __shfl_xor_sync(0xffffffffu, Dt[i], 1); Dt[i] += __shfl_xor_sync(0xffffffffu, Dt[i], 2);
        L2[i] = d.fx[(size_t)(q0 + g + 8 * i) * 32 + h] * 1.4426950408889634f; }
      float dq[16][4];
      for (int n = 0; n < 16; n++) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
      for (int kb = s0; kb <= q0; kb += 16) {
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, dp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int nt = 0; nt < 2; nt++) { const bf16* kr = QKV + (size_t)(kb + nt * 8 + g) * NQKV + NQ + kvg * HD + 2 * c; const bf16* vr = kr + NKV;
          for (int ks = 0; ks < 8; ks++) {
            const bf16* r0 = GO + (size_t)(q0 + g) * NQ + h * HD + ks * 16 + 2 * c; const bf16* r1 = r0 + (size_t)8 * NQ;
            const unsigned ga[4] = {ld2(r0), ld2(r1), ld2(r0 + 8), ld2(r1 + 8)};
            mma16816(sc[nt], qa[ks], ld2(kr + ks * 16), ld2(kr + ks * 16 + 8)); mma16816(dp[nt], ga, ld2(vr + ks * 16), ld2(vr + ks * 16 + 8)); } }
        for (int nt = 0; nt < 2; nt++) for (int e = 0; e < 4; e++) { const int t = q0 + g + 8 * (e >> 1), p = kb + nt * 8 + 2 * c + (e & 1);
          sc[nt][e] = p <= t ? exp2f(sc[nt][e] * sl2 - L2[e >> 1]) * (dp[nt][e] - Dt[e >> 1]) : 0.f; }
        const unsigned pa[4] = {pk(sc[0][0], sc[0][1]), pk(sc[0][2], sc[0][3]), pk(sc[1][0], sc[1][1]), pk(sc[1][2], sc[1][3])};
        const bf16* kr = QKV + (size_t)(kb + 2 * c) * NQKV + NQ + kvg * HD + g;
        for (int n = 0; n < 16; n++) { const bf16* k0 = kr + n * 8;
          mma16816(dq[n], pa, pk2(k0[0], k0[NQKV]), pk2(k0[(size_t)8 * NQKV], k0[(size_t)9 * NQKV])); }
      }
      for (int i = 0; i < 2; i++) { const int t = q0 + g + 8 * i, pos = t % C.rseq; float v[16][2];
        for (int n = 0; n < 16; n++) for (int e = 0; e < 2; e++) v[n][e] = dq[n][2 * i + e] * 0.08838834764831845f;
        for (int n = 0; n < 8; n++) for (int e = 0; e < 2; e++) { const int dd = n * 8 + 2 * c + e; const float ang = (float)pos * powf(C.rtheta, -(float)(2 * dd) / (float)HD);
          const float cs = cosf(ang), sn = sinf(ang), a0 = v[n][e], a1 = v[n + 8][e]; v[n][e] = a0 * cs + a1 * sn; v[n + 8][e] = a1 * cs - a0 * sn; }
        const bf16* xp = (const bf16*)d.ga.C + (size_t)t * NQKV + h * HD; float ss = 0.f, sg = 0.f;
        for (int n = 0; n < 16; n++) for (int e = 0; e < 2; e++) { const int dd = n * 8 + 2 * c + e; const float x = bf(xp[dd]); ss = fmaf(x, x, ss); sg = fmaf(v[n][e] * bf(d.w2[dd]), x, sg); }
        for (int o2 = 1; o2 < 4; o2 <<= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o2); sg += __shfl_xor_sync(0xffffffffu, sg, o2); }
        const float inv = rsqrtf(ss / HD + C.rms_eps), c3 = inv * inv * inv * sg / HD;
        float* out = C.dO + ((size_t)d.layer * tk + t) * NQV + h * HD;
        for (int n = 0; n < 16; n++) for (int e = 0; e < 2; e++) { const int dd = n * 8 + 2 * c + e; out[dd] = inv * bf(d.w2[dd]) * v[n][e] - bf(xp[dd]) * c3; }
      }
      return;
}

__device__ __noinline__ void radv_tc(const Ctx& C, const Ph& d, int local, int tk) {   // T_RADVC (--attn-bwd tc), kept out of line so its mma fragments don't raise train_task's register pressure
  const int tid = threadIdx.x;                                            // tensor-core attention backward for dv: warp = 16 keys x 1 kv head; per q head of the group (fixed order):
      //   S^T = K Q^T, P^T = exp(S^T - lse) masked causal, dv += P^T dO over later query blocks of the segment (mma.sync m16n8k16)
      const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3, kvg = local % NKVH;
      const int p0 = (local / NKVH) * 128 + warp * 16, s1 = p0 - p0 % C.rseq + C.rseq;
      const bf16* QKV = d.ga.A; const bf16* GO = d.ga.aux;
      const float sl2 = 0.08838834764831845f * 1.4426950408889634f;
      unsigned ka[8][4];
      for (int ks = 0; ks < 8; ks++) { const bf16* r0 = QKV + (size_t)(p0 + g) * NQKV + NQ + kvg * HD + ks * 16 + 2 * c; const bf16* r1 = r0 + (size_t)8 * NQKV;
        ka[ks][0] = ld2(r0); ka[ks][1] = ld2(r1); ka[ks][2] = ld2(r0 + 8); ka[ks][3] = ld2(r1 + 8); }
      float dv[16][4];
      for (int n = 0; n < 16; n++) dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;
      for (int hh = 0; hh < 4; hh++) { const int h = kvg * 4 + hh;
        for (int tb = p0; tb < s1; tb += 16) {
          float st[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          for (int nt = 0; nt < 2; nt++) { const bf16* qr = QKV + (size_t)(tb + nt * 8 + g) * NQKV + h * HD + 2 * c;
            for (int ks = 0; ks < 8; ks++) mma16816(st[nt], ka[ks], ld2(qr + ks * 16), ld2(qr + ks * 16 + 8)); }
          for (int nt = 0; nt < 2; nt++) for (int e = 0; e < 4; e++) { const int p = p0 + g + 8 * (e >> 1), t = tb + nt * 8 + 2 * c + (e & 1);
            st[nt][e] = t >= p ? exp2f(st[nt][e] * sl2 - d.fx[(size_t)t * 32 + h] * 1.4426950408889634f) : 0.f; }
          const unsigned pa[4] = {pk(st[0][0], st[0][1]), pk(st[0][2], st[0][3]), pk(st[1][0], st[1][1]), pk(st[1][2], st[1][3])};
          const bf16* gr = GO + (size_t)(tb + 2 * c) * NQ + h * HD + g;
          for (int n = 0; n < 16; n++) { const bf16* g0 = gr + n * 8;
            mma16816(dv[n], pa, pk2(g0[0], g0[NQ]), pk2(g0[(size_t)8 * NQ], g0[(size_t)9 * NQ])); }
        } }
      for (int i = 0; i < 2; i++) { float* out = C.dO + ((size_t)d.layer * tk + p0 + g + 8 * i) * NQV + NQ + kvg * HD;
        for (int n = 0; n < 16; n++) *reinterpret_cast<float2*>(out + n * 8 + 2 * c) = make_float2(dv[n][2 * i], dv[n][2 * i + 1]); }
      (void)lane; return;
}

__device__ __noinline__ void train_task(const Ctx& C, const Ph& d, int local, int step, char* smem) {
  float* red = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, tk = C.tk, rb_per_job = tk / ROWS;
  const bf16* X = C.lt ? C.Xt[C.lt % 2] : C.X + (size_t)(step % C.nx) * tk * D;
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
    case T_REMB: {                                             // real trunk: h = embed(tokens of this step's slot), fp32 residual stream
      const int* tok = C.rtok + (size_t)(step % C.nx) * tk; const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid; C.rh[k] = bf(C.Remb[(size_t)tok[k / D] * D + k % D]); }
      break;
    }
    case T_RRMS: {                                             // RMSNorm(h) * w -> bf16 (one warp per token row)
      const int warp = tid >> 5, lane = tid & 31, t = local * 8 + warp; const float* x = (d.ga.A ? (const float*)d.ga.A : C.rh) + (size_t)t * D;
      const bf16* w = d.ga.B; bf16* o = (bf16*)d.ga.C + (size_t)t * D; float ss = 0.f;
      for (int k = lane; k < D; k += 32) ss = fmaf(x[k], x[k], ss);
      for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      const float inv = rsqrtf(ss / D + C.rms_eps);
      for (int k = lane; k < D; k += 32) o[k] = __float2bfloat16(x[k] * inv * bf(w[k]));
      break;
    }
    case T_RQKN: {                                             // per-head q/k RMSNorm + rotate-half RoPE in place (warp per (token, head))
      const int warp = tid >> 5, lane = tid & 31, w8 = local * 8 + warp, t = w8 / 40, h = w8 % 40, pos = t % C.rseq;
      bf16* q = (d.ga.C ? (bf16*)d.ga.C : C.rqkv) + (size_t)t * NQKV + h * HD; const bf16* nw = d.ga.B + (h < 32 ? 0 : HD);
      float x[4], ss = 0.f;
      for (int j = 0; j < 4; j++) { x[j] = bf(q[lane + 32 * j]); ss = fmaf(x[j], x[j], ss); }
      for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      const float inv = rsqrtf(ss / HD + C.rms_eps);
      for (int j = 0; j < 4; j++) x[j] = x[j] * inv * bf(nw[lane + 32 * j]);
      for (int j = 0; j < 2; j++) { const int dd = lane + 32 * j; const float ang = (float)pos * powf(C.rtheta, -(float)(2 * dd) / (float)HD);
        const float c = cosf(ang), sn = sinf(ang), a0 = x[j], a1 = x[j + 2]; x[j] = a0 * c - a1 * sn; x[j + 2] = a1 * c + a0 * sn; }
      for (int j = 0; j < 4; j++) q[lane + 32 * j] = __float2bfloat16(x[j]);
      break;
    }
    case T_RATT: {                                             // causal GQA attention within each seq segment: warp per (query, head), online softmax, fixed key order
      const int warp = tid >> 5, lane = tid & 31, w8 = local * 8 + warp, t = w8 / 32, h = w8 % 32, g = h / 4, s0 = t - t % C.rseq;
      const bf16* QKV = d.ga.A ? d.ga.A : C.rqkv; const bf16* q = QKV + (size_t)t * NQKV + h * HD; float qv[4], acc[4] = {0.f, 0.f, 0.f, 0.f}, m = -1e30f, l = 0.f;
      for (int j = 0; j < 4; j++) qv[j] = bf(q[lane + 32 * j]) * 0.08838834764831845f;
      for (int p = s0; p <= t; p++) {
        const bf16* k = QKV + (size_t)p * NQKV + NQ + g * HD; float sc = 0.f;
        for (int j = 0; j < 4; j++) sc = fmaf(qv[j], bf(k[lane + 32 * j]), sc);
        for (int o2 = 16; o2 > 0; o2 >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o2);
        const float mn = fmaxf(m, sc), cr = __expf(m - mn), e = __expf(sc - mn); l = l * cr + e; m = mn;
        const bf16* v = k + NKV;
        for (int j = 0; j < 4; j++) acc[j] = acc[j] * cr + e * bf(v[lane + 32 * j]);
      }
      bf16* o = (d.ga.C ? (bf16*)d.ga.C : C.ratt) + (size_t)t * NQ + h * HD;
      for (int j = 0; j < 4; j++) o[lane + 32 * j] = __float2bfloat16(acc[j] / l);
      if (d.fx && lane == 0) d.fx[(size_t)t * 32 + h] = m + logf(l);   // natural-log LSE of the scaled scores (tail backward)
      break;
    }
    case T_RACT: {                                             // SwiGLU: act = silu(gate) * up from 64-row interleaved gate/up blocks
      const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid, t = k / TI; const int c = (int)(k % TI);
        const bf16* r = (d.ga.A ? d.ga.A : C.rgu) + t * 2 * TI + (c / 64) * 128 + c % 64; const float gg = bf(r[0]), uu = bf(r[64]);
        (d.ga.C ? (bf16*)d.ga.C : C.ract)[k] = __float2bfloat16(gg / (1.f + __expf(-gg)) * uu); }
      break;
    }
    case T_RATTC: {                                            // tensor-core causal GQA attention: warp = 16 queries x 1 head, mma.sync m16n8k16, online softmax (log2), fixed key-block order
      const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3, h = local % 32, kvg = h / 4;
      const int q0 = (local / 32) * 128 + warp * 16, s0 = q0 - q0 % C.rseq;
      const bf16* QKV = d.ga.A ? d.ga.A : C.rqkv;
      unsigned qa[8][4];
      for (int ks = 0; ks < 8; ks++) {
        const bf16* r0 = QKV + (size_t)(q0 + g) * NQKV + h * HD + ks * 16 + 2 * c; const bf16* r1 = r0 + (size_t)8 * NQKV;
        qa[ks][0] = ld2(r0); qa[ks][1] = ld2(r1); qa[ks][2] = ld2(r0 + 8); qa[ks][3] = ld2(r1 + 8);
      }
      float o[16][4], m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
      for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
      const float sl2 = 0.08838834764831845f * 1.4426950408889634f;
      for (int kb = s0; kb <= q0; kb += 16) {
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int nt = 0; nt < 2; nt++) { const bf16* kr = QKV + (size_t)(kb + nt * 8 + g) * NQKV + NQ + kvg * HD + 2 * c;
          for (int ks = 0; ks < 8; ks++) mma16816(s[nt], qa[ks], ld2(kr + ks * 16), ld2(kr + ks * 16 + 8)); }
        float cr[2];
        for (int i = 0; i < 2; i++) {                          // i = 0: row g (regs 0,1); i = 1: row g + 8 (regs 2,3)
          const int t = q0 + g + 8 * i; float mm = m[i];
          for (int nt = 0; nt < 2; nt++) for (int e = 0; e < 2; e++) { const int p = kb + nt * 8 + 2 * c + e; float& v = s[nt][2 * i + e]; v = p <= t ? v * sl2 : -1e30f; mm = fmaxf(mm, v); }
          mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, 1)); mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
          cr[i] = exp2f(m[i] - mm); m[i] = mm; l[i] *= cr[i];
        }
        float ps[2] = {0.f, 0.f};
        for (int nt = 0; nt < 2; nt++) for (int e = 0; e < 4; e++) { const float pe = exp2f(s[nt][e] - m[e >> 1]); s[nt][e] = pe; ps[e >> 1] += pe; }
        for (int i = 0; i < 2; i++) { ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 1); ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 2); l[i] += ps[i]; }
        for (int n = 0; n < 16; n++) { o[n][0] *= cr[0]; o[n][1] *= cr[0]; o[n][2] *= cr[1]; o[n][3] *= cr[1]; }
        const unsigned pa[4] = {pk(s[0][0], s[0][1]), pk(s[0][2], s[0][3]), pk(s[1][0], s[1][1]), pk(s[1][2], s[1][3])};
        const bf16* vr = QKV + (size_t)(kb + 2 * c) * NQKV + NQ + NKV + kvg * HD + g;
        for (int n = 0; n < 16; n++) { const bf16* v0 = vr + n * 8;
          mma16816(o[n], pa, pk2(v0[0], v0[NQKV]), pk2(v0[(size_t)8 * NQKV], v0[(size_t)9 * NQKV])); }
      }
      bf16* out = d.ga.C ? (bf16*)d.ga.C : C.ratt;
      for (int i = 0; i < 2; i++) { const int t = q0 + g + 8 * i; const float il = 1.f / l[i];
        for (int n = 0; n < 16; n++) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * NQ + h * HD + n * 8 + 2 * c) = __floats2bfloat162_rn(o[n][2 * i] * il, o[n][2 * i + 1] * il);
        if (d.fx && c == 0) d.fx[(size_t)t * 32 + h] = (m[i] + log2f(l[i])) * 0.6931471805599453f; }
      break;
    }
    case T_RQV: {                                              // real tail, job d.layer: q/k/v = shared base + s*U.B^T on q and v (k = base); pre-norm copy (C) + working copy (aux)
      const int j = d.layer; const float* Bq = C.P + (size_t)j * PJ + 2 * R * D; const float* Bv = Bq + (size_t)NQ * R;
      const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid, t = k / NQKV; const int o = (int)(k % NQKV);
        float y = bf(d.ga.A[k]);
        if (o < NQ || o >= NQ + NKV) { const float* u = C.U + ((size_t)j * tk + t) * R2 + (o < NQ ? 0 : R);
          const float* B = o < NQ ? Bq + (size_t)o * R : Bv + (size_t)(o - NQ - NKV) * R; float s = 0.f;
          for (int r = 0; r < R; r++) s = fmaf(u[r], B[r], s);
          y = fmaf(C.scale, s, y); }
        const bf16 v = __float2bfloat16(y); ((bf16*)d.ga.C)[k] = v; d.ga.aux[k] = v; }
      break;
    }
    case T_RCP: {                                              // fp32 copy (residual stream fork for one job's top layer)
      const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid; ((float*)d.ga.C)[k] = d.fx[k]; }
      break;
    }
    case T_CE: {                                               // full-vocab CE per row: lse, loss, dlogits = (softmax - onehot) / n_valid (bf16); 16 rows per task, warp per 2 rows
      const int warp = tid >> 5, lane = tid & 31; const int* lab = C.rlab + (size_t)(step % C.nx) * tk; const float inv = C.tinv[step % C.nx];
      __syncthreads();
      for (int rr = 0; rr < 2; rr++) { const int t = local * 16 + warp * 2 + rr; const float* x = d.fx + (size_t)t * VOC; const int y = lab[t];
        bf16* gr = (bf16*)d.ga.C + (size_t)t * VOC;
        if (y < 0) { for (int v = lane; v < VOC; v += 32) gr[v] = __float2bfloat16(0.f); if (lane == 0) red[warp * 2 + rr] = 0.f; continue; }
        float mm = -1e30f; for (int v = lane; v < VOC; v += 32) mm = fmaxf(mm, x[v]);
        for (int o2 = 16; o2 > 0; o2 >>= 1) mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, o2));
        float se = 0.f; for (int v = lane; v < VOC; v += 32) se += __expf(x[v] - mm);
        for (int o2 = 16; o2 > 0; o2 >>= 1) se += __shfl_xor_sync(0xffffffffu, se, o2);
        const float lse = mm + logf(se);
        for (int v = lane; v < VOC; v += 32) gr[v] = __float2bfloat16((__expf(x[v] - lse) - (v == y ? 1.f : 0.f)) * inv);
        if (lane == 0) red[warp * 2 + rr] = lse - x[y];
      }
      __syncthreads();
      if (tid == 0) { float s = 0.f; for (int i = 0; i < 16; i++) s += red[i]; C.lpart[(size_t)d.layer * rb_per_job + local] = s; }
      break;
    }
    case T_RNB: {                                              // RMSNorm backward (warp per row): dx = inv*w*dy - x*inv^3*sum(w*dy*x)/D (+ residual grad fx2); fp32 (C) + bf16 (aux)
      const int warp = tid >> 5, lane = tid & 31, t = local * 8 + warp; const float* x = (const float*)d.ga.A + (size_t)t * D; const float* dy = d.fx + (size_t)t * D;
      float ss = 0.f, sg = 0.f;
      for (int k = lane; k < D; k += 32) { ss = fmaf(x[k], x[k], ss); sg = fmaf(bf(d.ga.B[k]) * dy[k], x[k], sg); }
      for (int o2 = 16; o2 > 0; o2 >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o2); sg += __shfl_xor_sync(0xffffffffu, sg, o2); }
      const float inv = rsqrtf(ss / D + C.rms_eps), c3 = inv * inv * inv * sg / D;
      float* o = (float*)d.ga.C + (size_t)t * D; bf16* ob = d.ga.aux + (size_t)t * D;
      for (int k = lane; k < D; k += 32) { float v = inv * bf(d.ga.B[k]) * dy[k] - x[k] * c3; if (d.fx2) v += d.fx2[(size_t)t * D + k]; o[k] = v; ob[k] = __float2bfloat16(v); }
      break;
    }
    case T_RACTB: {                                            // SwiGLU backward into the 64-row interleaved gate/up layout
      const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid, t = k / TI; const int c = (int)(k % TI);
        const size_t r = t * 2 * TI + (c / 64) * 128 + c % 64; const float gg = bf(d.ga.A[r]), uu = bf(d.ga.A[r + 64]), da = bf(d.ga.B[k]);
        const float sg = 1.f / (1.f + __expf(-gg));
        ((bf16*)d.ga.C)[r] = __float2bfloat16(da * uu * sg * (1.f + gg * (1.f - sg))); ((bf16*)d.ga.C)[r + 64] = __float2bfloat16(da * gg * sg); }
      break;
    }
    case T_RADQ: {                                             // attention backward for dq (warp per (query, head)), then RoPE^T and q-norm backward -> dO[j][t][q cols]
      const int warp = tid >> 5, lane = tid & 31, w8 = local * 8 + warp, t = w8 / 32, h = w8 % 32, g = h / 4, s0 = t - t % C.rseq, pos = t % C.rseq;
      const bf16* QKV = d.ga.A; const bf16* q = QKV + (size_t)t * NQKV + h * HD; const bf16* oo = d.ga.B + (size_t)t * NQ + h * HD; const bf16* go = d.ga.aux + (size_t)t * NQ + h * HD;
      const float sc = 0.08838834764831845f, lse = d.fx[(size_t)t * 32 + h];
      float qv[4], gv[4], dq[4] = {0.f, 0.f, 0.f, 0.f}, Dt = 0.f;
      for (int i = 0; i < 4; i++) { qv[i] = bf(q[lane + 32 * i]) * sc; gv[i] = bf(go[lane + 32 * i]); Dt = fmaf(gv[i], bf(oo[lane + 32 * i]), Dt); }
      for (int o2 = 16; o2 > 0; o2 >>= 1) Dt += __shfl_xor_sync(0xffffffffu, Dt, o2);
      for (int p = s0; p <= t; p++) {
        const bf16* k = QKV + (size_t)p * NQKV + NQ + g * HD; const bf16* v = k + NKV; float s = 0.f, dp = 0.f;
        for (int i = 0; i < 4; i++) { s = fmaf(qv[i], bf(k[lane + 32 * i]), s); dp = fmaf(gv[i], bf(v[lane + 32 * i]), dp); }
        for (int o2 = 16; o2 > 0; o2 >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o2); dp += __shfl_xor_sync(0xffffffffu, dp, o2); }
        const float ds = __expf(s - lse) * (dp - Dt);
        for (int i = 0; i < 4; i++) dq[i] = fmaf(ds, bf(k[lane + 32 * i]), dq[i]);
      }
      for (int i = 0; i < 4; i++) dq[i] *= sc;
      for (int i = 0; i < 2; i++) { const int dd = lane + 32 * i; const float ang = (float)pos * powf(C.rtheta, -(float)(2 * dd) / (float)HD);
        const float cs = cosf(ang), sn = sinf(ang), a0 = dq[i], a1 = dq[i + 2]; dq[i] = a0 * cs + a1 * sn; dq[i + 2] = a1 * cs - a0 * sn; }
      const bf16* xp = (const bf16*)d.ga.C + (size_t)t * NQKV + h * HD; float x[4], ss = 0.f, sg = 0.f;
      for (int i = 0; i < 4; i++) { x[i] = bf(xp[lane + 32 * i]); ss = fmaf(x[i], x[i], ss); sg = fmaf(dq[i] * bf(d.w2[lane + 32 * i]), x[i], sg); }
      for (int o2 = 16; o2 > 0; o2 >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o2); sg += __shfl_xor_sync(0xffffffffu, sg, o2); }
      const float inv = rsqrtf(ss / HD + C.rms_eps), c3 = inv * inv * inv * sg / HD;
      float* out = C.dO + ((size_t)d.layer * tk + t) * NQV + h * HD;
      for (int i = 0; i < 4; i++) out[lane + 32 * i] = inv * bf(d.w2[lane + 32 * i]) * dq[i] - x[i] * c3;
      break;
    }
    case T_RADV: {                                             // attention backward for dv (warp per (key, kv head)): later queries of the group's 4 heads -> dO[j][p][v cols]
      const int warp = tid >> 5, lane = tid & 31, w8 = local * 8 + warp, p = w8 / NKVH, g = w8 % NKVH, s1 = p - p % C.rseq + C.rseq;
      const bf16* QKV = d.ga.A; const bf16* k = QKV + (size_t)p * NQKV + NQ + g * HD; float kv[4], acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int i = 0; i < 4; i++) kv[i] = bf(k[lane + 32 * i]);
      for (int t = p; t < s1; t++) for (int hh = 0; hh < 4; hh++) { const int h = g * 4 + hh; const bf16* q = QKV + (size_t)t * NQKV + h * HD; float s = 0.f;
        for (int i = 0; i < 4; i++) s = fmaf(bf(q[lane + 32 * i]) * 0.08838834764831845f, kv[i], s);
        for (int o2 = 16; o2 > 0; o2 >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o2);
        const float pe = __expf(s - d.fx[(size_t)t * 32 + h]); const bf16* go = d.ga.aux + (size_t)t * NQ + h * HD;
        for (int i = 0; i < 4; i++) acc[i] = fmaf(pe, bf(go[lane + 32 * i]), acc[i]); }
      float* out = C.dO + ((size_t)d.layer * tk + p) * NQV + NQ + g * HD;
      for (int i = 0; i < 4; i++) out[lane + 32 * i] = acc[i];
      break;
    }
    case T_RADQC: radq_tc(C, d, local, tk); break;
    case T_RADVC: radv_tc(C, d, local, tk); break;
    case T_GNN: {                                              // real-tail backward GEMM C = A.B with B stored [K][N] (mma.sync tile)
      gemm::Args a = d.ga;
      if (d.f32) gemm::gemm_tile<gemm::NN, gemm::EPI_F32>(a, local % d.tilesM, local / d.tilesM, smem);
      else gemm::gemm_tile<gemm::NN, gemm::EPI_BF16>(a, local % d.tilesM, local / d.tilesM, smem);
      break;
    }
    case T_TUP: case T_TDN: {                                  // frozen trunk GEMM (K chunk d.koff if split); reps > 1 = per-job arm (rep > 0 writes scratch)
      const int base = d.ntasks / d.reps, rep = local / base, l2 = local % base, x0 = d.kind == T_TUP && !d.layer;
      const bf16* A = x0 ? C.X + (size_t)(step % C.nx) * tk * D + d.koff : (const bf16*)d.ga.A;
#ifdef MW_GEMM_WG
      gemm::sm90::ArgsWG a; static_cast<gemm::Args&>(a) = d.ga; a.A = A; a.tmA = d.tmAx[x0 ? step % C.nx : 0]; a.tmB = d.tmB;
      if (rep) a.C = C.junk;
      if (d.f32) gemm::sm90::tile<gemm::NT, gemm::EPI_F32>(a, l2 % d.tilesM, l2 / d.tilesM, smem);
      else gemm::sm90::tile<gemm::NT, gemm::EPI_BF16>(a, l2 % d.tilesM, l2 / d.tilesM, smem);
#else
      gemm::Args a = d.ga; a.A = A; if (rep) a.C = C.junk;
      if (d.f32) gemm::gemm_tile<gemm::NT, gemm::EPI_F32>(a, l2 % d.tilesM, l2 / d.tilesM, smem);
      else gemm::gemm_tile<gemm::NT, gemm::EPI_BF16>(a, l2 % d.tilesM, l2 / d.tilesM, smem);
#endif
      break;
    }
    case T_PFCP: {                                             // prefill tenant: final-norm hidden of this chunk -> output slot step
      const bf16* src = d.ga.A; bf16* dst = (bf16*)d.ga.C + (size_t)step * tk * D; const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid; dst[k] = src[k]; }
      break;
    }
    case T_TCV: {                                              // split-K trunk: fp32 accumulator -> bf16 activations
      const float* src = (const float*)d.ga.A; bf16* dst = (bf16*)d.ga.C; const size_t i0 = (size_t)local * THREADS * 8;
      for (int i = 0; i < 8; i++) { const size_t k = i0 + (size_t)i * THREADS + tid; dst[k] = __float2bfloat16(src[k]); }
      break;
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
        const float lr = C.lrj ? C.lrj[j] : C.lr; if (C.wd != 0.f) C.P[i] *= 1.f - lr * C.wd;   // torch AdamW: decoupled decay first
        C.P[i] -= lr * (m / bc1) / (sqrtf(v / bc2) + C.eps);
      }
      break;
    }
    case T_LOSS: {
      if (tid < C.jobs) { float s = 0.f; for (int i = 0; i < rb_per_job; i++) s += C.lpart[(size_t)tid * rb_per_job + i]; C.loss[(size_t)step * C.jobs + tid] = s * (C.tinv ? C.tinv[step % C.nx] : C.inv); }
      break;
    }
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------ decode tasks (one token step)
__device__ __noinline__ void decode_task(const Ctx& C, const Ph& d, int local, int step, char* smem) {
  if (d.kind == D_REAL) { mmd::Ph p; p.kind = d.koff; p.ntasks = d.ntasks; p.layer = d.layer; p.norm = d.f32; mmd::decode_task(C.RM, p, local, step, smem); return; }
  float* sm = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, b = C.b, pos = C.ctx0 + (C.L ? step % C.L : step);
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

// request decode: a request's first token is released only at its arrival time
__device__ __forceinline__ bool arrived(const Ctx& C, int dec, int step, int ph) {
  return !(dec && C.L && ph == 0 && step % C.L == 0 && gtimer() < *reinterpret_cast<const volatile u64*>(C.t0) + C.arr[step / C.L]);
}
__device__ __forceinline__ void finish_dec(const Ctx& C, const Job& J, int ph, int step) {
  const int old = atomicAdd(J.done + ph, 1);
  if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) { C.tok_done[step] = gtimer();
    if (C.L && step == J.limit - 1) { __threadfence(); *reinterpret_cast<volatile int*>(C.finished) = 1; } }   // last request done: stop training
}
// --sm-split dynamic: non-blocking claim of a queue head; 1 = ran, 0 = head not ready, 2 = queue done
__device__ int try_claim(const Ctx& C, const Job& J, int dec, char* smem) {
  __shared__ volatile int s_c[4];
  if (threadIdx.x == 0) {
    int st = 0, step = 0, ph = 0, local = 0;
    const unsigned g = *reinterpret_cast<const volatile unsigned*>(J.ticket);
    if (g >= (unsigned)J.limit * (unsigned)J.tps || ld_vol(C.finished)) st = 2;
    else {
      step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
      while (r >= J.prefix[ph + 1]) ph++;
      local = r - J.prefix[ph];
      bool ready = arrived(C, dec, step, ph);
      for (int k = 0; k < J.dep_cnt[ph] && ready; k++) { const int2 e = J.deps[J.dep_off[ph] + k]; const int need = J.ntasks[e.x] * (step + 1 + e.y);
        if (need > 0 && ld_acq(J.done + e.x) < need) ready = false; }
      if (ready && atomicCAS(J.ticket, g, g + 1) == g) st = 1;
    }
    s_c[0] = st; s_c[1] = step; s_c[2] = ph; s_c[3] = local;
  }
  __syncthreads();
  const int st = s_c[0], step = s_c[1], ph = s_c[2], local = s_c[3];
  __syncthreads();
  if (st != 1) return st;
  const u64 ts = gtimer();
  if (dec) decode_task(C, J.ph[ph], local, step, smem); else train_task(C, J.ph[ph], local, step, smem);
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) { if (dec) finish_dec(C, J, ph, step); else { finish_train(C, J, ph, step); atomicMax(C.tmax + J.ph[ph].kind, gtimer() - ts); } }
  __syncthreads();
  return 1;
}
// strict priority on every SM: a ready decode instruction first, then a prefill tile (--prefill-chunks), a trunk/LoRA tile only when neither is ready
__device__ void dyn_loop(const Ctx& C, char* smem) {
  for (;;) {
    const int a = try_claim(C, C.dc, 1, smem); if (a == 1) continue;
    const int p = !C.pf || (int)blockIdx.x < C.dres ? 2 : try_claim(*C.pf, C.pf->tr, 0, smem); if (p == 1) continue;
    const int b = (int)blockIdx.x < C.dres ? 2 : try_claim(C, C.tr, 0, smem); if (b == 1) continue;
    if (a == 2 && b == 2 && p == 2) break;
    if (threadIdx.x == 0) __nanosleep(64);
  }
}

// ------------------------------------------------------------------------------------------------ persistent loop (OFA fetch-add ticket claim)
__device__ void job_loop(const Ctx& C, const Job& J, int job_id, char* smem) {
  __shared__ volatile int s_sel[4];
  const int tid = threadIdx.x; int cph = 0, rstep = -1, rph = -1, pend = 0; unsigned pend_g = 0;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  for (;;) {
    if (tid == 0) {
      int res = 0, step = 0, ph = 0, local = 0;
      if ((job_id == 1 && C.mode == 2 && ld_vol(C.finished)) || (job_id == 0 && C.L && ld_vol(C.finished)) ||
          (job_id == 0 && C.max_ns && gtimer() - *reinterpret_cast<const volatile u64*>(C.t0) > C.max_ns)) res = -2;
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
              if (ready && arrived(C, job_id, step, ph)) { rstep = step; rph = ph; break; }
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
    const u64 ts = gtimer();
    if (job_id == 0) train_task(C, J.ph[ph], local, step, smem); else decode_task(C, J.ph[ph], local, step, smem);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      if (job_id == 0) { finish_train(C, J, ph, step); atomicMax(C.tmax + J.ph[ph].kind, gtimer() - ts); }
      else finish_dec(C, J, ph, step);
    }
  }
}

__global__ void __launch_bounds__(THREADS, 2) mw_kernel(const __grid_constant__ Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  unsigned hw; asm volatile("mov.u32 %0, %%smid;" : "=r"(hw));
  const int smid = C.smid_rank ? C.smid_rank[hw] : (int)hw;
  char* smem = reinterpret_cast<char*>(dsm);
  if (threadIdx.x == 0) atomicCAS(C.t0, 0ull, gtimer());
  __syncthreads();
  if (C.mode == 3) dyn_loop(C, smem);
  else if (C.mode == 0) job_loop(C, C.tr, 0, smem);
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
  int lt = 0, jb = 0; std::string trunk_s = "shared";
  int nreq = 64, Lr = 64, kc = 0, rseq = 512; std::string ckpt, attn_s = "cuda", tail_s = "mse", tdata, dec_s = "synth", abwd_s = "cuda"; int dres = 0, dlim = 0, dP = 0, npf = 0; std::string pfdata; double rate = 0, max_s = 0; std::string split_s = "static";
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--mode")) mode_s = argv[++i]; else if (is("--jobs")) jobs = atoi(argv[++i]); else if (is("--tokens")) tk = atoi(argv[++i]);
    else if (is("--steps")) steps = atoi(argv[++i]); else if (is("--b")) b = atoi(argv[++i]); else if (is("--ctx")) ctx0 = atoi(argv[++i]);
    else if (is("--dec-tokens")) dtok = atoi(argv[++i]); else if (is("--infer-sms")) infer_sms = atoi(argv[++i]); else if (is("--dump")) dir = argv[++i];
    else if (is("--sm-split")) split_s = argv[++i]; else if (is("--rate")) rate = atof(argv[++i]); else if (is("--nreq")) nreq = atoi(argv[++i]);
    else if (is("--req-tokens")) Lr = atoi(argv[++i]); else if (is("--trunk-kchunk")) kc = atoi(argv[++i]); else if (is("--trunk-ckpt")) ckpt = argv[++i]; else if (is("--seq")) rseq = atoi(argv[++i]); else if (is("--max-seconds")) max_s = atof(argv[++i]);
    else if (is("--trunk-layers")) lt = atoi(argv[++i]); else if (is("--trunk")) trunk_s = argv[++i]; else if (is("--job-base")) jb = atoi(argv[++i]);
    else if (is("--attn")) attn_s = argv[++i]; else if (is("--dec")) dec_s = argv[++i]; else if (is("--attn-bwd")) abwd_s = argv[++i]; else if (is("--dec-reserve")) dres = atoi(argv[++i]); else if (is("--tail")) tail_s = argv[++i]; else if (is("--tail-data")) tdata = argv[++i];
    else if (is("--prefill-chunks")) npf = atoi(argv[++i]); else if (is("--prefill-data")) pfdata = argv[++i];
    else if (!strcmp(argv[i], "--lend")) lend = 1; else if (is("--seed")) seed = atoll(argv[++i]); else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  if (rate > 0) dtok = nreq * Lr;
  const int mode = mode_s == "train" ? 0 : mode_s == "dec" ? 1 : mode_s == "co" ? 2 : mode_s == "pf" ? 4 : -1;
  const bool treal = tail_s == "real", atc = attn_s == "tc", dreal = dec_s == "real", atcb = abwd_s == "tc";
  if (npf < 0 || (mode == 4 && npf < 1) || (npf && (ckpt.empty() || pfdata.empty() || (mode != 4 && (mode != 2 || split_s != "dynamic"))))) {
    fprintf(stderr, "bad --prefill-chunks/--prefill-data: needs --trunk-ckpt, and --mode pf or --mode co --sm-split dynamic\n"); return 2; }
  if (abwd_s != "cuda" && !atcb) { fprintf(stderr, "bad --attn-bwd cuda|tc\n"); return 2; }
  if ((dec_s != "synth" && !dreal) || (dreal && (ckpt.empty() || rate > 0)) || dres < 0 || (dres && (mode != 2 || split_s != "dynamic"))) {
    fprintf(stderr, "bad --dec synth|real (real needs --trunk-ckpt, no --rate) / --dec-reserve (co + dynamic split only)\n"); return 2; }
  if ((attn_s != "cuda" && !atc) || (tail_s != "mse" && !treal) || (atc && ckpt.empty()) || (treal && (ckpt.empty() || tdata.empty() || (mode != 0 && !dreal) || rseq % 128))) {
    fprintf(stderr, "bad --attn/--tail: tc attention and the real tail need --trunk-ckpt; --tail real needs --tail-data and --mode train\n"); return 2; }
  if (treal) nx = steps;   // one real microbatch per step (training_prefix row order)
  if (lt < 0 || lt > 64 || jb < 0 || (trunk_s != "shared" && trunk_s != "perjob") || (split_s != "static" && split_s != "dynamic") || (rate > 0 && (nreq < 1 || Lr < 2)) || kc < 0 || (!ckpt.empty() && (lt < 1 || rseq < 1 || tk % rseq))) { fprintf(stderr, "bad --trunk-layers (0..64) / --trunk shared|perjob / --job-base\n"); return 2; }
  if (mode < 0 || jobs < 1 || jobs > THREADS || tk % 128 || b < 1 || b > MAXB || ctx0 < 1 || steps < 1 || (mode != 0 && ctx0 + (rate > 0 ? Lr : dtok) > MAXPOS)) {
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

  Ctx C; memset(&C, 0, sizeof C); C.mode = (mode == 2 && split_s == "dynamic") || mode == 4 ? 3 : mode; C.infer_sms = infer_sms; C.lend = mode == 2 && lend; C.lent = dalloc<unsigned>(1); C.smid_rank = d_rank; C.finished = dalloc<int>(1);
  if (dres >= nblocks) { fprintf(stderr, "--dec-reserve must be < %d blocks\n", nblocks); return 2; }
  C.dres = dres; C.t0 = dalloc<u64>(1); C.max_ns = (u64)(max_s * 1e9); C.tmax = dalloc<u64>(64); C.L = rate > 0 ? Lr : 0;
  if (rate > 0) { std::vector<u64> arr(nreq); std::mt19937_64 g(seed * 7 + 11); std::exponential_distribution<double> ex(rate); double t = 0.02;
    for (int k = 0; k < nreq; k++) { t += ex(g); arr[k] = (u64)(t * 1e9); }
    u64* a = dalloc<u64>(nreq); CK(cudaMemcpy(a, arr.data(), nreq * 8, cudaMemcpyHostToDevice)); C.arr = a; }
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
  static_assert(TI == 9728, "Qwen3-4B MLP width");
  // --trunk-ckpt DIR (h100/mm_prep.py layout): real Qwen3-4B layers [0, lt) as the frozen trunk; the tail's base q/v = layer lt's real q/v
  size_t lst = 0; const bf16* RW = nullptr; int LmA = 0, VA = 0;
  if (!ckpt.empty()) {
    FILE* f = fopen((ckpt + "/meta.txt").c_str(), "r"); int H, Lm, NH, NKVm, I, V, P; float eps, theta;
    if (!f || fscanf(f, "%d %d %d %d %d %d %d %f %f", &H, &Lm, &NH, &NKVm, &I, &V, &P, &eps, &theta) != 9) { fprintf(stderr, "cannot read %s/meta.txt\n", ckpt.c_str()); return 2; }
    fclose(f);
    if (H != D || NH * HD != NQ || NKVm * HD != NKV || I != TI || lt >= Lm) { fprintf(stderr, "checkpoint shape mismatch (need Qwen3-4B, trunk-layers < %d)\n", Lm); return 2; }
    lst = (size_t)D + (size_t)NQKV * D + 2 * HD + (size_t)D * NQ + D + (size_t)2 * TI * D + (size_t)D * TI;
    const size_t total = (size_t)V * D + D + (size_t)(dreal || npf ? Lm : lt + 1) * lst; bf16* w = dalloc<bf16>(total);
    FILE* wf = fopen((ckpt + "/weights.bf16").c_str(), "rb"); if (!wf) { fprintf(stderr, "cannot open %s/weights.bf16\n", ckpt.c_str()); return 2; }
    std::vector<bf16> buf(1 << 26); size_t off = 0;
    while (off < total) { const size_t n = std::min(buf.size(), total - off); if (fread(buf.data(), 2, n, wf) != n) { fprintf(stderr, "short weights file\n"); return 2; }
      CK(cudaMemcpy(w + off, buf.data(), n * 2, cudaMemcpyHostToDevice)); off += n; }
    fclose(wf);
    std::vector<int> tok((size_t)nx * tk); FILE* tf = fopen((treal ? tdata + "/tail_tokens.i32" : ckpt + "/train_tokens.i32").c_str(), "rb");
    if (!tf || fread(tok.data(), 4, tok.size(), tf) != tok.size()) { fprintf(stderr, "cannot read %zu tokens from %s/train_tokens.i32\n", tok.size(), ckpt.c_str()); return 2; }
    fclose(tf); for (int t : tok) if (t < 0 || t >= V) { fprintf(stderr, "token id out of range\n"); return 2; }
    int* dt = dalloc<int>(tok.size()); CK(cudaMemcpy(dt, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
    C.Remb = w; RW = w + (size_t)V * D + D; LmA = Lm; VA = V; C.rtok = dt; C.rseq = rseq; C.rms_eps = eps; C.rtheta = theta;
    C.rh = dalloc<float>((size_t)tk * D); C.rxn = dalloc<bf16>((size_t)tk * D); C.rqkv = dalloc<bf16>((size_t)tk * NQKV);
    C.ratt = dalloc<bf16>((size_t)tk * NQ); C.rgu = dalloc<bf16>((size_t)tk * 2 * TI); C.ract = dalloc<bf16>((size_t)tk * TI);
    C.Wq = RW + (size_t)lt * lst + D; C.Wv = C.Wq + (size_t)(NQ + NKV) * D;
    printf("trunk-ckpt %s: real layers 0..%d, tail q/v from layer %d, seq %d, eps %g theta %g\n", ckpt.c_str(), lt - 1, lt, rseq, eps, theta);
    if (dreal) {   // real decode on the same resident weights (all Lm layers): mm.cu's model state, prompt.i32 from mm_prep.py
      mmd::Model& M = C.RM;
      if (V != VOC || P + dtok > mmd::SPOS || NH / NKVm > mmd::MAXHPK) { fprintf(stderr, "--dec real: need vocab %d and prompt + dec-tokens <= %d\n", VOC, mmd::SPOS); return 2; }
      M.H = D; M.L = Lm; M.NH = NH; M.NKV = NKVm; M.I = TI; M.V = V; M.NQ = NQ; M.NK = NKV; M.nqkv = NQKV; M.b = b; M.P = P; M.maxpos = P + dtok; M.eps = eps; M.theta = theta;
      M.emb = w; M.fnorm = w + (size_t)V * D; M.w = RW; M.lst = lst; M.o_in = 0; M.o_qkv = D; M.o_qn = M.o_qkv + (size_t)NQKV * D; M.o_kn = M.o_qn + HD;
      M.o_o = M.o_kn + HD; M.o_post = M.o_o + (size_t)D * NQ; M.o_gu = M.o_post + D; M.o_d = M.o_gu + (size_t)2 * TI * D;
      std::vector<int> pr((size_t)mmd::MAXB * P); FILE* pf = fopen((ckpt + "/prompt.i32").c_str(), "rb");
      if (!pf || fread(pr.data(), 4, pr.size(), pf) != pr.size()) { fprintf(stderr, "cannot read %s/prompt.i32\n", ckpt.c_str()); return 2; }
      fclose(pf); { int* dp = dalloc<int>(pr.size()); CK(cudaMemcpy(dp, pr.data(), pr.size() * 4, cudaMemcpyHostToDevice)); M.prompt = dp; }
      M.h = dalloc<float>((size_t)b * D); M.xn = dalloc<float>((size_t)b * D); M.q = dalloc<float>((size_t)b * NQ); M.kraw = dalloc<float>((size_t)b * NKV);
      M.att = dalloc<float>((size_t)b * NQ); M.act = dalloc<float>((size_t)b * TI);
      M.lmax = dalloc<float>((size_t)b * (VOC / 128)); M.lidx = dalloc<int>((size_t)b * (VOC / 128)); M.lsum_part = dalloc<float>((size_t)b * (VOC / 128)); M.logitsP = dalloc<float>((size_t)b * VOC);
      dlim = P + dtok - 1; dP = P; M.toks = dalloc<int>((size_t)dlim * mmd::MAXB); M.tok_done = dalloc<u64>(dlim);
      M.kc = dalloc<bf16>((size_t)Lm * b * M.maxpos * NKV); M.vc = dalloc<bf16>((size_t)Lm * b * M.maxpos * NKV);
      { std::vector<bf16> e((size_t)b * D); for (int i = 0; i < b; i++) CK(cudaMemcpy(e.data() + (size_t)i * D, w + (size_t)pr[(size_t)i * P] * D, D * 2, cudaMemcpyDeviceToHost));
        std::vector<float> h(e.size()); for (size_t i = 0; i < e.size(); i++) h[i] = __bfloat162float(e[i]); CK(cudaMemcpy(M.h, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
      printf("dec real: Qwen3-4B %d layers on the trunk-ckpt weights, b=%d, prompt %d + %d tokens\n", Lm, b, P, dtok);
    }
  }                                   // Qwen3-4B MLP width: trunk layer = frozen up [TI][D] then down [D][TI] GEMM
  C.lt = lt; const bf16* Wup = lt ? fb((size_t)lt * TI * D, 40, 1.7320508f / sqrtf((float)D)) : nullptr;
  const bf16* Wdn = lt ? fb((size_t)lt * D * TI, 41, 1.7320508f / sqrtf((float)TI)) : nullptr;
  if (lt) { C.Xt[0] = dalloc<bf16>((size_t)tk * D); C.Xt[1] = dalloc<bf16>((size_t)tk * D); C.Ht = dalloc<bf16>((size_t)tk * TI); C.junk = reinterpret_cast<bf16*>(dalloc<float>((size_t)tk * TI)); }
  if (kc && (kc % 64 || D % kc || TI % kc)) { fprintf(stderr, "--trunk-kchunk must be a multiple of 64 dividing %d and %d\n", D, TI); return 2; }
  float* Cf = lt && kc ? dalloc<float>((size_t)tk * TI) : nullptr;
  C.P = dalloc<float>((size_t)jobs * PJ);
  for (int j = 0; j < jobs; j++) fill_f32<<<256, 256>>>(C.P + (size_t)j * PJ, 2 * R * D, seed * 977 + jb + j, 1.7320508f / sqrtf((float)D));   // A random, B zero (LoRA init)
  // --tail real: training_prefix tail inputs from h100/tail_prep.py (tokens, shifted labels, LoRA A init, per-job lr); one scratch set, jobs serialized
  C.rlab = nullptr; C.tinv = nullptr; C.lrj = nullptr; C.wd = 0.f;
  bf16 *Yb = nullptr, *Qpre = nullptr, *Qn = nullptr, *Tatt = nullptr, *Txn = nullptr, *Tgu = nullptr, *Tact = nullptr, *Thf = nullptr, *TdL = nullptr;
  bf16 *Tdh3b = nullptr, *Tdact = nullptr, *Tdgu = nullptr, *Tdh2b = nullptr, *Tdatt = nullptr;
  float *Tlse = nullptr, *Th2 = nullptr, *Th3 = nullptr, *TL = nullptr, *Tdhf = nullptr, *Tdh3 = nullptr, *Tdxn = nullptr, *Tdh2 = nullptr;
  if (treal) {
    auto rd = [&](const char* name, void* dst, size_t bytes, size_t skip) { FILE* f = fopen((tdata + "/" + name).c_str(), "rb");
      if (!f || fseek(f, (long)skip, SEEK_SET) || fread(dst, 1, bytes, f) != bytes) { fprintf(stderr, "cannot read %zu bytes at %zu from %s/%s\n", bytes, skip, tdata.c_str(), name); exit(2); }
      fclose(f); };
    std::vector<float> a0((size_t)jobs * 2 * R * D), lr(jobs); std::vector<int> lab((size_t)nx * tk);
    rd("tail_A.f32", a0.data(), a0.size() * 4, (size_t)jb * 2 * R * D * 4); rd("tail_lr.f32", lr.data(), lr.size() * 4, (size_t)jb * 4); rd("tail_labels.i32", lab.data(), lab.size() * 4, 0);
    for (int j = 0; j < jobs; j++) CK(cudaMemcpy(C.P + (size_t)j * PJ, a0.data() + (size_t)j * 2 * R * D, (size_t)2 * R * D * 4, cudaMemcpyHostToDevice));
    std::vector<float> tinv(nx);
    for (int x = 0; x < nx; x++) { int nv = 0; for (int t = 0; t < tk; t++) { const int y = lab[(size_t)x * tk + t]; if (y >= VOC) { fprintf(stderr, "label out of range\n"); return 2; } nv += y >= 0; }
      if (!nv) { fprintf(stderr, "microbatch %d has no labels\n", x); return 2; } tinv[x] = 1.f / nv; }
    int* dl = dalloc<int>(lab.size()); CK(cudaMemcpy(dl, lab.data(), lab.size() * 4, cudaMemcpyHostToDevice)); C.rlab = dl;
    float* dti = dalloc<float>(nx); CK(cudaMemcpy(dti, tinv.data(), nx * 4, cudaMemcpyHostToDevice)); C.tinv = dti;
    float* dlr = dalloc<float>(jobs); CK(cudaMemcpy(dlr, lr.data(), jobs * 4, cudaMemcpyHostToDevice)); C.lrj = dlr; C.wd = 0.01f;
    const size_t T = tk;
    Yb = dalloc<bf16>(T * NQKV); Qpre = dalloc<bf16>(T * NQKV); Qn = dalloc<bf16>(T * NQKV); Tatt = dalloc<bf16>(T * NQ); Tlse = dalloc<float>(T * 32);
    Th2 = dalloc<float>(T * D); Th3 = dalloc<float>(T * D); Txn = dalloc<bf16>(T * D); Thf = dalloc<bf16>(T * D); Tgu = dalloc<bf16>(T * 2 * TI); Tact = dalloc<bf16>(T * TI);
    TL = dalloc<float>(T * VOC); TdL = dalloc<bf16>(T * VOC); Tdhf = dalloc<float>(T * D); Tdh3 = dalloc<float>(T * D); Tdh3b = dalloc<bf16>(T * D);
    Tdact = dalloc<bf16>(T * TI); Tdgu = dalloc<bf16>(T * 2 * TI); Tdxn = dalloc<float>(T * D); Tdh2 = dalloc<float>(T * D); Tdh2b = dalloc<bf16>(T * D); Tdatt = dalloc<bf16>(T * NQ);
    printf("tail real: %s, %d microbatches, jobs %d..%d, lr %g..%g, AdamW wd 0.01, CE over %d vocab\n", tdata.c_str(), nx, jb, jb + jobs - 1, lr[0], lr[jobs - 1], VOC);
  }
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
  if (!dir.empty() && mode != 1 && !treal) { dump(dir, "init_P.f32", C.P, (size_t)jobs * PJ); dump(dir, "X.bf16", C.X, (size_t)nx * tk * D); dump(dir, "Wq.bf16", C.Wq, (size_t)NQ * D);
    dump(dir, "Wv.bf16", C.Wv, (size_t)NKV * D); dump(dir, "Tq.bf16", C.Tq, (size_t)tk * NQ); dump(dir, "Tv.bf16", C.Tv, (size_t)tk * NKV); }

  // train graph (per step); (phase, -1) = previous step
  Builder Bt; const int rbt = jobs * (tk / ROWS);
  Ph q = mk(T_BASEQ, 0), v = mk(T_BASEV, 0);
  for (Ph* p : {&q, &v}) { const int n = p == &q ? NQ : NKV; p->ga.A = C.X; p->ga.B = p == &q ? C.Wq : C.Wv; p->ga.C = p == &q ? (void*)C.Yq : (void*)C.Yv;
    p->ga.lda = D; p->ga.ldb = D; p->ga.ldc = n; p->ga.ldaux = 8; p->ga.M = tk; p->ga.N = n; p->ga.K = D; p->tilesM = tk / 128; p->ntasks = (tk / 128) * (n / 128);
    if (!gemm::args_ok(p->ga)) { fprintf(stderr, "bad GEMM args\n"); return 2; } }
  if (treal) { q.ga.N = NQKV; q.ga.ldc = NQKV; q.ga.C = Yb; q.ntasks = (tk / 128) * (NQKV / 128); }   // real tail: shared base q/k/v of layer lt
#ifdef MW_GEMM_WG
  for (Ph* p : {&q, &v}) { std::vector<const CUtensorMap*> maps(nx);
    for (int x = 0; x < nx; x++) maps[x] = gemm::sm90::make_map(lt ? C.Xt[lt % 2] : C.X + (size_t)x * tk * D, D, tk, D, 64, 128);
    const CUtensorMap** dm; CK(cudaMalloc(&dm, nx * sizeof(void*))); CK(cudaMemcpy(dm, maps.data(), nx * sizeof(void*), cudaMemcpyHostToDevice));
    p->tmAx = dm; p->tmB = gemm::sm90::make_map(p->ga.B, D, p->ga.N, D, 64, 128); }
  printf("gemm_backend wgmma\n");
#else
  printf("gemm_backend mma_sync\n");
#endif
  const int KS1 = kc && D > kc ? D / kc : 1, KS2 = kc && TI > kc ? TI / kc : 1;   // split-K chunks (up K=D, down K=TI): bounds trunk tile time
  const int XT = treal ? jobs * 22 - 1 : 0;   // real tail: 22 per-job phases replace T_UPG
  const int T0 = !ckpt.empty() ? (trunk_s == "perjob" ? jobs : 1) * (2 + 9 * lt) : lt * ((KS1 > 1 ? KS1 + 1 : 1) + (KS2 > 1 ? KS2 + 1 : 1)), ADAM = T0 + 7 + XT, LOSS = T0 + 8 + XT, reps = trunk_s == "perjob" ? jobs : 1;
  int prev = -1;
  auto dep_prev = [&](const Ph& t) { prev = prev < 0 ? Bt.add(t, {make_int2(ADAM, -1)}) : Bt.add(t, {make_int2(prev, 0)}); };
  if (ckpt.empty()) for (int l = 0; l < lt; l++) for (int k = 0; k < 2; k++) {   // trunk phases first: tickets are claimed in phase order
    const int n = k ? D : TI, kk = k ? TI : D, ks = k ? KS2 : KS1, kcs = kk / ks;
    const bf16* A = k ? C.Ht : (l ? C.Xt[l % 2] : C.X); const bf16* Bw = k ? Wdn + (size_t)l * D * TI : Wup + (size_t)l * TI * D;
    bf16* Cb = k ? C.Xt[(l + 1) % 2] : C.Ht;
    for (int c = 0; c < ks; c++) {   // chunk c accumulates into Cf after chunk c-1 (fixed order: deterministic)
      Ph t = mk(k ? T_TDN : T_TUP, 0); t.layer = l; t.reps = reps; t.f32 = ks > 1; t.koff = c * kcs;
      t.ga.A = A + (size_t)c * kcs; t.ga.B = Bw + (size_t)c * kcs; t.ga.C = ks > 1 ? (void*)Cf : (void*)Cb; t.ga.accumulate = c > 0;
      t.ga.lda = kk; t.ga.ldb = kk; t.ga.ldc = n; t.ga.ldaux = 8; t.ga.M = tk; t.ga.N = n; t.ga.K = kcs; t.tilesM = tk / 128; t.ntasks = reps * (tk / 128) * (n / 128);
      if (!gemm::args_ok(t.ga)) { fprintf(stderr, "bad trunk GEMM args\n"); return 2; }
#ifdef MW_GEMM_WG
      { const int nm = (!k && !l) ? nx : 1; std::vector<const CUtensorMap*> maps(nm);
        for (int x = 0; x < nm; x++) maps[x] = gemm::sm90::make_map((const bf16*)t.ga.A + (size_t)x * tk * D, kcs, tk, kk, 64, 128);
        const CUtensorMap** dm; CK(cudaMalloc(&dm, nm * sizeof(void*))); CK(cudaMemcpy(dm, maps.data(), nm * sizeof(void*), cudaMemcpyHostToDevice));
        t.tmAx = dm; t.tmB = gemm::sm90::make_map(t.ga.B, kcs, n, kk, 64, 128); }
#endif
      dep_prev(t);
    }
    if (ks > 1) { Ph t = mk(T_TCV, tk * n / (THREADS * 8)); t.reps = 1; t.ga.A = reinterpret_cast<const bf16*>(Cf); t.ga.C = Cb; dep_prev(t); }
  }
  if (!ckpt.empty()) for (int r = 0; r < reps; r++) {   // real trunk; per-job arm recomputes the whole trunk once per job (identical values)
    auto ew = [&](int kind, int nt, const bf16* Bp, void* Cp) { Ph t = mk(kind, nt); t.reps = 1; t.ga.B = Bp; t.ga.C = Cp; dep_prev(t); };
    auto gm = [&](const bf16* A, const bf16* Bp, void* Cp, int n, int k, int f32) {   // f32: C += A B^T into the fp32 residual
      Ph t = mk(T_TDN, 0); t.layer = 1; t.reps = 1; t.f32 = f32; t.koff = 0;
      t.ga.A = A; t.ga.B = Bp; t.ga.C = Cp; t.ga.accumulate = f32; t.ga.lda = k; t.ga.ldb = k; t.ga.ldc = n; t.ga.ldaux = 8; t.ga.M = tk; t.ga.N = n; t.ga.K = k;
      t.tilesM = tk / 128; t.ntasks = (tk / 128) * (n / 128);
      if (!gemm::args_ok(t.ga)) { fprintf(stderr, "bad real-trunk GEMM args\n"); exit(2); }
#ifdef MW_GEMM_WG
      { const CUtensorMap* m0 = gemm::sm90::make_map(A, k, tk, k, 64, 128); const CUtensorMap** dm; CK(cudaMalloc(&dm, sizeof(void*)));
        CK(cudaMemcpy(dm, &m0, sizeof(void*), cudaMemcpyHostToDevice)); t.tmAx = dm; t.tmB = gemm::sm90::make_map(Bp, k, n, k, 64, 128); }
#endif
      dep_prev(t); };
    ew(T_REMB, tk * D / (THREADS * 8), nullptr, nullptr);
    for (int l = 0; l < lt; l++) {
      const bf16* W = RW + (size_t)l * lst; const bf16* Wn = W + D + (size_t)NQKV * D; const bf16* Wo = Wn + 2 * HD; const bf16* Wp = Wo + (size_t)D * NQ;
      const bf16* Wgu = Wp + D; const bf16* Wd = Wgu + (size_t)2 * TI * D;
      ew(T_RRMS, tk / 8, W, C.rxn); gm(C.rxn, W + D, C.rqkv, NQKV, D, 0); ew(T_RQKN, tk * 40 / 8, Wn, nullptr); if (atc) ew(T_RATTC, tk / 4, nullptr, nullptr); else ew(T_RATT, tk * 32 / 8, nullptr, nullptr);
      gm(C.ratt, Wo, C.rh, D, NQ, 1); ew(T_RRMS, tk / 8, Wp, C.rxn); gm(C.rxn, Wgu, C.rgu, 2 * TI, D, 0);
      ew(T_RACT, tk * TI / (THREADS * 8), nullptr, nullptr); gm(C.ract, Wd, C.rh, D, TI, 1);
    }
    ew(T_RRMS, tk / 8, RW + (size_t)lt * lst, C.Xt[lt % 2]);   // cut: layer lt input_layernorm(h) = tail input
  }
  if (prev != T0 - 1) { fprintf(stderr, "trunk phase count mismatch\n"); return 2; }
  if (lt) { Bt.add(q, {make_int2(LOSS, -1), make_int2(T0 - 1, 0)}); Bt.add(v, {make_int2(LOSS, -1), make_int2(T0 - 1, 0)});
    Bt.add(mk(T_DOWN, rbt), {make_int2(ADAM, -1), make_int2(T0 - 1, 0)}); }
  else { Bt.add(q, {make_int2(LOSS, -1)}); Bt.add(v, {make_int2(LOSS, -1)}); Bt.add(mk(T_DOWN, rbt), {make_int2(ADAM, -1)}); }
  if (!treal) Bt.add(mk(T_UPG, rbt), {make_int2(T0 + 0, 0), make_int2(T0 + 1, 0), make_int2(T0 + 2, 0)});
  else {   // real top layer per job (training_prefix tail): forward, lm_head + CE, backward to dq/dv -> dO; jobs chained in order over one scratch set
    const bf16* W = RW + (size_t)lt * lst; const bf16* Wn = W + D + (size_t)NQKV * D; const bf16* Wo = Wn + 2 * HD; const bf16* Wp = Wo + (size_t)D * NQ;
    const bf16* Wgu = Wp + D; const bf16* Wd = Wgu + (size_t)2 * TI * D; const bf16* Wf = C.Remb + (size_t)VOC * D;
    int cprev = -1, jj = 0;
    auto chn = [&](Ph t) { t.layer = jj; cprev = cprev < 0 ? Bt.add(t, {make_int2(T0 + 0, 0), make_int2(T0 + 1, 0), make_int2(T0 + 2, 0)}) : Bt.add(t, {make_int2(cprev, 0)}); };
    auto ewj = [&](int kind, int nt, const bf16* A, const bf16* Bp, void* Cp, bf16* aux, float* fx, float* fx2, const bf16* w2) {
      Ph t = mk(kind, nt); t.ga.A = A; t.ga.B = Bp; t.ga.C = Cp; t.ga.aux = aux; t.fx = fx; t.fx2 = fx2; t.w2 = w2; chn(t); };
    auto gmj = [&](int kind, const bf16* A, const bf16* Bp, void* Cp, int n, int k, int f32, int acc) {   // T_TDN: C (+)= A.B^T (B [N][K]); T_GNN: C = A.B (B [K][N])
      Ph t = mk(kind, (tk / 128) * (n / 128)); t.reps = 1; t.f32 = f32; t.tilesM = tk / 128;
      t.ga.A = A; t.ga.B = Bp; t.ga.C = Cp; t.ga.accumulate = acc; t.ga.lda = k; t.ga.ldb = kind == T_GNN ? n : k; t.ga.ldc = n; t.ga.ldaux = 8; t.ga.M = tk; t.ga.N = n; t.ga.K = k;
      if (!gemm::args_ok(t.ga)) { fprintf(stderr, "bad real-tail GEMM args\n"); exit(2); }
#ifdef MW_GEMM_WG
      if (kind == T_TDN) { const CUtensorMap* m0 = gemm::sm90::make_map(A, k, tk, k, 64, 128); const CUtensorMap** dm; CK(cudaMalloc(&dm, sizeof(void*)));
        CK(cudaMemcpy(dm, &m0, sizeof(void*), cudaMemcpyHostToDevice)); t.tmAx = dm; t.tmB = gemm::sm90::make_map(Bp, k, n, k, 64, 128); }
#endif
      chn(t); };
    const int E8 = (int)((size_t)tk * D / (THREADS * 8)), EI = (int)((size_t)tk * TI / (THREADS * 8));
    for (jj = 0; jj < jobs; jj++) {
      ewj(T_RQV, (int)((size_t)tk * NQKV / (THREADS * 8)), Yb, nullptr, Qpre, Qn, nullptr, nullptr, nullptr);
      ewj(T_RQKN, tk * 40 / 8, nullptr, Wn, Qn, nullptr, nullptr, nullptr, nullptr);
      ewj(atc ? T_RATTC : T_RATT, atc ? tk / 4 : tk * 32 / 8, Qn, nullptr, Tatt, nullptr, Tlse, nullptr, nullptr);
      ewj(T_RCP, E8, nullptr, nullptr, Th2, nullptr, C.rh, nullptr, nullptr);
      gmj(T_TDN, Tatt, Wo, Th2, D, NQ, 1, 1);
      ewj(T_RRMS, tk / 8, reinterpret_cast<const bf16*>(Th2), Wp, Txn, nullptr, nullptr, nullptr, nullptr);
      gmj(T_TDN, Txn, Wgu, Tgu, 2 * TI, D, 0, 0);
      ewj(T_RACT, EI, Tgu, nullptr, Tact, nullptr, nullptr, nullptr, nullptr);
      ewj(T_RCP, E8, nullptr, nullptr, Th3, nullptr, Th2, nullptr, nullptr);
      gmj(T_TDN, Tact, Wd, Th3, D, TI, 1, 1);
      ewj(T_RRMS, tk / 8, reinterpret_cast<const bf16*>(Th3), Wf, Thf, nullptr, nullptr, nullptr, nullptr);
      gmj(T_TDN, Thf, C.Remb, TL, VOC, D, 1, 0);                                  // logits (fp32, one job at a time)
      ewj(T_CE, tk / 16, nullptr, nullptr, TdL, nullptr, TL, nullptr, nullptr);
      gmj(T_GNN, TdL, C.Remb, Tdhf, D, VOC, 1, 0);                                // dhidden = dlogits . E (lm_head frozen: no weight grad)
      ewj(T_RNB, tk / 8, reinterpret_cast<const bf16*>(Th3), Wf, Tdh3, Tdh3b, Tdhf, nullptr, nullptr);
      gmj(T_GNN, Tdh3b, Wd, Tdact, TI, D, 0, 0);
      ewj(T_RACTB, EI, Tgu, Tdact, Tdgu, nullptr, nullptr, nullptr, nullptr);
      gmj(T_GNN, Tdgu, Wgu, Tdxn, D, 2 * TI, 1, 0);
      ewj(T_RNB, tk / 8, reinterpret_cast<const bf16*>(Th2), Wp, Tdh2, Tdh2b, Tdxn, Tdh3, nullptr);
      gmj(T_GNN, Tdh2b, Wo, Tdatt, NQ, D, 0, 0);
      ewj(atcb ? T_RADQC : T_RADQ, atcb ? tk / 4 : tk * 32 / 8, Qn, Tatt, Qpre, Tdatt, Tlse, nullptr, Wn);
      ewj(atcb ? T_RADVC : T_RADV, atcb ? tk / 16 : tk * NKVH / 8, Qn, nullptr, nullptr, Tdatt, Tlse, nullptr, nullptr);
    }
  }
  Bt.add(mk(T_DB, jobs * (NQV / THREADS)), {make_int2(T0 + 3 + XT, 0)});
  Bt.add(mk(T_DU, rbt), {make_int2(T0 + 3 + XT, 0)});
  Bt.add(mk(T_DA, jobs * (D / THREADS)), {make_int2(T0 + 5 + XT, 0)});
  Bt.add(mk(T_ADAM, jobs * (PJ / (THREADS * 4))), {make_int2(T0 + 4 + XT, 0), make_int2(T0 + 6 + XT, 0)});
  Bt.add(mk(T_LOSS, 1), {make_int2(T0 + 3 + XT, 0)});
  if ((int)Bt.ph.size() != LOSS + 1) { fprintf(stderr, "tail phase count mismatch\n"); return 2; }
  printf("tail %s, attn %s, attn-bwd %s\n", tail_s.c_str(), attn_s.c_str(), abwd_s.c_str());
  printf("trunk: %d frozen layers (%s, k-chunk %d), job-base %d, sm-split %s, kernel launches 1 for %d steps\n", lt, trunk_s.c_str(), kc, jb, split_s.c_str(), steps);
  upload(Bt, mode == 1 || mode == 4 ? 0 : steps, ADAM, C.tr);
  // decode graph (per token)
  Builder Bd;
  Bd.add(mk(D_DOWN, 1), {make_int2(5, -1)}); Bd.add(mk(D_QKV, NQKV / 128), {make_int2(0, 0)}); Bd.add(mk(D_ATT, b * NKVH), {make_int2(1, 0)});
  Bd.add(mk(D_O, D / 128), {make_int2(2, 0)}); Bd.add(mk(D_HEAD, nhead), {make_int2(3, 0)}); Bd.add(mk(D_NEXT, 1), {make_int2(4, 0)});
  if (dreal) {   // real decode graph (mm.cu load_model order): per layer norm, qkv, attn, o, norm, gate/up, down; final norm, lm_head, argmax
    Bd = Builder(); int pv = -1;
    auto add = [&](int k, int n, int l, int nm) { Ph p = mk(D_REAL, n); p.koff = k; p.layer = l; p.f32 = nm; pv = Bd.add(p, {pv < 0 ? make_int2(0, -1) : make_int2(pv, 0)}); };
    for (int l = 0; l < C.RM.L; l++) { add(mmd::K_NORM, b, l, 0); add(mmd::K_QKV, NQKV / 128, l, 0); add(mmd::K_ATT, b * NKVH, l, 0); add(mmd::K_O, D / 128, l, 0);
      add(mmd::K_NORM, b, l, 1); add(mmd::K_GU, 2 * TI / 128, l, 0); add(mmd::K_DN, D / 128, l, 0); }
    add(mmd::K_NORM, b, 0, 2); add(mmd::K_HEAD, VOC / 128, 0, 0); add(mmd::K_NEXT, 1, 0, 0);
    Bd.deps[Bd.dep_off[0]] = make_int2(pv, -1);   // layer-0 norm waits for the previous token's argmax
    upload(Bd, mode == 0 || mode == 4 ? 0 : dlim, pv, C.dc); C.tok_done = C.RM.tok_done; dtok = dlim;
  } else upload(Bd, mode == 0 || mode == 4 ? 0 : dtok, 5, C.dc);
  printf("train: %d phases, %d tasks/step; decode: 6 phases, %d tasks/token\n", (int)Bt.ph.size(), C.tr.tps, C.dc.tps);
  Ctx Pc = C; bf16* pf_out = nullptr; int pf_last = -1;
  if (npf) {   // prefill tenant: chunks of tk tokens (rseq segments) through all real layers + final norm; own activations and queue, chunks in order
    if (VA != VOC) { fprintf(stderr, "--prefill-chunks: need vocab %d\n", VOC); return 2; }
    std::vector<int> pt((size_t)npf * tk); FILE* pfp = fopen(pfdata.c_str(), "rb");
    if (!pfp || fread(pt.data(), 4, pt.size(), pfp) != pt.size()) { fprintf(stderr, "cannot read %zu tokens from %s\n", pt.size(), pfdata.c_str()); return 2; }
    fclose(pfp); for (int t : pt) if (t < 0 || t >= VOC) { fprintf(stderr, "prefill token id out of range\n"); return 2; }
    int* dpt = dalloc<int>(pt.size()); CK(cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
    Pc.pf = nullptr; Pc.rtok = dpt; Pc.nx = npf; Pc.step_done = dalloc<u64>(npf); Pc.finished = dalloc<int>(1);
    Pc.rh = dalloc<float>((size_t)tk * D); Pc.rxn = dalloc<bf16>((size_t)tk * D); Pc.rqkv = dalloc<bf16>((size_t)tk * NQKV);
    Pc.ratt = dalloc<bf16>((size_t)tk * NQ); Pc.rgu = dalloc<bf16>((size_t)tk * 2 * TI); Pc.ract = dalloc<bf16>((size_t)tk * TI);
    bf16* pfn = dalloc<bf16>((size_t)tk * D); pf_out = dalloc<bf16>((size_t)npf * tk * D);
    Builder Bp;
    auto pdep = [&](Ph t) { pf_last = Bp.add(t, {pf_last < 0 ? make_int2(0, -1) : make_int2(pf_last, 0)}); };
    auto pew = [&](int kind, int nt, const bf16* Bw, void* Cp) { Ph t = mk(kind, nt); t.reps = 1; t.ga.B = Bw; t.ga.C = Cp; pdep(t); };
    auto pgm = [&](const bf16* A, const bf16* Bw, void* Cp, int n, int k, int f32) {
      Ph t = mk(T_TDN, 0); t.layer = 1; t.reps = 1; t.f32 = f32; t.koff = 0;
      t.ga.A = A; t.ga.B = Bw; t.ga.C = Cp; t.ga.accumulate = f32; t.ga.lda = k; t.ga.ldb = k; t.ga.ldc = n; t.ga.ldaux = 8; t.ga.M = tk; t.ga.N = n; t.ga.K = k;
      t.tilesM = tk / 128; t.ntasks = (tk / 128) * (n / 128);
      if (!gemm::args_ok(t.ga)) { fprintf(stderr, "bad prefill GEMM args\n"); exit(2); }
#ifdef MW_GEMM_WG
      { const CUtensorMap* m0 = gemm::sm90::make_map(A, k, tk, k, 64, 128); const CUtensorMap** dm; CK(cudaMalloc(&dm, sizeof(void*)));
        CK(cudaMemcpy(dm, &m0, sizeof(void*), cudaMemcpyHostToDevice)); t.tmAx = dm; t.tmB = gemm::sm90::make_map(Bw, k, n, k, 64, 128); }
#endif
      pdep(t); };
    pew(T_REMB, tk * D / (THREADS * 8), nullptr, nullptr);
    for (int l = 0; l < LmA; l++) {
      const bf16* W = RW + (size_t)l * lst; const bf16* Wn = W + D + (size_t)NQKV * D; const bf16* Wo = Wn + 2 * HD; const bf16* Wp = Wo + (size_t)D * NQ;
      const bf16* Wgu = Wp + D; const bf16* Wd = Wgu + (size_t)2 * TI * D;
      pew(T_RRMS, tk / 8, W, Pc.rxn); pgm(Pc.rxn, W + D, Pc.rqkv, NQKV, D, 0); pew(T_RQKN, tk * 40 / 8, Wn, nullptr);
      if (atc) pew(T_RATTC, tk / 4, nullptr, nullptr); else pew(T_RATT, tk * 32 / 8, nullptr, nullptr);
      pgm(Pc.ratt, Wo, Pc.rh, D, NQ, 1); pew(T_RRMS, tk / 8, Wp, Pc.rxn); pgm(Pc.rxn, Wgu, Pc.rgu, 2 * TI, D, 0);
      pew(T_RACT, tk * TI / (THREADS * 8), nullptr, nullptr); pgm(Pc.ract, Wd, Pc.rh, D, TI, 1);
    }
    pew(T_RRMS, tk / 8, C.Remb + (size_t)VOC * D, pfn);   // final norm
    { Ph t = mk(T_PFCP, tk * D / (THREADS * 8)); t.reps = 1; t.ga.A = pfn; t.ga.C = pf_out; pdep(t); }
    Bp.deps[Bp.dep_off[0]] = make_int2(pf_last, -1);   // chunk c+1 starts after chunk c's output copy (one activation set)
    upload(Bp, npf, pf_last, Pc.tr);
    Ctx* dpc; CK(cudaMalloc(&dpc, sizeof(Ctx))); CK(cudaMemcpy(dpc, &Pc, sizeof(Ctx), cudaMemcpyHostToDevice)); C.pf = dpc;
    printf("prefill: %d chunks x %d tokens (seq %d), %d layers + final norm, %d phases, %d tasks/chunk\n", npf, tk, rseq, LmA, (int)Bp.ph.size(), Pc.tr.tps);
  }
  { size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot)); printf("device memory used %.1f of %.1f GB\n", (tot - fr) / 1e9, tot / 1e9); }

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); mw_kernel<<<nblocks, THREADS, SMEM>>>(C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); unsigned lent = 0; CK(cudaMemcpy(&lent, C.lent, 4, cudaMemcpyDeviceToHost)); printf("kernel %.3f s, lending %s, lent train tasks %u\n", ms / 1e3, C.lend ? "on" : "off", lent);

  u64 tr_lo = 0, tr_hi = 0;
  if (mode != 1 && mode != 4) {
    std::vector<u64> sd(steps); CK(cudaMemcpy(sd.data(), C.step_done, steps * 8, cudaMemcpyDeviceToHost));
    std::vector<float> loss((size_t)steps * jobs); CK(cudaMemcpy(loss.data(), C.loss, loss.size() * 4, cudaMemcpyDeviceToHost));
    int sdone = 0; while (sdone < steps && sd[sdone]) sdone++;   // early stop: --max-seconds or last decode request done
    const int w0 = std::min(std::max(sdone - 1, 0), 3); tr_lo = sdone ? sd[w0] : 0; tr_hi = sdone ? sd[sdone - 1] : 0;
    const double span = (tr_hi - tr_lo) * 1e-9; const int n = sdone - 1 - w0;
    printf("loss job0: step0 %.6f last %.6f (%d steps done)\n", loss[0], loss[(size_t)std::max(sdone - 1, 0) * jobs], sdone);
    if (n > 0) printf("train: steady %.4f ms/step -> %.2f steps/s (%d jobs x %d tokens: %.0f job-tok/s)\n", span * 1e3 / n, n / span, jobs, tk, (double)n * jobs * tk / span);
    dump(dir, "loss.f32", C.loss, (size_t)steps * jobs); dump(dir, "step_done.u64", C.step_done, (size_t)steps); dump(dir, "P.f32", C.P, (size_t)jobs * PJ); dump(dir, "G.f32", C.G, (size_t)jobs * PJ); dump(dir, "tmax.u64", C.tmax, 64); if (!ckpt.empty()) dump(dir, "Xt.bf16", C.Xt[lt % 2], (size_t)tk * D);
  }
  if (npf) {
    std::vector<u64> pd(npf); u64 k0 = 0; CK(cudaMemcpy(pd.data(), Pc.step_done, npf * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&k0, C.t0, 8, cudaMemcpyDeviceToHost));
    int pdn = 0; while (pdn < npf && pd[pdn]) pdn++;
    std::vector<double> ct; for (int c = 0; c < pdn; c++) ct.push_back((pd[c] - (c ? pd[c - 1] : k0)) * 1e-6);
    double c50, c99; pct(ct, c50, c99); const double sp = pdn ? (pd[pdn - 1] - k0) * 1e-9 : 0;
    printf("prefill: %d of %d chunks done in %.3f s -> %.1f tok/s, chunk p50 %.3f ms p99 %.3f ms\n", pdn, npf, sp, pdn * (double)tk / (sp + 1e-12), c50, c99);
    dump(dir, "pf_out.bf16", pf_out, (size_t)pdn * tk * D); dump(dir, "pf_done.u64", Pc.step_done, (size_t)pdn);
  }
  if (mode != 0 && mode != 4) {
    std::vector<u64> td(dtok); CK(cudaMemcpy(td.data(), C.tok_done, dtok * 8, cudaMemcpyDeviceToHost));
    int done = 0; while (done < dtok && td[done]) done++;
    std::vector<double> lat; int inwin = 0; u64 lo = 0, hi = 0;
    for (int t = 1; t < done; t++) {
      if ((mode == 2 && (td[t] < tr_lo || td[t] > tr_hi)) || (rate > 0 && t % Lr == 0) || (dreal && t < dP)) continue;   // request mode: TPOT excludes queueing before a first token
      lat.push_back((td[t] - td[t - 1]) * 1e-6); if (!inwin) lo = td[t - 1]; hi = td[t]; inwin++;
    }
    double p50, p99; pct(lat, p50, p99); const double win = (hi - lo) * 1e-9;
    printf("decode: b=%d, %d token steps (%d in window %.3f s) -> %.1f tok/s, tpot p50 %.4f ms p99 %.4f ms\n", b, done, inwin, win, inwin * b / (win + 1e-12), p50, p99);
    if (!dir.empty() && dreal) { dump(dir, "toks.i32", C.RM.toks, (size_t)done * mmd::MAXB); dump(dir, "tok_done.u64", C.tok_done, (size_t)done); dump(dir, "logitsP.f32", C.RM.logitsP, (size_t)b * VOC); }
    else if (!dir.empty()) { dump(dir, "toks.i32", C.toks, (size_t)done * MAXB); dump(dir, "tok_done.u64", C.tok_done, (size_t)done); if (C.arr) { dump(dir, "arr.u64", const_cast<u64*>(C.arr), (size_t)nreq); dump(dir, "t0.u64", C.t0, 1); } dump(dir, "lsum.f32", C.lsum, (size_t)done * MAXB); dump(dir, "logits0.f32", C.logits0, (size_t)b * VOC); }
  }
  return 0;
}
