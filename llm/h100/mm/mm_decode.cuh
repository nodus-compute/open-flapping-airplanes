// mm_decode.cuh : real-weight Qwen3 decode instructions (h100/mm_prep.py layout), shared by mm.cu and trunk.cu (--dec real).
//   Fixed reduction order, no float atomics.
#pragma once
#include <cmath>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

namespace mmd {

constexpr int HD = 128, MAXB = 32, CHB = 8, THREADS = 256, ROWS = 16, MAXHPK = 4, SPOS = 4096;

enum Kind { K_NORM, K_QKV, K_ATT, K_O, K_GU, K_DN, K_HEAD, K_NEXT };
struct Ph { int kind, ntasks, layer, norm; };
struct Model {
  int H, L, NH, NKV, I, V, NQ, NK, nqkv, b, P, maxpos; float eps, theta;
  const bf16 *emb, *fnorm, *w; size_t lst, o_in, o_qkv, o_qn, o_kn, o_o, o_post, o_gu, o_d;
  float *h, *xn, *q, *kraw, *att, *act, *lmax, *lsum_part, *logitsP; int *lidx, *toks; const int* prompt; bf16 *kc, *vc; u64* tok_done;
};

__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ float block_sum(float v, float* red) {
  red[threadIdx.x] = v; __syncthreads();
  float s = 0.f;
  if (threadIdx.x == 0) { for (int i = 0; i < THREADS; i++) s += red[i]; red[0] = s; }
  __syncthreads(); s = red[0]; __syncthreads(); return s;
}
// row-dot GEMV over 128 consecutive rows of W[rows][K] (bf16) for b inputs xin[b][K] (fp32), batch in chunks of 8; fixed order.
template <class F>
__device__ __forceinline__ void gemv128(const bf16* W, int K, int row0, int nrows, const float* xin, int b, F emit) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int rr = 0; rr < ROWS; rr++) {
    const int row = row0 + warp * ROWS + rr;
    if (row >= nrows) break;
    const bf16* w = W + (size_t)row * K;
    for (int c0 = 0; c0 < b; c0 += CHB) {
      float acc[CHB];
#pragma unroll
      for (int i = 0; i < CHB; i++) acc[i] = 0.f;
      for (int k = lane * 8; k < K; k += 256) {
        const uint4 raw = *reinterpret_cast<const uint4*>(w + k);
        const bf16* wv = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int i = 0; i < CHB; i++) if (c0 + i < b) {
          const float* x = xin + (size_t)(c0 + i) * K + k;
#pragma unroll
          for (int e = 0; e < 8; e++) acc[i] = fmaf(bf(wv[e]), x[e], acc[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < CHB; i++) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
      }
      if (lane == 0) for (int i = 0; i < CHB && c0 + i < b; i++) emit(c0 + i, row, acc[i]);
    }
  }
}

__device__ void decode_task(const Model& M, const Ph& d, int local, int step, char* smem) {
  float* sm = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, b = M.b, pos = step, l = d.layer, H = M.H;
  const bf16* LW = M.w + (size_t)l * M.lst;
  switch (d.kind) {
    case K_NORM: {                                             // row `local`: xn = h * rsqrt(mean(h^2) + eps) * w
      const float* x = M.h + (size_t)local * H;
      const bf16* w = d.norm == 0 ? LW + M.o_in : d.norm == 1 ? LW + M.o_post : M.fnorm;
      float ss = 0.f; for (int k = tid; k < H; k += THREADS) ss = fmaf(x[k], x[k], ss);
      const float inv = rsqrtf(block_sum(ss, sm) / H + M.eps);
      for (int k = tid; k < H; k += THREADS) M.xn[(size_t)local * H + k] = x[k] * inv * bf(w[k]);
      break;
    }
    case K_QKV:
      gemv128(LW + M.o_qkv, H, local * 128, M.nqkv, M.xn, b, [&](int i, int row, float v) {
        if (row < M.NQ) M.q[(size_t)i * M.NQ + row] = v;
        else if (row < M.NQ + M.NK) M.kraw[(size_t)i * M.NK + row - M.NQ] = v;
        else M.vc[(((size_t)l * b + i) * M.maxpos + pos) * M.NK + row - M.NQ - M.NK] = __float2bfloat16(v);
      });
      break;
    case K_ATT: {                                              // task (row i, kv head g): q/k head RMSNorm + RoPE, k -> cache, GQA attention
      const int hpk = M.NH / M.NKV, i = local / M.NKV, g = local % M.NKV, n = pos + 1, warp = tid >> 5, lane = tid & 31;
      __shared__ float sq[MAXHPK][HD];
      __shared__ float ssum[MAXHPK];
      bf16* kcl = M.kc + ((size_t)l * b + i) * M.maxpos * M.NK;
      const bf16* vcl = M.vc + ((size_t)l * b + i) * M.maxpos * M.NK;
      if (warp <= hpk) {
        const float* src = warp < hpk ? M.q + (size_t)i * M.NQ + (g * hpk + warp) * HD : M.kraw + (size_t)i * M.NK + g * HD;
        const bf16* nw = LW + (warp < hpk ? M.o_qn : M.o_kn);
        float x[4]; float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) { x[j] = src[lane + 32 * j]; ss = fmaf(x[j], x[j], ss); }
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / HD + M.eps);
#pragma unroll
        for (int j = 0; j < 4; j++) x[j] = x[j] * inv * bf(nw[lane + 32 * j]);
        float y[4];
#pragma unroll
        for (int j = 0; j < 2; j++) {                          // rotate-half RoPE: pairs (d, d + 64), d = lane + 32 j
          const int dd = lane + 32 * j; const float invf = powf(M.theta, -(float)(2 * dd) / (float)HD); const float ang = (float)pos * invf;
          const float c = cosf(ang), s = sinf(ang);
          y[j] = x[j] * c - x[j + 2] * s; y[j + 2] = x[j + 2] * c + x[j] * s;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) { if (warp < hpk) sq[warp][lane + 32 * j] = y[j]; else kcl[(size_t)pos * M.NK + g * HD + lane + 32 * j] = __float2bfloat16(y[j]); }
      }
      __syncthreads();
      for (int p = warp; p < n; p += THREADS / 32) {
        const bf16* k = kcl + (size_t)p * M.NK + g * HD;
        float kv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) kv[j] = bf(k[lane + 32 * j]);
        for (int hh = 0; hh < hpk; hh++) {
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 4; j++) s = fmaf(sq[hh][lane + 32 * j], kv[j], s);
          for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
          if (lane == 0) sm[hh * SPOS + p] = s * 0.08838834764831845f;
        }
      }
      __syncthreads();
      if (tid < hpk) { float m = -INFINITY; for (int p = 0; p < n; p++) m = fmaxf(m, sm[tid * SPOS + p]); float s = 0.f;
        for (int p = 0; p < n; p++) { const float e = __expf(sm[tid * SPOS + p] - m); sm[tid * SPOS + p] = e; s += e; } ssum[tid] = s; }
      __syncthreads();
      for (int o = tid; o < hpk * HD; o += THREADS) {
        const int hh = o / HD, e = o % HD; float a = 0.f;
        for (int p = 0; p < n; p++) a = fmaf(sm[hh * SPOS + p], bf(vcl[(size_t)p * M.NK + g * HD + e]), a);
        M.att[(size_t)i * M.NQ + (g * hpk + hh) * HD + e] = a / ssum[hh];
      }
      break;
    }
    case K_O: gemv128(LW + M.o_o, M.NQ, local * 128, H, M.att, b, [&](int i, int row, float v) { M.h[(size_t)i * H + row] += v; }); break;
    case K_GU: {                                               // rows interleaved per task: 64 gate rows then the matching 64 up rows
      gemv128(LW + M.o_gu, H, local * 128, 2 * M.I, M.xn, b, [&](int i, int row, float v) { sm[i * 128 + row - local * 128] = v; });
      __syncthreads();
      for (int e = tid; e < b * 64; e += THREADS) { const int i = e / 64, j = e % 64; const float gt = sm[i * 128 + j], up = sm[i * 128 + 64 + j];
        M.act[(size_t)i * M.I + local * 64 + j] = gt / (1.f + __expf(-gt)) * up; }
      break;
    }
    case K_DN: gemv128(LW + M.o_d, M.I, local * 128, H, M.act, b, [&](int i, int row, float v) { M.h[(size_t)i * H + row] += v; }); break;
    case K_HEAD: {
      float* lg = sm;
      gemv128(M.emb, H, local * 128, M.V, M.xn, b, [&](int i, int row, float v) { lg[i * 128 + row - local * 128] = v; if (step == M.P - 1) M.logitsP[(size_t)i * M.V + row] = v; });
      __syncthreads();
      if (tid < b) { float m = -INFINITY, s = 0.f; int mi = 0; for (int r = 0; r < 128; r++) { const float v = lg[tid * 128 + r]; s += v; if (v > m) { m = v; mi = local * 128 + r; } }
        M.lmax[(size_t)tid * d.ntasks + local] = m; M.lidx[(size_t)tid * d.ntasks + local] = mi; M.lsum_part[(size_t)tid * d.ntasks + local] = s; }
      break;
    }
    case K_NEXT: {                                             // greedy token; next input = prompt token (teacher forcing) or the argmax
      const int nt = M.V / 128;
      __shared__ int stok[MAXB];
      if (tid < b) { float m = -INFINITY; int mi = 0; for (int t = 0; t < nt; t++) { const float v = M.lmax[(size_t)tid * nt + t]; if (v > m) { m = v; mi = M.lidx[(size_t)tid * nt + t]; } }
        M.toks[(size_t)step * MAXB + tid] = mi; stok[tid] = step + 1 < M.P ? M.prompt[tid * M.P + step + 1] : mi; }
      __syncthreads();
      for (int e = tid; e < b * H; e += THREADS) M.h[e] = bf(M.emb[(size_t)stok[e / H] * H + e % H]);
      break;
    }
    default: break;
  }
}

}  // namespace mmd
