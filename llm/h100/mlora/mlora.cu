// mlora.cu : multi-adapter LoRA decode as a persistent-megakernel instruction (H100, sm_90a). ONE launch decodes --dec-tokens steps of a
//   batch-b (1..64) Qwen3-4B-shaped decoder layer (q/k/v GEMV, GQA attention over a KV cache, o-proj + residual, tied 151936-row lm_head,
//   greedy argmax, next-token embedding) whose rows carry different rank-8 q/v LoRA adapters (row i -> adapter i % --adapters).
//   --lora none : base layer, no LoRA.
//   --lora loop : per-adapter unfused LoRA: base q/v written un-adapted, then for each adapter a serial LoRA-down / LoRA-up phase pair over
//                 that adapter's rows (one "kernel" per adapter, the PEFT/PyTorch loop, kept inside the same persistent launch).
//   --lora sgmv : grouped SGMV/BGMV instruction (Punica-style): one LoRA-down phase over all rows gathers each row's own adapter, and the
//                 LoRA-up delta is applied in the base q/k/v GEMV epilogue with the row's adapter, so LoRA adds one phase per token.
//   loop and sgmv evaluate every element with the same per-element functions in the same order, so their outputs must be byte-equal.
//   Weights are hash-initialised (Qwen3-4B shapes, not the checkpoint); fixed reduction order everywhere, no float atomics.
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

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;

constexpr int D = 2560, NQ = 4096, NKV = 1024, NQV = NQ + NKV, NQKV = NQ + 2 * NKV, HD = 128, NKVH = 8, VOC = 151936, R = 8, R2 = 2 * R;
constexpr int PJ = 2 * R * D + NQV * R;            // per-adapter fp32: Aq[R][D] Av[R][D] Bq[NQ][R] Bv[NKV][R]
constexpr int MAXB = 64, CHB = 8, MAXA = 64, MAXPOS = 4096, THREADS = 256, ROWS = 16;
constexpr int SMEM = 4 * MAXPOS * 4;               // 64 KB: attention scores; lm_head partials need MAXB*128*4 = 32 KB
static_assert(MAXB * 128 * 4 <= SMEM, "lm_head partials fit smem");

enum Kind { D_DOWN, D_QKV, L_DOWN, L_UP, D_ATT, D_O, D_HEAD, D_NEXT };
enum { LM_NONE = 0, LM_LOOP = 1, LM_SGMV = 2 };
struct Ph { int kind, ntasks, arg; };
struct Job { int nph, tps, limit, last_ph; const Ph* ph; const int *prefix, *ntasks, *dep_off, *dep_cnt; const int2* deps; int* done; unsigned* ticket; };
struct Ctx {
  Job dc; int b, ctx0, lmode; float scale;
  const bf16 *Wqkv, *Wo, *E; const float* adp; const int *aid, *rows, *aoff;
  float *h, *du, *q, *vst, *att, *lmax, *lsum_part, *lsum, *logits0; int *lidx, *toks; bf16 *kc, *vc; u64* tok_done;
};

__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }

// row-dot GEMV over 128 consecutive rows of W[rows][K] (bf16) for b inputs xin[b][K] (fp32), batch in chunks of 8 (the weight row is
// re-read from L1 per chunk). Warp w owns 16 rows; lane-strided 8-element chunks, fixed-order xor-shuffle reduce; lane 0 emits.
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
// LoRA down for one row i with adapter a: du[i][r] = x_i . A_a[r], r < 2R (q then v). Warp w owns r = w, w + 8; fixed order.
__device__ __forceinline__ void lora_down_row(const Ctx& C, int i, int a) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* x = C.h + (size_t)i * D; const float* A = C.adp + (size_t)a * PJ;
  for (int r = warp; r < R2; r += THREADS / 32) {
    const float* ar = A + (size_t)r * D; float s = 0.f;
    for (int k = lane; k < D; k += 32) s = fmaf(x[k], ar[k], s);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) C.du[i * R2 + r] = s;
  }
}
// LoRA up for one output element: s = sum_r du[i][r] * B_a[o][r] over the q (o < NQ) or v (o >= NQ) half; fixed order.
__device__ __forceinline__ float lora_up(const Ctx& C, int i, int a, int o) {
  const float* u = C.du + i * R2; const float* Bq = C.adp + (size_t)a * PJ + 2 * R * D;
  const float* Bv = Bq + (size_t)NQ * R; float s = 0.f;
  if (o < NQ) { for (int r = 0; r < R; r++) s = fmaf(u[r], Bq[(size_t)o * R + r], s); }
  else { for (int r = 0; r < R; r++) s = fmaf(u[R + r], Bv[(size_t)(o - NQ) * R + r], s); }
  return s;
}

__device__ void decode_task(const Ctx& C, const Ph& d, int local, int step, char* smem) {
  float* sm = reinterpret_cast<float*>(smem);
  const int tid = threadIdx.x, b = C.b, pos = C.ctx0 + step;
  switch (d.kind) {
    case D_DOWN: lora_down_row(C, local, C.aid[local]); break;              // SGMV shrink: row `local` gathers its own adapter
    case L_DOWN: { const int i = C.rows[C.aoff[d.arg] + local]; lora_down_row(C, i, d.arg); break; }
    case D_QKV: {
      gemv128(C.Wqkv, D, local * 128, NQKV, C.h, b, [&](int i, int row, float v) {
        if (row < NQ) C.q[(size_t)i * NQ + row] = C.lmode == LM_SGMV ? fmaf(C.scale, lora_up(C, i, C.aid[i], row), v) : v;
        else if (row < NQ + NKV) C.kc[((size_t)i * MAXPOS + pos) * NKV + row - NQ] = __float2bfloat16(v);
        else { const int o = row - NQ - NKV;
          if (C.lmode == LM_LOOP) C.vst[(size_t)i * NKV + o] = v;
          else C.vc[((size_t)i * MAXPOS + pos) * NKV + o] = __float2bfloat16(C.lmode == LM_SGMV ? fmaf(C.scale, lora_up(C, i, C.aid[i], NQ + o), v) : v); }
      });
      break;
    }
    case L_UP: {                                               // adapter d.arg, 128 q|v output rows for each of its rows
      const int a = d.arg, n0 = C.aoff[a], cnt = C.aoff[a + 1] - n0;
      for (int j = tid; j < cnt * 128; j += THREADS) {
        const int i = C.rows[n0 + j / 128], o = local * 128 + j % 128; const float s = lora_up(C, i, a, o);
        if (o < NQ) { float* qq = C.q + (size_t)i * NQ + o; *qq = fmaf(C.scale, s, *qq); }
        else C.vc[((size_t)i * MAXPOS + pos) * NKV + o - NQ] = __float2bfloat16(fmaf(C.scale, s, C.vst[(size_t)i * NKV + o - NQ]));
      }
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

// persistent loop (OFA fetch-add ticket claim, per-phase completion counters; (phase, -1) deps = previous token)
__global__ void __launch_bounds__(THREADS, 2) mlora_kernel(const Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  char* smem = reinterpret_cast<char*>(dsm);
  __shared__ volatile int s_sel[4];
  const Job& J = C.dc; const int tid = threadIdx.x; int cph = 0, rstep = -1, rph = -1;
  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;
  for (;;) {
    if (tid == 0) {
      int res = 0, step = 0, ph = 0, local = 0;
      const unsigned g = atomicAdd(J.ticket, 1u);
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
            __nanosleep(64);
          }
        }
      }
      s_sel[0] = res; s_sel[1] = step; s_sel[2] = ph; s_sel[3] = local;
    }
    __syncthreads();
    const int res = s_sel[0], step = s_sel[1], ph = s_sel[2], local = s_sel[3];
    __syncthreads();
    if (res < 0) break;
    decode_task(C, J.ph[ph], local, step, smem);
    __threadfence();
    __syncthreads();
    if (tid == 0) { const int old = atomicAdd(J.done + ph, 1); if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) C.tok_done[step] = gtimer(); }
  }
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
static Ph mk(int kind, int n, int arg = 0) { Ph d; d.kind = kind; d.ntasks = n; d.arg = arg; return d; }
template <class X> static X* dalloc(size_t n) { X* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(X))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(X))); return p; }
template <class X> static X* dup(const std::vector<X>& v) { X* p = dalloc<X>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice)); return p; }
static void upload(Builder& b, int limit, int last, Job& J) {
  const int n = (int)b.ph.size(); std::vector<int> pre(n + 1, 0); for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + b.ntasks[i];
  J.nph = n; J.tps = pre[n]; J.limit = limit; J.last_ph = last;
  J.ph = dup(b.ph); J.prefix = dup(pre); J.ntasks = dup(b.ntasks); J.dep_off = dup(b.dep_off); J.dep_cnt = dup(b.dep_cnt);
  J.deps = b.deps.empty() ? dalloc<int2>(1) : dup(b.deps); J.done = dalloc<int>(n); J.ticket = dalloc<unsigned>(1);
}
template <class X> static void dump(const std::string& dir, const char* name, const X* dev, size_t n) {
  if (dir.empty()) return; std::vector<X> h(n); CK(cudaMemcpy(h.data(), dev, n * sizeof(X), cudaMemcpyDeviceToHost));
  FILE* f = fopen((dir + "/" + name).c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s/%s\n", dir.c_str(), name); exit(1); } fwrite(h.data(), sizeof(X), n, f); fclose(f);
}
static void pct(std::vector<double> v, double& p50, double& p99) { if (v.empty()) { p50 = p99 = 0; return; } std::sort(v.begin(), v.end()); p50 = v[v.size() / 2]; p99 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))]; }

int main(int argc, char** argv) {
  std::string lora_s = "sgmv", dir; int b = 8, na = 1, ctx0 = 1024, dtok = 512, ref = 0; u64 seed = 1234;
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--lora")) lora_s = argv[++i]; else if (is("--b")) b = atoi(argv[++i]); else if (is("--adapters")) na = atoi(argv[++i]);
    else if (is("--ctx")) ctx0 = atoi(argv[++i]); else if (is("--dec-tokens")) dtok = atoi(argv[++i]); else if (is("--dump")) dir = argv[++i];
    else if (!strcmp(argv[i], "--dump-ref")) ref = 1; else if (is("--seed")) seed = atoll(argv[++i]); else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  const int lmode = lora_s == "none" ? LM_NONE : lora_s == "loop" ? LM_LOOP : lora_s == "sgmv" ? LM_SGMV : -1;
  if (lmode == LM_NONE) na = 0;
  if (lmode < 0 || b < 1 || b > MAXB || (lmode != LM_NONE && (na < 1 || na > std::min(b, MAXA))) || ctx0 < 1 || dtok < 2 || ctx0 + dtok > MAXPOS) {
    fprintf(stderr, "bad args: --lora none|loop|sgmv, 1<=b<=%d, 1<=adapters<=min(b,%d), ctx+dec-tokens<=%d\n", MAXB, MAXA, MAXPOS); return 2; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); const int nsm = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(mlora_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlora_kernel, THREADS, SMEM)); occ = std::min(occ, 2);
  const int nblocks = nsm * occ;
  printf("GPU %s, %d SMs, %d blocks (%d/SM); lora %s, adapters %d, b=%d, ctx %d, %d tokens\n", prop.name, nsm, nblocks, occ, lora_s.c_str(), na, b, ctx0, dtok);

  Ctx C; memset(&C, 0, sizeof C); C.b = b; C.ctx0 = ctx0; C.lmode = lmode; C.scale = 2.f;
  auto fb = [&](size_t n, u64 s, float sc) { bf16* p = dalloc<bf16>(n); fill_bf16<<<1024, 256>>>(p, n, seed * 131 + s, sc); return p; };
  bf16* Wqkv = fb((size_t)NQKV * D, 1, 1.7320508f / sqrtf((float)D)); C.Wqkv = Wqkv;    // rows q 4096 | k 1024 | v 1024, [out][in]
  bf16* Wo = fb((size_t)D * NQ, 2, 1.7320508f / sqrtf((float)NQ)); C.Wo = Wo;
  bf16* E = fb((size_t)VOC * D, 3, 1.7320508f); C.E = E;
  // adapters: A random, B random (nonzero so every adapter changes the output); row i -> adapter i % na; rows grouped per adapter
  float* adp = dalloc<float>((size_t)std::max(na, 1) * PJ); C.adp = adp;
  for (int a = 0; a < na; a++) { fill_f32<<<256, 256>>>(adp + (size_t)a * PJ, 2 * R * D, seed * 977 + a, 1.7320508f / sqrtf((float)D));
    fill_f32<<<256, 256>>>(adp + (size_t)a * PJ + 2 * R * D, (size_t)NQV * R, seed * 1777 + a, 0.02f); }
  std::vector<int> aid(b), rows, aoff(std::max(na, 1) + 1, 0);
  for (int i = 0; i < b; i++) aid[i] = na ? i % na : 0;
  for (int a = 0; a < na; a++) { for (int i = 0; i < b; i++) if (aid[i] == a) rows.push_back(i); aoff[a + 1] = (int)rows.size(); }
  if (rows.empty()) rows.push_back(0);
  C.aid = dup(aid); C.rows = dup(rows); C.aoff = dup(aoff);
  bf16* kc = fb((size_t)MAXB * MAXPOS * NKV, 9, 1.7320508f); bf16* vc = fb((size_t)MAXB * MAXPOS * NKV, 10, 1.7320508f); C.kc = kc; C.vc = vc;
  C.h = dalloc<float>((size_t)MAXB * D); C.du = dalloc<float>(MAXB * R2); C.q = dalloc<float>((size_t)MAXB * NQ); C.vst = dalloc<float>((size_t)MAXB * NKV);
  C.att = dalloc<float>((size_t)MAXB * NQ);
  const int nhead = VOC / 128;
  C.lmax = dalloc<float>((size_t)MAXB * nhead); C.lidx = dalloc<int>((size_t)MAXB * nhead); C.lsum_part = dalloc<float>((size_t)MAXB * nhead);
  C.toks = dalloc<int>((size_t)dtok * MAXB); C.lsum = dalloc<float>((size_t)dtok * MAXB); C.logits0 = dalloc<float>((size_t)MAXB * VOC); C.tok_done = dalloc<u64>(dtok);
  CK(cudaDeviceSynchronize());
  { std::vector<bf16> e((size_t)b * D); CK(cudaMemcpy(e.data(), E + D, e.size() * 2, cudaMemcpyDeviceToHost));        // inputs = embeddings of tokens 1..b
    std::vector<float> h(e.size()); for (size_t i = 0; i < e.size(); i++) h[i] = __bfloat162float(e[i]); CK(cudaMemcpy(C.h, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  if (ref && !dir.empty()) {                                   // inputs for the PyTorch reference of token 0 (before the launch overwrites anything)
    dump(dir, "h0.f32", C.h, (size_t)b * D); dump(dir, "Wqkv.bf16", Wqkv, (size_t)NQKV * D); dump(dir, "Wo.bf16", Wo, (size_t)D * NQ); dump(dir, "E.bf16", E, (size_t)VOC * D);
    dump(dir, "adp.f32", adp, (size_t)std::max(na, 1) * PJ); dump(dir, "aid.i32", C.aid, (size_t)b);
    std::vector<bf16> hk((size_t)b * ctx0 * NKV);
    for (bf16* src : {kc, vc}) { for (int i = 0; i < b; i++) CK(cudaMemcpy(hk.data() + (size_t)i * ctx0 * NKV, src + (size_t)i * MAXPOS * NKV, (size_t)ctx0 * NKV * 2, cudaMemcpyDeviceToHost));
      FILE* f = fopen((dir + (src == kc ? "/kc.bf16" : "/vc.bf16")).c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write ref kv\n"); return 1; } fwrite(hk.data(), 2, hk.size(), f); fclose(f); }
  }

  // decode graph (per token); (phase, -1) = previous token
  Builder Bd; const int nqkv = NQKV / 128, nup = NQV / 128;
  int qkv;
  if (lmode == LM_SGMV) { const int dn = Bd.add(mk(D_DOWN, b), {make_int2(0, -1)}); qkv = Bd.add(mk(D_QKV, nqkv), {make_int2(dn, 0)}); }
  else qkv = Bd.add(mk(D_QKV, nqkv), {make_int2(0, -1)});
  int prev = qkv;
  if (lmode == LM_LOOP) for (int a = 0; a < na; a++) { const int cnt = aoff[a + 1] - aoff[a];
    const int dn = Bd.add(mk(L_DOWN, cnt, a), {make_int2(prev, 0)}); prev = Bd.add(mk(L_UP, nup, a), {make_int2(dn, 0)}); }
  const int att = Bd.add(mk(D_ATT, b * NKVH), {make_int2(prev, 0)}); const int o = Bd.add(mk(D_O, D / 128), {make_int2(att, 0)});
  const int hd = Bd.add(mk(D_HEAD, nhead), {make_int2(o, 0)}); const int nx = Bd.add(mk(D_NEXT, 1), {make_int2(hd, 0)});
  Bd.deps[Bd.dep_off[0]] = make_int2(nx, -1);                  // first phase waits for the previous token's next-token phase
  upload(Bd, dtok, nx, C.dc);
  printf("decode: %d phases, %d tasks/token\n", (int)Bd.ph.size(), C.dc.tps);
  { size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot)); printf("device memory used %.1f of %.1f GB\n", (tot - fr) / 1e9, tot / 1e9); }

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); mlora_kernel<<<nblocks, THREADS, SMEM>>>(C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<u64> td(dtok); CK(cudaMemcpy(td.data(), C.tok_done, dtok * 8, cudaMemcpyDeviceToHost));
  int done = 0; while (done < dtok && td[done]) done++;
  const int w0 = std::min(done - 1, 8);                        // skip warm-up tokens
  std::vector<double> lat; for (int t = w0 + 1; t < done; t++) lat.push_back((td[t] - td[t - 1]) * 1e-6);
  double p50, p99; pct(lat, p50, p99); const double win = done > w0 + 1 ? (td[done - 1] - td[w0]) * 1e-9 : 0.0;
  const double tps = win > 0 ? (double)(done - 1 - w0) * b / win : 0.0;
  printf("kernel %.3f s; decode: b=%d adapters %d lora %s, %d/%d token steps, steady %.1f tok/s (%.3g tok per GPU-hour), tpot p50 %.4f ms p99 %.4f ms\n",
         ms / 1e3, b, na, lora_s.c_str(), done, dtok, tps, tps * 3600, p50, p99);
  if (!dir.empty()) {
    dump(dir, "toks.i32", C.toks, (size_t)done * MAXB); dump(dir, "tok_done.u64", C.tok_done, (size_t)done); dump(dir, "lsum.f32", C.lsum, (size_t)done * MAXB);
    dump(dir, "logits0.f32", C.logits0, (size_t)b * VOC);
    FILE* f = fopen((dir + "/result.json").c_str(), "w"); if (!f) { fprintf(stderr, "cannot write result.json\n"); return 1; }
    fprintf(f, "{\"lora\": \"%s\", \"b\": %d, \"adapters\": %d, \"ctx\": %d, \"dec_tokens\": %d, \"done\": %d, \"tok_s\": %.6f, \"tok_per_gpu_hour\": %.1f, "
               "\"tpot_p50_ms\": %.6f, \"tpot_p99_ms\": %.6f, \"kernel_s\": %.6f, \"gpu\": \"%s\", \"sms\": %d}\n",
            lora_s.c_str(), b, na, ctx0, dtok, done, tps, tps * 3600, p50, p99, ms / 1e3, prop.name, nsm);
    fclose(f);
  }
  return done == dtok ? 0 : 1;
}
