// mm.cu : multi-model decode megakernel (H100, sm_90a). ONE persistent launch decodes two different real bases (Qwen3-0.6B and Qwen3-4B,
//   weights from h100/mm_prep.py at pinned HF revisions, both resident in HBM): full decoder stack per token (RMSNorm, fused q/k/v GEMV,
//   per-head q/k RMSNorm + RoPE, GQA attention over a per-layer KV cache, o-proj + residual, RMSNorm, SwiGLU gate/up GEMV, down GEMV +
//   residual, final norm, tied lm_head, greedy argmax). Each model has its own instruction queue (OFA ticket + per-phase completion
//   counters); in --mode co every block pulls a READY instruction from either queue (non-blocking peek + CAS, alternating which model it
//   tries first), so the two models' instructions interleave on the same SMs. --mode solo0 / solo1 run one model alone.
//   The prompt is fed token by token (teacher forcing) through the same path; generated tokens start at step P-1.
//   Fixed reduction order, no float atomics: per-model token ids are byte-equal between solo and co-run.
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

#include "mm_decode.cuh"
using namespace mmd;

constexpr int SMEM = MAXHPK * SPOS * 4;            // 64 KB: attention scores [hpk][pos]; also gate/up and lm_head partials
static_assert(MAXB * 128 * 4 <= SMEM, "partials fit smem");

struct Job { int nph, tps, limit, last_ph; const Ph* ph; const int *prefix, *ntasks, *dep_off, *dep_cnt; const int2* deps; int* done; unsigned* ticket; };
struct Ctx { Job J[2]; Model M[2]; int mode; };

__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ __forceinline__ bool ph_ready(const Job& J, int step, int ph) {
  for (int k = 0; k < J.dep_cnt[ph]; k++) { const int2 e = J.deps[J.dep_off[ph] + k]; const int need = J.ntasks[e.x] * (step + 1 + e.y);
    if (need > 0 && ld_acq(J.done + e.x) < need) return false; }
  return true;
}
__device__ __forceinline__ void run_task(const Ctx& C, int m, int ph, int local, int step, char* smem) {
  const Job& J = C.J[m];
  decode_task(C.M[m], J.ph[ph], local, step, smem);
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) { const int old = atomicAdd(J.done + ph, 1); if (ph == J.last_ph && old == J.ntasks[ph] * (step + 1) - 1) C.M[m].tok_done[step] = gtimer(); }
  __syncthreads();
}
// non-blocking claim of model m's queue head: 1 = ran, 0 = not ready, 2 = queue done
__device__ int try_claim(const Ctx& C, int m, char* smem) {
  __shared__ volatile int s_l[4];
  const Job& J = C.J[m];
  if (threadIdx.x == 0) {
    int st = 0, step = 0, ph = 0, local = 0;
    const unsigned g = *reinterpret_cast<volatile unsigned*>(J.ticket);
    if (g >= (unsigned)J.limit * (unsigned)J.tps) st = 2;
    else {
      step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);
      int lo = 0, hi = J.nph - 1; while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (J.prefix[mid] <= r) lo = mid; else hi = mid - 1; }
      ph = lo; local = r - J.prefix[ph];
      if (ph_ready(J, step, ph) && atomicCAS(J.ticket, g, g + 1) == g) st = 1;
    }
    s_l[0] = st; s_l[1] = step; s_l[2] = ph; s_l[3] = local;
  }
  __syncthreads();
  const int st = s_l[0], step = s_l[1], ph = s_l[2], local = s_l[3];
  __syncthreads();
  if (st == 1) run_task(C, m, ph, local, step, smem);
  return st;
}
__device__ void job_loop(const Ctx& C, int m, char* smem) {
  __shared__ volatile int s_sel[4];
  const Job& J = C.J[m]; const int tid = threadIdx.x; int cph = 0;
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
        while (!ph_ready(J, step, ph)) __nanosleep(64);
      }
      s_sel[0] = res; s_sel[1] = step; s_sel[2] = ph; s_sel[3] = local;
    }
    __syncthreads();
    const int res = s_sel[0], step = s_sel[1], ph = s_sel[2], local = s_sel[3];
    __syncthreads();
    if (res < 0) break;
    run_task(C, m, ph, local, step, smem);
  }
}
__global__ void __launch_bounds__(THREADS, 2) mm_kernel(const Ctx C) {
  extern __shared__ __align__(128) unsigned char dsm[];
  char* smem = reinterpret_cast<char*>(dsm);
  if (C.mode < 2) { job_loop(C, C.mode, smem); return; }
  for (unsigned it = blockIdx.x;; it++) {
    const int f = it & 1;
    const int a = try_claim(C, f, smem);
    if (a == 1) continue;
    const int c = try_claim(C, 1 - f, smem);
    if (c == 1) continue;
    if (a == 2 && c == 2) break;
    if (threadIdx.x == 0) __nanosleep(64);
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------ host
struct Builder {
  std::vector<Ph> ph; std::vector<int> dep_off, dep_cnt, ntasks; std::vector<int2> deps;
  int add(Ph d, std::vector<int2> after) { dep_off.push_back((int)deps.size()); for (auto a : after) deps.push_back(a); dep_cnt.push_back((int)after.size());
    ph.push_back(d); ntasks.push_back(d.ntasks); return (int)ph.size() - 1; }
};
template <class X> static X* dalloc(size_t n) { X* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(X))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(X))); return p; }
template <class X> static X* dup(const std::vector<X>& v) { X* p = dalloc<X>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice)); return p; }
static void upload(Builder& b, int limit, int last, Job& J) {
  const int n = (int)b.ph.size(); std::vector<int> pre(n + 1, 0); for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + b.ntasks[i];
  J.nph = n; J.tps = pre[n]; J.limit = limit; J.last_ph = last;
  J.ph = dup(b.ph); J.prefix = dup(pre); J.ntasks = dup(b.ntasks); J.dep_off = dup(b.dep_off); J.dep_cnt = dup(b.dep_cnt); J.deps = dup(b.deps);
  J.done = dalloc<int>(n); J.ticket = dalloc<unsigned>(1);
}
template <class X> static void dump(const std::string& dir, const std::string& name, const X* dev, size_t n) {
  if (dir.empty()) return; std::vector<X> h(n); CK(cudaMemcpy(h.data(), dev, n * sizeof(X), cudaMemcpyDeviceToHost));
  FILE* f = fopen((dir + "/" + name).c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s/%s\n", dir.c_str(), name.c_str()); exit(1); } fwrite(h.data(), sizeof(X), n, f); fclose(f);
}
static void pct(std::vector<double> v, double& p50, double& p99) { if (v.empty()) { p50 = p99 = 0; return; } std::sort(v.begin(), v.end()); p50 = v[v.size() / 2]; p99 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))]; }

// model dir (h100/mm_prep.py): meta.txt "H L NH NKV I V P eps theta", weights.bf16 = emb[V][H] fnorm[H] then per layer
//   in_norm[H] qkv[(NH+2NKV)*128][H] q_norm[128] k_norm[128] o[H][NH*128] post_norm[H] gu[2I][H] (64 gate / 64 up row blocks) down[H][I];
//   prompt.i32 = [MAXB][P] token ids
static void load_model(const std::string& dir, int b, int T, Model& M, Builder& B, int& limit, int& last) {
  FILE* f = fopen((dir + "/meta.txt").c_str(), "r"); if (!f) { fprintf(stderr, "cannot open %s/meta.txt\n", dir.c_str()); exit(1); }
  if (fscanf(f, "%d %d %d %d %d %d %d %f %f", &M.H, &M.L, &M.NH, &M.NKV, &M.I, &M.V, &M.P, &M.eps, &M.theta) != 9) { fprintf(stderr, "bad meta.txt\n"); exit(1); }
  fclose(f);
  if (M.NH % M.NKV || M.NH / M.NKV > MAXHPK || M.H % 128 || M.I % 64 || M.V % 128 || M.P + T > SPOS) { fprintf(stderr, "unsupported model shape in %s\n", dir.c_str()); exit(1); }
  M.b = b; M.NQ = M.NH * HD; M.NK = M.NKV * HD; M.nqkv = M.NQ + 2 * M.NK; M.maxpos = M.P + T;
  const size_t H = M.H;
  M.o_in = 0; M.o_qkv = H; M.o_qn = M.o_qkv + (size_t)M.nqkv * H; M.o_kn = M.o_qn + HD; M.o_o = M.o_kn + HD; M.o_post = M.o_o + H * M.NQ;
  M.o_gu = M.o_post + H; M.o_d = M.o_gu + 2 * (size_t)M.I * H; M.lst = M.o_d + H * M.I;
  const size_t total = (size_t)M.V * H + H + (size_t)M.L * M.lst;
  bf16* w = dalloc<bf16>(total);
  FILE* wf = fopen((dir + "/weights.bf16").c_str(), "rb"); if (!wf) { fprintf(stderr, "cannot open %s/weights.bf16\n", dir.c_str()); exit(1); }
  std::vector<bf16> buf(1 << 26); size_t off = 0;
  while (off < total) { const size_t n = std::min(buf.size(), total - off); if (fread(buf.data(), 2, n, wf) != n) { fprintf(stderr, "short weights file %s\n", dir.c_str()); exit(1); }
    CK(cudaMemcpy(w + off, buf.data(), n * 2, cudaMemcpyHostToDevice)); off += n; }
  fclose(wf);
  M.emb = w; M.fnorm = w + (size_t)M.V * H; M.w = M.fnorm + H;
  std::vector<int> pr((size_t)MAXB * M.P); FILE* pf = fopen((dir + "/prompt.i32").c_str(), "rb");
  if (!pf || fread(pr.data(), 4, pr.size(), pf) != pr.size()) { fprintf(stderr, "cannot read %s/prompt.i32\n", dir.c_str()); exit(1); } fclose(pf);
  M.prompt = dup(pr);
  M.h = dalloc<float>((size_t)b * H); M.xn = dalloc<float>((size_t)b * H); M.q = dalloc<float>((size_t)b * M.NQ); M.kraw = dalloc<float>((size_t)b * M.NK);
  M.att = dalloc<float>((size_t)b * M.NQ); M.act = dalloc<float>((size_t)b * M.I);
  const int nh = M.V / 128; M.lmax = dalloc<float>((size_t)b * nh); M.lidx = dalloc<int>((size_t)b * nh); M.lsum_part = dalloc<float>((size_t)b * nh);
  M.logitsP = dalloc<float>((size_t)b * M.V); limit = M.P + T - 1; M.toks = dalloc<int>((size_t)limit * MAXB); M.tok_done = dalloc<u64>(limit);
  M.kc = dalloc<bf16>((size_t)M.L * b * M.maxpos * M.NK); M.vc = dalloc<bf16>((size_t)M.L * b * M.maxpos * M.NK);
  { std::vector<bf16> e((size_t)b * H); for (int i = 0; i < b; i++) CK(cudaMemcpy(e.data() + (size_t)i * H, M.emb + (size_t)pr[(size_t)i * M.P] * H, H * 2, cudaMemcpyDeviceToHost));
    std::vector<float> h(e.size()); for (size_t i = 0; i < e.size(); i++) h[i] = __bfloat162float(e[i]); CK(cudaMemcpy(M.h, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  auto mk = [](int kind, int n, int layer, int norm) { Ph d; d.kind = kind; d.ntasks = n; d.layer = layer; d.norm = norm; return d; };
  int prev = -1;
  auto add = [&](Ph d) { prev = B.add(d, {prev < 0 ? make_int2(0, -1) : make_int2(prev, 0)}); };
  for (int l = 0; l < M.L; l++) {
    add(mk(K_NORM, b, l, 0)); add(mk(K_QKV, M.nqkv / 128, l, 0)); add(mk(K_ATT, b * M.NKV, l, 0)); add(mk(K_O, M.H / 128, l, 0));
    add(mk(K_NORM, b, l, 1)); add(mk(K_GU, 2 * M.I / 128, l, 0)); add(mk(K_DN, M.H / 128, l, 0));
  }
  add(mk(K_NORM, b, 0, 2)); add(mk(K_HEAD, M.V / 128, 0, 0)); add(mk(K_NEXT, 1, 0, 0));
  last = prev; B.deps[B.dep_off[0]] = make_int2(last, -1);     // layer-0 norm waits for the previous token's next-token phase
}

int main(int argc, char** argv) {
  std::string mode_s = "co", dirs[2], dir; int b = 1, T[2] = {256, 256};
  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--mode")) mode_s = argv[++i]; else if (is("--model0")) dirs[0] = argv[++i]; else if (is("--model1")) dirs[1] = argv[++i];
    else if (is("--b")) b = atoi(argv[++i]); else if (is("--tokens0")) T[0] = atoi(argv[++i]); else if (is("--tokens1")) T[1] = atoi(argv[++i]);
    else if (is("--dump")) dir = argv[++i]; else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  const int mode = mode_s == "solo0" ? 0 : mode_s == "solo1" ? 1 : mode_s == "co" ? 2 : -1;
  if (mode < 0 || b < 1 || b > MAXB || T[0] < 2 || T[1] < 2 || (mode != 1 && dirs[0].empty()) || (mode != 0 && dirs[1].empty())) {
    fprintf(stderr, "usage: --mode solo0|solo1|co --model0 DIR --model1 DIR --b 1..%d --tokens0 N --tokens1 N [--dump DIR]\n", MAXB); return 2; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); const int nsm = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(mm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mm_kernel, THREADS, SMEM)); occ = std::min(occ, 2);
  const int nblocks = nsm * occ;
  Ctx C; memset(&C, 0, sizeof C); C.mode = mode;
  for (int m = 0; m < 2; m++) {
    if ((mode == 0 && m == 1) || (mode == 1 && m == 0)) continue;
    Builder B; int limit = 0, last = 0; load_model(dirs[m], b, T[m], C.M[m], B, limit, last); upload(B, limit, last, C.J[m]);
    printf("model%d %s: H %d L %d heads %d/%d I %d V %d, prompt %d + %d tokens, %d phases, %d tasks/token\n", m, dirs[m].c_str(), C.M[m].H, C.M[m].L,
           C.M[m].NH, C.M[m].NKV, C.M[m].I, C.M[m].V, C.M[m].P, T[m], (int)B.ph.size(), C.J[m].tps);
  }
  CK(cudaDeviceSynchronize());
  { size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot)); printf("GPU %s, %d SMs, %d blocks (%d/SM); mode %s, b=%d; device memory used %.1f of %.1f GB\n",
                                                          prop.name, nsm, nblocks, occ, mode_s.c_str(), b, (tot - fr) / 1e9, tot / 1e9); }
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); mm_kernel<<<nblocks, THREADS, SMEM>>>(C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); printf("kernel %.3f s\n", ms / 1e3);
  std::string js = "{\"mode\": \"" + mode_s + "\", \"b\": " + std::to_string(b) + ", \"gpu\": \"" + prop.name + "\", \"kernel_s\": " + std::to_string(ms / 1e3) + ", \"models\": {";
  int ok = 1;
  for (int m = 0; m < 2; m++) {
    if (!C.J[m].limit) continue;
    const Model& M = C.M[m]; const int limit = C.J[m].limit;
    std::vector<u64> td(limit); CK(cudaMemcpy(td.data(), M.tok_done, limit * 8, cudaMemcpyDeviceToHost));
    int done = 0; while (done < limit && td[done]) done++;
    if (done != limit) ok = 0;
    std::vector<double> lat; for (int s = M.P; s < done; s++) lat.push_back((td[s] - td[s - 1]) * 1e-6);
    double p50, p99; pct(lat, p50, p99);
    const double win = done > M.P ? (td[done - 1] - td[M.P - 1]) * 1e-9 : 0; const double tps = win > 0 ? (double)(done - M.P) * b / win : 0;
    printf("model%d: %d/%d steps, %d generated tokens/row, steady %.1f tok/s (%.4g tok per GPU-hour), tpot p50 %.4f ms p99 %.4f ms\n",
           m, done, limit, done - M.P + 1, tps, tps * 3600, p50, p99);
    char buf[512]; snprintf(buf, sizeof buf, "%s\"%d\": {\"dir\": \"%s\", \"P\": %d, \"steps\": %d, \"done\": %d, \"tok_s\": %.6f, \"tpot_p50_ms\": %.6f, \"tpot_p99_ms\": %.6f, "
                            "\"start_ns\": %llu, \"end_ns\": %llu}", js.back() == '{' ? "" : ", ", m, dirs[m].c_str(), M.P, limit, done, tps, p50, p99,
                            done >= M.P ? td[M.P - 1] : 0ull, done ? td[done - 1] : 0ull);
    js += buf;
    if (!dir.empty()) { const std::string s = std::to_string(m);
      dump(dir, "toks" + s + ".i32", M.toks, (size_t)done * MAXB); dump(dir, "tok_done" + s + ".u64", M.tok_done, (size_t)done); dump(dir, "logitsP" + s + ".f32", M.logitsP, (size_t)b * M.V); }
  }
  js += "}}\n";
  if (!dir.empty()) { FILE* f = fopen((dir + "/result.json").c_str(), "w"); if (!f) { fprintf(stderr, "cannot write result.json\n"); return 1; } fputs(js.c_str(), f); fclose(f); }
  return ok ? 0 : 1;
}
