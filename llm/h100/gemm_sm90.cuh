// gemm_sm90.cuh -- Hopper (sm_90a) GEMM tile backend for the open-flapping-airplanes megakernel (Nodus fork).
// Same contract as llm/gemm/gemm.cuh (one task = one 128x128 output tile, all 256 threads of a block, NN/NT/TN layouts, same epilogues),
// but TMA 2D loads (128B swizzle) into a 3-stage mbarrier pipeline and wgmma.mma_async m64n128k16 bf16->fp32: warpgroup 0 computes
// rows 0..63, warpgroup 1 rows 64..127. Descriptor strides follow the B300-verified tcgen05 kit (gemm_sm100_v2.cuh): K-major LBO 16 /
// SBO 1024, MN-major LBO 8192 / SBO 1024, k16 advance 32 B (K-major) or 2048 B (MN-major), same TMA boxes ({64,128} K-major, {64,64} MN-major).
// Compiled for anything other than sm_90a it falls back to gemm::gemm_tile (mma.sync).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../gemm/gemm.cuh"

namespace gemm {
namespace sm90 {

constexpr int BK = 64, STAGES = 3;
constexpr int A_BYTES = 128 * BK * 2, B_BYTES = 128 * BK * 2, STAGE_BYTES = A_BYTES + B_BYTES;

struct ArgsWG : Args {
  const CUtensorMap* tmA;
  const CUtensorMap* tmB;
};

__host__ __device__ constexpr int smem_bytes() { return STAGES * STAGE_BYTES + 1024 + 64; }

__host__ inline bool args_ok(const ArgsWG& p) { return gemm::args_ok(p) && p.K % BK == 0 && p.tmA && p.tmB; }

inline const CUtensorMap* make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t ld_elems, uint32_t box_inner, uint32_t box_outer) {
  typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                               const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static EncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    if (!fn) { fprintf(stderr, "gemm_sm90: cuTensorMapEncodeTiled not found\n"); exit(1); }
    enc = (EncodeFn)fn;
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {ld_elems * 2};
  cuuint32_t box[2] = {box_inner, box_outer}, es[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "gemm_sm90: cuTensorMapEncodeTiled failed (%d)\n", (int)r); exit(1); }
  CUtensorMap* d; cudaMalloc(&d, sizeof(CUtensorMap));
  cudaMemcpy(d, &m, sizeof m, cudaMemcpyHostToDevice);
  return d;
}

//   A K-major (NN, NT): [M][K] -> {K, M} box {64,128}      A MN-major (TN): [K][M] -> {M, K} box {64,64} (two boxes per 128 rows)
//   B K-major (NT):     [N][K] -> {K, N} box {64,128}      B MN-major (NN, TN): [K][N] -> {N, K} box {64,64} (two boxes per 128 cols)
inline void prepare(ArgsWG& p, int layout) {
  const bool akc = layout != TN, bkc = layout == NT;
  p.tmA = akc ? make_map(p.A, p.K, p.M, p.lda, 64, 128) : make_map(p.A, p.M, p.K, p.lda, 64, 64);
  p.tmB = bkc ? make_map(p.B, p.K, p.N, p.ldb, 64, 128) : make_map(p.B, p.N, p.K, p.ldb, 64, 64);
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_inval(uint32_t a) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t mbar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               ::"r"(dst), "l"(tm), "r"(mbar), "r"(c0), "r"(c1) : "memory");
}
// SM90 matrix descriptor: start>>4 [0,14), LBO>>4 [16,30), SBO>>4 [32,46), base offset 0 (1024-B aligned stages), layout 1 = 128B swizzle [62,64)
__device__ __forceinline__ uint64_t desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void acc_fence(float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; i++) asm volatile("" : "+f"(d[i])::"memory");
}
template <int TA, int TB>
__device__ __forceinline__ void wgmma128(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, %67, %68;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(da), "l"(db), "n"(1), "n"(TA), "n"(TB));
}

template <int LAYOUT, int EPI>
__device__ __forceinline__ void tile_wg(const ArgsWG& p, int tile_m, int tile_n, char* smem) {
  constexpr bool A_MN = (LAYOUT == TN), B_MN = (LAYOUT != NT);
  const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (raw + 1023u) & ~1023u;
  const uint32_t meta = sbase + STAGES * STAGE_BYTES;
  auto full = [&](int s) { return meta + 8u * s; };
  const int m0 = tile_m * 128, n0 = tile_n * 128, KT = p.K / BK;

  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");     // earlier tasks wrote this smem through the generic proxy
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) mbar_init(full(s), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto issue = [&](int kt) {   // thread 0 only
    const int s = kt % STAGES; const uint32_t sA = sbase + s * STAGE_BYTES, sB = sA + A_BYTES; const int k0 = kt * BK;
    mbar_expect_tx(full(s), STAGE_BYTES);
    if (!A_MN) tma_load_2d(sA, p.tmA, full(s), k0, m0);
    else { tma_load_2d(sA, p.tmA, full(s), m0, k0); tma_load_2d(sA + 8192, p.tmA, full(s), m0 + 64, k0); }
    if (!B_MN) tma_load_2d(sB, p.tmB, full(s), k0, n0);
    else { tma_load_2d(sB, p.tmB, full(s), n0, k0); tma_load_2d(sB + 8192, p.tmB, full(s), n0 + 64, k0); }
  };
  if (tid == 0) for (int kt = 0; kt < STAGES && kt < KT; kt++) issue(kt);

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; i++) d[i] = 0.f;
  for (int kt = 0; kt < KT; kt++) {
    const int s = kt % STAGES;
    mbar_wait(full(s), (kt / STAGES) & 1);
    const uint32_t sA = sbase + s * STAGE_BYTES + wg * 8192, sB = sbase + s * STAGE_BYTES + A_BYTES;
    acc_fence(d);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < BK / 16; kk++) {
      const uint64_t da = A_MN ? desc(sA + kk * 2048, 8192, 1024) : desc(sA + kk * 32, 16, 1024);
      const uint64_t db = B_MN ? desc(sB + kk * 2048, 8192, 1024) : desc(sB + kk * 32, 16, 1024);
      wgmma128<A_MN ? 1 : 0, B_MN ? 1 : 0>(d, da, db);
    }
    wg_commit();
    wg_wait<1>();                                   // the wgmmas of tile kt-1 are done: its slot can be refilled
    acc_fence(d);
    __syncthreads();
    if (tid == 0 && kt >= 1 && kt - 1 + STAGES < KT) issue(kt - 1 + STAGES);
  }
  wg_wait<0>();
  acc_fence(d);
  // accumulator fragment of m64nNk16: d[4i+{0,1}] -> (row 16w + l/4, col 8i + 2(l%4) + {0,1}), d[4i+{2,3}] -> row + 8
  const int r0 = m0 + wg * 64 + w * 16 + (l >> 2), c0 = n0 + 2 * (l & 3);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    epi_store<EPI>(p, r0, c0 + 8 * i, d[4 * i], d[4 * i + 1]);
    epi_store<EPI>(p, r0 + 8, c0 + 8 * i, d[4 * i + 2], d[4 * i + 3]);
  }
  __syncthreads();
  if (tid == 0) for (int s = 0; s < STAGES; s++) mbar_inval(full(s));
  __syncthreads();   // smem free for the caller's next task
}
#endif

template <int LAYOUT, int EPI>
__device__ __forceinline__ void tile(const ArgsWG& p, int tile_m, int tile_n, char* smem) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  tile_wg<LAYOUT, EPI>(p, tile_m, tile_n, smem);
#else
  gemm_tile<LAYOUT, EPI>(p, tile_m, tile_n, smem);
#endif
}

}  // namespace sm90
}  // namespace gemm
