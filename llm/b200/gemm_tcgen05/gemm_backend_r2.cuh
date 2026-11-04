// gemm_backend_r2.cuh -- round-2 backend switch.  -DOFA_GEMM_BACKEND=0 mma.sync (llm/gemm/gemm.cuh) | 1 = v1 tcgen05 (128x128, kept for reference) | 2 = v2 (gemm_sm100_v2.cuh).
// Uniform interface used by ofa_r2.cu and test_gemm_r2.cu:
//   tg::Args, tg::args_ok(p), tg::prepare(p, layout, wide, nsub)  (host; no-op for backend 0), tg::choose(M,N,slots,&wide,&nsub)  (host policy; backend 0/1: wide=0,nsub=1),
//   tg::ntasks(p) (number of scheduler tasks of a GEMM phase), tg::RAW_TASK (backend 2: tile() takes the TASK index as 3rd arg and 0 as 4th; else (tile_m, tile_n)),
//   tg::tile<L,E>(p, a, b, smem), tg::smem_bytes(), tg::block_begin()/block_end() (once per persistent block: persistent TMEM allocation with -DG2_PERSIST_TMEM).
#pragma once
#ifndef OFA_GEMM_BACKEND
#define OFA_GEMM_BACKEND 2
#endif
#if OFA_GEMM_BACKEND == 3
// Nodus fork: Hopper wgmma + TMA backend (llm/h100/gemm_sm90.cuh); one task = one 128x128 tile like the mma.sync backend
#include "../../h100/gemm_sm90.cuh"
namespace tg {
using Args = gemm::sm90::ArgsWG;
constexpr bool RAW_TASK = false;
inline bool args_ok(const Args& p) { return gemm::sm90::args_ok(p); }
inline void prepare(Args& p, int layout, int = 0, int = 1) { gemm::sm90::prepare(p, layout); }
inline void choose(int, int, int, int* wide, int* nsub, int = -1, int = 0) { *wide = 0; *nsub = 1; }
inline int ntasks(const Args& p) { return (p.M / 128) * (p.N / 128); }
__host__ __device__ constexpr int smem_bytes() { return gemm::sm90::smem_bytes(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int tm, int tn, char* smem) { gemm::sm90::tile<L, E>(p, tm, tn, smem); }
__device__ __forceinline__ void block_begin() {}
__device__ __forceinline__ void block_end() {}
}  // namespace tg
#elif OFA_GEMM_BACKEND == 2
#include "gemm_sm100_v2.cuh"
namespace tg {
using Args = gemm::sm100v2::ArgsTC;
constexpr bool RAW_TASK = true;
inline bool args_ok(const Args& p) { return gemm::sm100v2::args_ok(p); }
inline void prepare(Args& p, int layout, int wide = 0, int nsub = 1) { gemm::sm100v2::prepare(p, layout, wide, nsub); }
inline void choose(int M, int N, int slots, int* wide, int* nsub, int fw = -1, int fn = 0) { gemm::sm100v2::choose(M, N, slots, wide, nsub, fw, fn); }
inline int ntasks(const Args& p) { return gemm::sm100v2::ntasks(p); }
__host__ __device__ constexpr int smem_bytes() { return gemm::sm100v2::smem_bytes(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int a, int b, char* smem) { gemm::sm100v2::tile<L, E>(p, a, b, smem); }
__device__ __forceinline__ void block_begin() { gemm::sm100v2::blk_begin(); }
__device__ __forceinline__ void block_end() { gemm::sm100v2::blk_end(); }
}  // namespace tg
#elif OFA_GEMM_BACKEND == 1
#include "gemm_sm100.cuh"
namespace tg {
using Args = gemm::sm100::ArgsTC;
constexpr bool RAW_TASK = false;
inline bool args_ok(const Args& p) { return gemm::sm100::args_ok(p); }
inline void prepare(Args& p, int layout, int = 0, int = 1) { gemm::sm100::prepare(p, layout); }
inline void choose(int, int, int, int* wide, int* nsub, int = -1, int = 0) { *wide = 0; *nsub = 1; }
inline int ntasks(const Args& p) { return (p.M / 128) * (p.N / 128); }
__host__ __device__ constexpr int smem_bytes() { return gemm::sm100::smem_bytes(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int tm, int tn, char* smem) { gemm::sm100::tile<L, E>(p, tm, tn, smem); }
__device__ __forceinline__ void block_begin() {}
__device__ __forceinline__ void block_end() {}
}  // namespace tg
#else
#include "../../gemm/gemm.cuh"
namespace tg {
using Args = gemm::Args;
constexpr bool RAW_TASK = false;
inline bool args_ok(const Args& p) { return gemm::args_ok(p); }
inline void prepare(Args&, int, int = 0, int = 1) {}
inline void choose(int, int, int, int* wide, int* nsub, int = -1, int = 0) { *wide = 0; *nsub = 1; }
inline int ntasks(const Args& p) { return (p.M / 128) * (p.N / 128); }
__host__ __device__ constexpr int smem_bytes() { return gemm::smem_bytes<>(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int tm, int tn, char* smem) { gemm::gemm_tile<L, E>(p, tm, tn, smem); }
__device__ __forceinline__ void block_begin() {}
__device__ __forceinline__ void block_end() {}
}  // namespace tg
#endif
