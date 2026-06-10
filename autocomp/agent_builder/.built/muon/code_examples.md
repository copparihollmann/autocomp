## example_gemm_simt_tiled.hpp

SUMMARY: Demonstrates a tiled BF16 GEMM kernel for the Muon GPU using SIMT execution, showcasing shared memory tiling, ILP memory loading, thread-tile decomposition, and BF16 packed arithmetic with mu_barrier/mu_fence_smem synchronization primitives.

```cpp
#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>
#include <stdint.h>

#ifndef GEMM_SIMT_NUM_WARPS
#define GEMM_SIMT_NUM_WARPS 4
#endif

#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(GEMM_SIMT_NUM_WARPS)
#define THREADBLOCK_SIZE MU_BLOCK_SIZE(GEMM_SIMT_NUM_WARPS)
#define ILP_MEM 2

// all numbers below in number of BF16 elements
#define BK 32
#define BM 16
#define BN 32
#ifndef TM
#define TM 1
#endif
#ifndef TN
#define TN 2
#endif
#define TBM (BM / TM)
#define TBN (BN / TN)
#define BLOCK_SIZE (BM * BN)

#define A_WORDS (BM * BK / 2)
#define B_WORDS (BK * BN / 2)
#define C_TILES (TBM * TBN)
#define A_ITERS (A_WORDS / THREADBLOCK_SIZE)
#define B_ITERS (B_WORDS / THREADBLOCK_SIZE)
#define C_ITERS (C_TILES / THREADBLOCK_SIZE)
#define A_FULL_ITERS ((A_ITERS / ILP_MEM) * ILP_MEM)
#define B_FULL_ITERS ((B_ITERS / ILP_MEM) * ILP_MEM)

static_assert(BM % TM == 0, "Thread tile M must evenly divide the CTA tile M");
static_assert(BN % TN == 0, "Thread tile N must evenly divide the CTA tile N");
static_assert(TN % 2 == 0, "Thread tile N must contain packed BF16 pairs");
static_assert(A_WORDS % THREADBLOCK_SIZE == 0, "A tile load must evenly divide across the threadblock");
static_assert(B_WORDS % THREADBLOCK_SIZE == 0, "B tile load must evenly divide across the threadblock");
static_assert(C_TILES >= THREADBLOCK_SIZE, "C tile work must cover the threadblock");
static_assert(C_TILES % THREADBLOCK_SIZE == 0, "C tile work must evenly divide across the threadblock");

extern "C" uint32_t __mu_num_warps = GEMM_SIMT_NUM_WARPS;

struct GEMMArgs {
  __global uint32_t* A;
  __global uint32_t* B;
  __global uint32_t* C;
  uint32_t M;
  uint32_t K;
  uint32_t N;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

// C = A * B where A is MxK, B is KxN, C is MxN (all bf16)
static inline void gemm(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GEMMArgs*>(arg);
  uint32_t tid = tid_in_threadblock;
  uint32_t total_blocks = args->M * args->N / BLOCK_SIZE;
  uint32_t blocks_per_cluster = total_blocks / MU_NUM_CLUSTERS;
  uint32_t block_N = args->N / BN;

  uint32_t M = args->M;
  uint32_t N = args->N;
  uint32_t K = args->K;
  __global uint32_t *A = args->A;
  __global uint32_t *B = args->B;
  __global uint32_t *C = args->C;
  __shared uint32_t *As = sdata;
  __shared uint32_t *Bs = sdata + BM * BK / 2;

  for (uint32_t c_block = 0; c_block < blocks_per_cluster; c_block++) {
    uint32_t block_idx = threadblock_id * blocks_per_cluster + c_block;
    uint32_t block_x_idx = block_idx / block_N;
    uint32_t block_y_idx = block_idx % block_N;

    // clear out accum
    _Float16 acc[C_ITERS][TM * TN];
    #pragma unroll
    for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
      #pragma unroll
      for (uint32_t i = 0; i < TM*TN; i++) acc[c_iter][i] = 0;
    }

    // stream across K
    for (uint32_t k_block = 0; k_block < K; k_block += BK) {
      // load A tile to smem with ILP
      #pragma unroll
      for (uint32_t base = 0; base < A_FULL_ITERS; base += ILP_MEM) {
        uint32_t a_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2));
          uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
          a_val[u] = A[A_x * (K / 2) + A_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          As[elem_idx] = a_val[u];
        }
      }
      #if (A_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = A_FULL_ITERS; block < A_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2));
        uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
        As[elem_idx] = A[A_x * (K / 2) + A_y];
      }
      #endif

      // load B tile to smem with ILP
      #pragma unroll
      for (uint32_t base = 0; base < B_FULL_ITERS; base += ILP_MEM) {
        uint32_t b_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2));
          uint32_t B_x = k_block + (elem_idx / (BN / 2));
          b_val[u] = B[B_x * (N / 2) + B_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          Bs[elem_idx] = b_val[u];
        }
      }
      #if (B_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = B_FULL_ITERS; block < B_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2));
        uint32_t B_x = k_block + (elem_idx / (BN / 2));
        Bs[elem_idx] = B[B_x * (N / 2) + B_y];
      }
      #endif

      // synchronize smem writes before compute
      mu_fence_smem();
      mu_barrier(0, BLOCK_NUM_WARPS);

      // compute thread-tile GEMM over BK slice
      #pragma unroll
      for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
        uint32_t c_tile = tid + c_iter * THREADBLOCK_SIZE;
        uint32_t thread_x = c_tile / TBN;
        uint32_t thread_y = c_tile % TBN;
        for (uint32_t k = 0; k < BK / 2; k++) {
          for (uint32_t i = 0; i < TM; i++) {
            uint32_t a_idx = thread_x * TM + i;
            auto [a0, a1] = unpack_bf16x2(As[a_idx * BK / 2 + k]);
            for (uint32_t j = 0; j < TN / 2; j++) {
              uint32_t b_idx = thread_y * TN / 2 + j;
              auto [b00, b10] = unpack_bf16x2(Bs[2*k * (BN / 2) + b_idx]);
              auto [b01, b11] = unpack_bf16x2(Bs[(2*k + 1) * (BN / 2) + b_idx]);
              acc[c_iter][i * TN + 2 * j]     += a0 * b00 + a1 * b01;
              acc[c_iter][i * TN + 2 * j + 1] += a0 * b10 + a1 * b11;
            }
          }
        }
      }

      mu_barrier(0, BLOCK_NUM_WARPS);
    }

    // store C tile from registers to global memory
    #pragma unroll
    for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
      uint32_t c_tile = tid + c_iter * THREADBLOCK_SIZE;
      uint32_t thread_x = c_tile / TBN;
      uint32_t thread_y = c_tile % TBN;
      uint32_t c_row = block_x_idx * BM + thread_x * TM;
      uint32_t c_col = block_y_idx * BN / 2 + thread_y * TN / 2;
      #pragma unroll
      for (uint32_t i = 0; i < TM; i++) {
        uint32_t c_x = c_row + i;
        for (uint32_t j = 0; j < TN / 2; j++) {
          uint32_t c_y = c_col + j;
          C[c_x * (N / 2) + c_y] = pack_bf16x2(
            acc[c_iter][i * TN + 2*j],
            acc[c_iter][i * TN + 2*j + 1]
          );
        }
      }
    }
  }
}
```

## example_softmax_warp_reduction.hpp

SUMMARY: Demonstrates a BF16 softmax kernel for the Muon GPU using warp/block reductions with shared memory, ILP (instruction-level parallelism), and mu_schedule/mu_barrier/mu_fence_smem intrinsics across a three-pass algorithm (find max, compute denominator, normalize).

```cpp
#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>
#include <math.h>
#include <stdint.h>

#ifndef SOFTMAX_NUM_WARPS
#define SOFTMAX_NUM_WARPS 4
#endif

#ifndef SOFTMAX_ILP
#define SOFTMAX_ILP 4
#endif

#define DOUBLE_BLOCK_SIZE MU_DOUBLE_BLOCK_SIZE(SOFTMAX_NUM_WARPS)
#define BLOCK_SIZE MU_BLOCK_SIZE(SOFTMAX_NUM_WARPS)
#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(SOFTMAX_NUM_WARPS)
#define THREAD_DIV (MU_NUM_MAX_WARPS / SOFTMAX_NUM_WARPS)

extern "C" uint32_t __mu_num_warps = SOFTMAX_NUM_WARPS;

struct SoftmaxArgs {
  __global uint32_t* x;
  uint32_t rows;
  uint32_t cols;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

template <uint32_t MAX_STRIDE>
static inline void reduce_max(__shared uint32_t *buf_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      _Float16 a = as_bf16((uint16_t)buf_sdata[tid]);
      _Float16 b = as_bf16((uint16_t)buf_sdata[tid + (stride >> 1)]);
      buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)fmaxf(a, b));
    }
  }
}

template <uint32_t MAX_STRIDE>
static inline void reduce_sum(__shared uint32_t *buf_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      _Float16 a = as_bf16((uint16_t)buf_sdata[tid]);
      _Float16 b = as_bf16((uint16_t)buf_sdata[tid + (stride >> 1)]);
      buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, a + b);
    }
  }
}

// requires that cols + BLOCK_SIZE * 2 fits in smem (one row + max of one row + denom of one row)
// requires that you do NOT spawn a threadblock for a non-existent row
static inline void softmax(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SoftmaxArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;

  uint32_t rows_per_block = args->rows / MU_NUM_CLUSTERS;
  uint32_t row_elems = args->cols;
  uint32_t row_elems_fp32 = args->cols / 2;

  #pragma unroll
  for (uint32_t row = 0; row < rows_per_block; row++) {
    uint32_t block_elem_idx = (threadblock_id * rows_per_block + row) * row_elems_fp32;
    uint32_t chunks_per_block = (row_elems + DOUBLE_BLOCK_SIZE - 1) / DOUBLE_BLOCK_SIZE;

    __global uint32_t *x = args->x + block_elem_idx;
    __shared uint32_t *x_sdata = sdata;
    __shared uint32_t *buf_sdata = sdata + row_elems;

    // pass 1: find max
    _Float16 max_acc[SOFTMAX_ILP];
    uint32_t x_fp32[SOFTMAX_ILP];
    #pragma unroll SOFTMAX_ILP
    for (int i = 0; i < SOFTMAX_ILP; i++)
      max_acc[i] = as_bf16(NEG_INF_BF16_BITS);

    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += SOFTMAX_ILP) {
      #pragma unroll SOFTMAX_ILP
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x_fp32[i] = x[idx];
      }
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x_sdata[idx] = x_fp32[i];
      }
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        auto [x1, x0] = unpack_bf16x2(x_fp32[i]);
        max_acc[i] = fmaxf(fmaxf(x0, x1), max_acc[i]);
      }
    }

    _Float16 max = max_acc[0];
    #pragma unroll SOFTMAX_ILP
    for (int i = 1; i < SOFTMAX_ILP; i++)
      max = fmaxf(max, max_acc[i]);

    buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, max);
    mu_fence_smem();

    // warp reduce max
    reduce_max<MU_NUM_THREADS>(buf_sdata, tid, lane_id);
    if (lane_id == 0)
      buf_sdata[warp_id] = buf_sdata[tid];
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    // block reduce max
    if (warp_id == 0)
      reduce_max<MU_NUM_THREADS / THREAD_DIV>(buf_sdata, tid, lane_id);
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    _Float16 m = as_bf16((uint16_t)buf_sdata[0]);

    // pass 2: compute denom with known max
    _Float16 denom_acc[SOFTMAX_ILP];
    #pragma unroll SOFTMAX_ILP
    for (int i = 0; i < SOFTMAX_ILP; i++)
      denom_acc[i] = 0;

    uint32_t xss[SOFTMAX_ILP];
    _Float16 x0s[SOFTMAX_ILP], x1s[SOFTMAX_ILP];
    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += SOFTMAX_ILP) {
      #pragma unroll SOFTMAX_ILP
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        xss[i] = x_sdata[idx];
      }
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        auto [x1, x0] = unpack_bf16x2(xss[i]);
        x0s[i] = x0; x1s[i] = x1;
      }
      for (int i = 0; i < SOFTMAX_ILP; i++) {
        denom_acc[i] += mu_fexp(x0s[i] - m) + mu_fexp(x1s[i] - m);
      }
    }

    _Float16 denom = denom_acc[0];
    #pragma unroll SOFTMAX_ILP
    for (int i = 1; i < SOFTMAX_ILP; i++)
      denom += denom_acc[i];

    buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, denom);
    mu_fence_smem();

    // warp reduce denom
    reduce_sum<MU_NUM_THREADS>(buf_sdata, tid, lane_id);
    if (lane_id == 0)
      buf_sdata[warp_id] = buf_sdata[tid];
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    // block reduce denom
    if (warp_id == 0) {
      reduce_sum<MU_NUM_THREADS / THREAD_DIV>(buf_sdata, tid, lane_id);
      if (lane_id == 0) {
        _Float16 denom_final = as_bf16((uint16_t)buf_sdata[0]);
        _Float16 inv_denom = as_bf16(ONE_BF16_BITS) / denom_final;
        buf_sdata[0] = (uint32_t)__builtin_bit_cast(uint16_t, inv_denom);
      }
    }
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    _Float16 inv_d = as_bf16((uint16_t)buf_sdata[0]);

    // pass 3: compute softmax output
    uint32_t sh_ld[SOFTMAX_ILP], exps[SOFTMAX_ILP];
    _Float16 lows[SOFTMAX_ILP], his[SOFTMAX_ILP];
    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += SOFTMAX_ILP) {
      #pragma unroll SOFTMAX_ILP
      for (uint32_t i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        sh_ld[i] = x_sdata[idx];
      }
      #pragma unroll SOFTMAX_ILP
      for (uint32_t i = 0; i < SOFTMAX_ILP; i++) {
        auto [lo, hi] = unpack_bf16x2(sh_ld[i]);
        lows[i] = lo; his[i] = hi;
      }
      #pragma unroll SOFTMAX_ILP
      for (uint32_t i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        exps[i] = pack_bf16x2(mu_fexp(lows[i] - m) * inv_d, mu_fexp(his[i] - m) * inv_d);
      }
      #pragma unroll SOFTMAX_ILP
      for (uint32_t i = 0; i < SOFTMAX_ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x[idx] = exps[i];
      }
    }

    mu_barrier(0, BLOCK_NUM_WARPS);
  }
}
```

## example_vecadd_ilp.hpp

SUMMARY: Demonstrates a vectorized float32 addition kernel for the Muon GPU using ILP (instruction-level parallelism) and outer loop unrolling, showing warp-based indexing, chunked memory access patterns, and the mu_schedule API.

```cpp
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef VECADD_ILP
#define VECADD_ILP 1
#endif

#ifndef VECADD_NUM_WARPS
#define VECADD_NUM_WARPS 4
#endif

#ifndef VECADD_OUTER_UNROLL
#define VECADD_OUTER_UNROLL 1
#endif

extern "C" uint32_t __mu_num_warps = VECADD_NUM_WARPS;

struct VecAddArgs {
  __global float* A;
  __global float* B;
  __global float* C;
  uint32_t n;
};

template <uint32_t ILP>
static inline void vecadd_impl(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<VecAddArgs*>(raw_arg);

  constexpr uint32_t kIlp = ILP;
  constexpr uint32_t kWarpWidth = MU_NUM_THREADS;
  constexpr uint32_t kOuter = VECADD_OUTER_UNROLL;
  const uint32_t tid_in_warp = tid_in_threadblock % kWarpWidth;
  const uint32_t warp_id = tid_in_threadblock / kWarpWidth;
  const uint32_t warps_per_threadblock = threads_per_threadblock / kWarpWidth;
  const uint32_t elems_per_chunk = kWarpWidth * kIlp;
  uint32_t chunk;
  uint32_t chunk_stride;

#if MU_NUM_CLUSTERS == 1
  (void)threadblock_id;
  chunk = warp_id * elems_per_chunk;
  chunk_stride = warps_per_threadblock * elems_per_chunk;
#else
  const uint32_t global_warp_id = threadblock_id * warps_per_threadblock + warp_id;
  const uint32_t global_warp_stride = MU_NUM_CLUSTERS * warps_per_threadblock;
  chunk = global_warp_id * elems_per_chunk;
  chunk_stride = global_warp_stride * elems_per_chunk;
#endif

  const uint32_t outer_stride = chunk_stride * kOuter;
  float a_vals[kIlp];
  float b_vals[kIlp];
  float c_vals[kIlp];

  #pragma unroll 1
  for (; chunk < args->n; chunk += outer_stride) {
    #pragma unroll VECADD_OUTER_UNROLL
    for (uint32_t outer = 0; outer < kOuter; ++outer) {
      const uint32_t lane_base = chunk + outer * chunk_stride + tid_in_warp;

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        a_vals[i] = args->A[idx];
        b_vals[i] = args->B[idx];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        c_vals[i] = a_vals[i] + b_vals[i];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        args->C[idx] = c_vals[i];
      }
    }
  }
}

static inline void vecadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  vecadd_impl<VECADD_ILP>(
    raw_arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}
```
## Verified: shared-memory tiled matmul, +28% over global-memory baseline
Stages A and bank-padded B into SMEM, one barrier, then accumulates from SMEM.
The +16-float column pad avoids 16-way bank conflicts (lane stride 4 lines, 4 banks).
Measured: 63,401 cycles vs 88,560 baseline (64x64x64 fp32).

```cpp
static inline float bits_to_float(uint32_t b) { union { uint32_t u; float f; } c; c.u = b; return c.f; }
static inline uint32_t float_to_bits(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }

static inline void kernel_body(void* raw_arg, uint32_t tid, uint32_t nthreads, uint32_t) {
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t M = args->M, N = args->N, K = args->K;
  const uint32_t num_warps = nthreads / MU_NUM_THREADS;
  const uint32_t smem_a = 0, smem_b = M * K * 4, PAD = 16;
  for (uint32_t i = tid; i < M * K; i += nthreads)
    store_shared(smem_a + i * 4, 0, float_to_bits(args->A[i]));
  for (uint32_t i = tid; i < K * N; i += nthreads) {
    const uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * (K + PAD) + k) * 4, 0, float_to_bits(args->B[i]));
  }
  mu_barrier(1, num_warps);
  for (uint32_t idx = tid; idx < M * N; idx += nthreads) {
    const uint32_t row = idx / N, col = idx % N;
    float acc = 0.0f;
    const uint32_t a_base = smem_a + row * K * 4;
    const uint32_t b_base = smem_b + col * (K + PAD) * 4;
    #pragma unroll 8
    for (uint32_t k = 0; k < K; k++)
      acc += bits_to_float(load32_shared(a_base + k * 4)) * bits_to_float(load32_shared(b_base + k * 4));
    args->C[idx] = acc;
  }
}
```

Next: add per-thread register tiling (compute 4 outputs per thread) to reuse smem reads.

## Verified: SMEM matmul + register tiling (measured on 64x64x64 matmul)

Measured kernel-only cycles:
- baseline (global memory): 88560 cycles
- SMEM staged, padded B layout: 63401 cycles (1.40x)
- SMEM + 4 outputs/thread register tile: 46080 cycles (1.92x)
- SMEM + 8 outputs/thread register tile: 42983 cycles (2.06x)
- SMEM + 2x8 (2 rows x 8 cols) thread tile: 40207 cycles (2.20x)

Register tiling: each thread computes several adjacent outputs in one row.
The A operand is loaded once per k and reused across all columns, cutting
SMEM loads almost in half. Keep accumulators in locals; use one base address
per column (stride = (K + PAD) * 4 bytes). Past 8 outputs/thread, the inner
loop becomes address-arithmetic/issue-bound — use 2D tiles (2 rows x 8 cols)
and incremented pointers instead of per-iteration offset math.

```cpp
const uint32_t quarter = (M * N) / 4;
for (uint32_t q = tid; q < quarter; q += nthreads) {
  const uint32_t row = q / (N / 4);
  const uint32_t col0 = (q % (N / 4)) * 4;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  const uint32_t a_base = smem_a + row * K * 4;
  const uint32_t stride = (K + PAD) * 4;
  const uint32_t b0 = smem_b + col0 * stride;
  #pragma unroll 4
  for (uint32_t k = 0; k < K; k++) {
    const uint32_t off = k * 4;
    const float a = bits_to_float(load32_shared(a_base + off));
    acc0 += a * bits_to_float(load32_shared(b0 + off));
    acc1 += a * bits_to_float(load32_shared(b0 + stride + off));
    acc2 += a * bits_to_float(load32_shared(b0 + 2 * stride + off));
    acc3 += a * bits_to_float(load32_shared(b0 + 3 * stride + off));
  }
  const uint32_t c = row * N + col0;
  args->C[c + 0] = acc0; args->C[c + 1] = acc1;
  args->C[c + 2] = acc2; args->C[c + 3] = acc3;
}
```

## Verified: SMEM attention (single head, fp32)

Pattern: stage K row-major (padded), V transposed (column-major, padded),
softmax in SMEM scratch, register-tile both matmuls (one Q/P row x 4 K rows /
V columns each thread). For seq 64 also stage Q (~80 KiB total).
For seq 96 stage only K, V, S (~100 KiB); stream Q from global (each element
is read once).

Measured kernel-only cycles:
- attention 64x64x64: baseline 260599 -> 106900 (2.44x)
- attention 96x64: baseline 559764 -> 221318 (2.53x)

Counterexample (don't copy): a once-read kernel (conv patch-embed, stride 16)
does NOT benefit from im2col into SMEM — staging cost dominates and the result
ran 1.26x slower. Stage only data that is re-read several times.

## Conv patch-embed: measured negative results (16 OC x 16 patches, K = 768)

Every SMEM variant FAILED to beat the 102.5k-cycle baseline:
im2col 129k, weights-only 155k, plain copy 99k, +pitch pad 98k, split-K 105k.
This kernel reads each element a few times only — staging cost ~ saved cost.
Issue/latency-bound at 2 FMA/cycle. Don't spend candidates on SMEM staging here;
focus on instruction-count reduction in the inner loop and load coalescing.

## Pitfalls that fail correctness or stall timing (avoid in every kernel)

- Cross-core visibility: after mu_schedule, results aren't visible across cores
  until `mu_barrier(0, MU_NUM_CORES)`. Phase barriers within a threadblock use
  `mu_barrier(1, num_warps)`. A missing barrier => a fixed count of phantom errors.
- Configs run 8 warps/core. Derive the thread->output mapping from
  `threads_per_threadblock`, never a literal warp/thread count, or you write OOB.
- SMEM bank conflicts: a column stride that is a multiple of 16 floats serializes
  all 16 lanes onto one bank (sim hangs under timing). Always pad strides +16 floats
  or use an odd pitch.
- Only stage operands that are RE-READ many times (matmul/attention A,B,Q,K,V).
  Staging once-read data (e.g. conv inputs at stride 16) is pure overhead.
- Verdict/IO: never rely on device prints (volatile stores after big loops/strings
  get dropped by the backend). Compare against gold with rel+abs FP tolerance, not
  bitwise (RTL divider ~1 ULP; tiling reassociates sums).

## Activation kernels (swiglu, softmax) are near-optimal at baseline — measured

- SwiGLU 64x128 (elementwise): baseline strided loop is coalesced and exp-bound.
  4-wide ILP unrolling made it ~1.7x SLOWER (constant exp count + 8 live loads =
  register pressure). Do not unroll elementwise exp kernels.
- Softmax 64x67 rows: register-caching the row (float r[COLS]) SPILLS (67 floats >>
  32 GPRs) and was no faster than the 3-pass baseline. Don't cache rows wider than
  ~16 floats. A real win needs warp-cooperative reduction (split a row across lanes,
  shfl/SMEM reduce) so no thread holds the whole row.

## *** HARD CONSTRAINT: <= 31 distinct registers per kernel (RTL physical reg file) ***

The target core (RadianceSingleClusterConfig) has numPhysRegs=256 shared across
numWarps=8 ALWAYS-active warps. A kernel that uses N distinct architectural
registers needs 8*N physical registers; if 8*N > 256 (i.e. N > 32) the RTL Rename
stage asserts globalOverSubscription and the kernel is UNRUNNABLE on hardware.
Cyclotron does NOT model this, so a register-tiled kernel can look fast on cyclotron
yet be illegal. Budget: keep distinct registers in kernel_body <= 31.

What blows the budget (measured distinct-register counts, 64x64x64 matmul / attn):
- baseline 1-out/thread .............. 20 regs  LEGAL  (RTL pass)
- lean SMEM, 1-out, NO unroll ........ 22 regs  LEGAL
- SMEM + #pragma unroll 8 ............ 34 regs  ILLEGAL
- SMEM + 4-out register tile ......... ~38 regs ILLEGAL
- SMEM + 2x8 thread tile ............. 50 regs  ILLEGAL
- SMEM attention + 4-out tile ........ 54 regs  ILLEGAL

Rules of thumb to stay <= 31:
- Do NOT use deep #pragma unroll (each unrolled iteration keeps its loads live).
  unroll 1-2 max. Prefer incrementing one pointer over many base-address registers.
- Do NOT register-tile (multiple accumulators per thread). One accumulator/thread.
- Keep the number of simultaneously-live SMEM addresses small (recompute, don't hold
  8 base pointers).
- SMEM staging itself is fine and helps; the register blowup comes from unroll +
  tiling, not from using shared memory.

Realizable RTL speedups are therefore modest (e.g. matmul lean-SMEM ~1.35x), NOT the
2x+ that heavy tiling shows on cyclotron. Optimize WITHIN the 31-register budget.
Pre-check any candidate with scripts/muon/reg_check.sh <prob> <cand.cpp> before trusting
a cyclotron speedup.

## Launched suite — realistic results & lessons (RTL-anchored)

Best register-legal + correct kernel per target (cyclotron FP-exact correctness; calibrated cycles):
- matmul 64^3: lean SMEM staging = 2.52x on RTL (1.35x on cyclotron — cyclotron under-counts the
  global-memory penalty; the RTL number is the real one). SMEM staging (cut global traffic) is THE
  win. Register tiling (2x8, 50 regs) is FASTER on cyclotron but asserts globalOverSubscription on
  RTL — illegal.
- softmax 64x67: unroll8 (independent max/sum accumulators + branchless fmax) = 2.16x (calibrated;
  was a misleading 6.4x before memory calibration). Real, register-legal (ran on RTL).
- conv, swiglu, attention: BASELINE is best — no optimization beats it at these shapes. conv is
  compute-bound (768 MAC/output); swiglu is exp-bound elementwise; attention (seq64,d64) is
  global-scratch-access-bound with short dot-products, so SMEM staging (overhead > savings) AND
  ILP-unroll (0.81x) both make it SLOWER, and register-tiling is RTL-illegal.

LESSONS for generating kernels:
1. CUT GLOBAL TRAFFIC (SMEM staging of reused operands) is the highest-value lever on this HW —
   memory is ~30x more expensive than cyclotron's default model suggests.
2. COMPUTE-ONLY optims (unroll/ILP) only help when the kernel is genuinely compute-bound and the
   reduction chain is the bottleneck (softmax). They are MASKED on memory-bound kernels (the
   saving is constant but the memory-dominated total hides it) and can be net-SLOWER (attention).
3. Register tiling / deep unroll often wins on cyclotron but is RTL-ILLEGAL (globalOverSubscription).
   Keep peak simultaneously-live registers modest; SMEM staging at 1 output/thread is the safe win.
4. Don't add SMEM/ILP blindly — if the kernel is already compute-bound or has little reuse
   (conv, swiglu, small attention), the staging/ILP overhead makes it slower.
