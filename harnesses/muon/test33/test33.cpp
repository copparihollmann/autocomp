// Autocomp harness: FUSED QKV projection (MX-Gemmini) + RoPE (Muon SIMT), sharing SMEM.
//   out[M,N] (bf16) = RoPE( bf16( X[M,K] @ W[K,N] ), cos, sin )
// The MX accelerator does the fp8 matmul into SMEM; the SIMT lanes apply RoPE to the
// SMEM-resident result and write out -- the Q/K projection never round-trips through DRAM.
// Golden is bit-exact (hardware matmul semantics; bf16 truncation matches the kernel).
//
// The candidate kernel is substituted between the markers and must define kernel_body().

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2   // mxgemm_lib is warp-specialized for 2 warps (see test20 note)
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;

#include "data"

// fp8 path: mxgemm_lib references these LUTs; unused for fp8, provide zeros (as the
// standalone gemm_mxgemmini fp8 kernels do).
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_lib.hpp"

// The fused GEMM config: single output tile, full bf16 output (no requant). RoPE needs the
// real float values, so QUANT_OUTPUT stays false.
constexpr GemmConfig GEMM_CFG{
    .TILE_M = MATMUL_M,
    .TILE_N = MATMUL_N,
    .TILE_K = MATMUL_K,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

struct KernelArgs {
  uint8_t *C;                 // bf16 output [M][N]
  const float *cosc, *sinc;   // RoPE caches [M][N]
  uint32_t M, N, K;
};

// SUBSTITUTE HERE
// SUBSTITUTE END

static KernelArgs kernel_args;

#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};

static void verify_body(void *, uint32_t tid_in_threadblock,
                        uint32_t threads_per_threadblock, uint32_t) {
  uint32_t errors = 0;
  for (uint32_t i = tid_in_threadblock; i < VERIFY_COUNT; i += threads_per_threadblock) {
    if (C_raw[i] != gold_raw[i]) errors++;
  }
  lane_errors[tid_in_threadblock] = errors;
  mu_fence();
}

static inline uint32_t hart_id() {
  uint32_t id; asm volatile("csrr %0, mhartid" : "=r"(id)::"memory"); return id;
}

int main() {
  kernel_args = {reinterpret_cast<uint8_t *>(reinterpret_cast<uint32_t>((uint16_t*)C_raw)),
                 cos_raw, sin_raw, MATMUL_M, MATMUL_N, MATMUL_K};
  mu_schedule(kernel_body, &kernel_args, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for (volatile uint32_t d = 0; d < DRAIN_ITERS; d++) { asm volatile("" ::: "memory"); }

  mu_schedule(verify_body, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  if (hart_id() != 0) { for (;;) {} }
  mu_fence();
  uint32_t total = 0;
  for (uint32_t t = 0; t < VERIFY_LANES; t++) total += lane_errors[t];
  uint32_t code = total ? ((total << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
