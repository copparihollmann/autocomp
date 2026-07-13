// Autocomp harness: MX-Gemmini FP8 matmul on the WHOLE Radiance config.
//
//   C[M,N] (bf16) = A[M,K] (fp8 e4m3) @ B[K,N] (fp8 e4m3), per-32-K-block e8m0 scales.
//
// Unlike the SIMT-only problems, the Muon warps here do not do the math: they orchestrate
// the MX matrix accelerator over its MMIO command block (rs1/rs2/inst at 0x84000), stage
// the e8m0 scale factors into SMEM, and move C out of SMEM. The accelerator's scratchpad
// IS cluster SMEM, so operands and results share the address space.
//
// The candidate kernel is substituted between the SUBSTITUTE markers and must define:
//   void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
//                    uint32_t threads_per_threadblock, uint32_t threadblock_id);
//
// VERDICT IS BIT-EXACT, not a tolerance. The golden encodes the hardware's exact
// accumulation semantics (16-deep accumulator, acc_e/acc_m precision schedule) and is
// verified bit-exact against real spike libgemmini. An FP tolerance would be wrong here:
// the idealized numpy/torch model disagrees with the hardware on ~90% of elements, so a
// tolerance loose enough to admit it would admit genuinely broken kernels too.

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

// NOTE: do NOT call this NUM_WARPS. VX_config.h (pulled in via mu_intrinsics.h) already
// defines NUM_WARPS as 8, so an `#ifndef NUM_WARPS` guard here silently inherits 8 and
// mu_schedule launches every warp slot. mxgemm_lib is heavily inlined, and 8 warps' worth
// of live registers blows the 256-entry physical register file (RTL Rename.scala asserts;
// cyclotron panics with globalOverSubscription). The mxgemm driver is warp-specialized
// and written for 2 warps, matching the standalone gemm_mxgemmini kernels.
#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;

#include "data"

// mxgemm_lib.hpp drives the accelerator; it references A_in/B_in/*_scales_*/*_lut from
// `data`, so it must be included after it.
#include "mxgemm_lib.hpp"

struct KernelArgs {
  uint8_t *C; // bf16 output, MATMUL_M * MATMUL_N * 2 bytes
  uint32_t M, N, K;
};

// SUBSTITUTE HERE
// SUBSTITUTE END

static KernelArgs kernel_args;

// Per-lane error counts; verification is spread across the threadblock's lanes so it does
// not dominate the cycle metric (a single-lane loop over M*N gmem loads costs ~20x the
// kernel). NOTE: a threadblock spans BOTH cores, so tid_in_threadblock runs to
// MU_NUM_THREADS * MU_NUM_CORES - 1, not MU_NUM_THREADS - 1. Sizing this to 16 silently
// dropped half the checks (and overflowed the array).
#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};

// Lane-parallel verification pass: lane t checks elements t, t+16, t+32, ...
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
  uint32_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id)::"memory");
  return id;
}

int main() {
  // mxgemm_lib takes an unqualified pointer; strip the __global address space the same
  // way the library does for its own operands (reinterpret via uint32_t).
  kernel_args = {reinterpret_cast<uint8_t *>(reinterpret_cast<uint32_t>(C_raw)),
                 MATMUL_M, MATMUL_N, MATMUL_K};

  mu_schedule(kernel_body, &kernel_args, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); // both cores fully done before verification

  // Let the kernel's C stores drain before anything reads them. mxgemm()'s SIMT move-out
  // ends with plain stores and no trailing fence; on RTL those are still in flight when
  // verification runs, so the check reads stale memory and reports spurious errors (this
  // is the known Muon "RTL fails but the kernel is correct" write-drain artifact).
  // Cyclotron retires stores in order, so DRAIN_ITERS stays 0 there and costs nothing;
  // vcs_gate_mx.sh compiles with -DDRAIN_ITERS=200000.
  mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for (volatile uint32_t d = 0; d < DRAIN_ITERS; d++) {
    asm volatile("" ::: "memory");
  }

  // Verify in its own scheduled pass. Doing it inline in main() would run on a single
  // lane (mu_schedule's manager path ends with vx_tmc(1)) -- a 1/16-coverage check that
  // still reports PASS. Re-enabling lanes by hand is worse: the dormant lanes resume with
  // stale registers and compare garbage addresses.
  mu_schedule(verify_body, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);

  if (hart_id() != 0) {
    for (;;) {}
  }

  mu_fence(); // lane_errors[] written by the verify pass must be visible here
  uint32_t total = 0;
  for (uint32_t t = 0; t < VERIFY_LANES; t++) total += lane_errors[t];

  // Verdict via tohost: the simulator's ECALL reports rs1, so encode ECALL with
  // rs1 = code. 0 = pass, else (errors<<1)|1 -> prints "case=<errors>".
  uint32_t code = total ? ((total << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
