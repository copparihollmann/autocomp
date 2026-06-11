// Autocomp harness: layer_norm OUT[r,:] = (IN[r,:]-mean)/sqrt(var+eps)*gamma+beta (fp32, 64x768)
//
// The candidate kernel is substituted between the SUBSTITUTE markers and must define:
//   void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
//                    uint32_t threads_per_threadblock, uint32_t threadblock_id);

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

// Single-byte console writes; the simulator echoes every store. Keep verdict to one
// conditional store: longer per-char sequences can be dropped by the Muon compiler.
// Verdict is returned as main()'s exit code; the simulator prints "Error: <code>".
// (Device console printing is unreliable with this toolchain - do not print.)

#ifndef NUM_WARPS
#define NUM_WARPS 4 // occupancy per core; threadblock spans MU_NUM_CORES cores
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

// Relative FP tolerance: accommodates reassociated accumulation (tiling/split-K)
// and the RTL's ~1-ULP-approximate divider.
#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

// fp32 reciprocal-sqrt (no libm/sqrtf libcall on this toolchain): fast-inverse-sqrt seed
// + 3 Newton iters (~1e-6 rel). layer_norm golden uses exact 1/sqrt; tolerance covers the residual.
static inline float mu_rsqrt(float x) {
  union { float f; uint32_t i; } u; u.f = x;
  u.i = 0x5f3759dfu - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

struct KernelArgs {
  __global float *in, *gamma, *beta, *out;
  uint32_t rows, cols;
};

// SUBSTITUTE HERE
// SUBSTITUTE END

static KernelArgs kernel_args;

static inline float fabsf_(float x) { return x < 0.0f ? -x : x; }
static inline bool close_enough(float c, float g) {
  return fabsf_(c - g) <= TOLERANCE_REL * fabsf_(g) + TOLERANCE_ABS;
}
static inline uint32_t hart_id() {
  uint32_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id)::"memory");
  return id;
}

int main() {
  kernel_args = {x_raw, gamma_raw, beta_raw, out_raw, ROWS, COLS};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); // both cores fully done before verification

  // Verify + report once: single lane; non-zero harts spin (sim ends when hart 0 exits).
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) {
    for (;;) {}
  }

  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++) {
    if (!close_enough(out_raw[i], gold_raw[i])) errors++;
  }
  // Verdict via tohost: the simulator's ECALL reports rs1, so encode ECALL with
  // rs1 = code. 0 = pass, else (errors<<1)|1 -> prints "case=<errors>".
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
