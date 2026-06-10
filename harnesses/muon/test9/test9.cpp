// Autocomp harness: GELU C[m,n] = x * sigmoid(1.702*x) (fp32 sigmoid-approx, 64x512)
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

// fp32 exp shared by all candidates; the golden output is computed with this same
// polynomial (see gen_data.py), so kernels must use mu_exp().
static inline float mu_exp(float x) {
  const float LOG2E = 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;
  float t = x * LOG2E;
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * LN2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
  union { uint32_t i; float f; } s;
  s.i = (uint32_t)((k + 127) << 23);
  return p * s.f;
}

struct KernelArgs {
  __global float *A, *C;
  uint32_t M, N;
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
  kernel_args = {A_raw, C_raw, M, N};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); // both cores fully done before verification

  // Verify + report once: single lane; non-zero harts spin (sim ends when hart 0 exits).
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) {
    for (;;) {}
  }

  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++) {
    if (!close_enough(C_raw[i], gold_raw[i])) errors++;
  }
  // Verdict via tohost: the simulator's ECALL reports rs1, so encode ECALL with
  // rs1 = code. 0 = pass, else (errors<<1)|1 -> prints "case=<errors>".
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
