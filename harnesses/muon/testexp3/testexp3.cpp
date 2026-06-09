// Isolation harness: OUT[i] = mu_exp(IN[i]) elementwise (NO sum, NO division).
// If this passes RTL but softmax fails, the divergence is in the sum/reciprocal,
// not exp. If this fails RTL, exp/float-int conversion is the culprit.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 2.0e-4f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 1.0e-5f
#endif

#include "data"

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
  __global float *in, *out;
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
  kernel_args = {in_raw, out_raw, ROWS, COLS};
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) { for (;;) {} }
  // #25 write-drain fix: let the compute warps' global stores drain to memory before hart 0
  // reads the outputs. Without this, small/fast kernels race (verify reads stale data -> false
  // mismatches); slower kernels happen to drain in time. Validated on conv (RTL). Tunable.
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for (volatile uint32_t _d = 0; _d < DRAIN_ITERS; _d++) { asm volatile("" ::: "memory"); }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++) {
    if (!close_enough(out_raw[i], gold_raw[i])) errors++;
  }
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
