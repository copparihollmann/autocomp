// Compute-throughput probes to calibrate cyclotron's compute model vs RTL (no memory traffic, so
// they isolate the compute pipeline and RTL-sim fast). Two modes selected at compile time:
//   CHAIN=1 : serial dependency chain (acc = acc*c + acc) -> isolates FP LATENCY (back-to-back
//             dependent FMAs; throughput/ILP can't hide it).
//   CHAIN=0 : ILP mode (N independent accumulators) -> isolates ISSUE/THROUGHPUT (latency hidden).
// ITERS sets the work. No global/SMEM access (operands live in registers), so cyclotron's gmem/smem
// models are out of the picture -> the cyc-vs-RTL gap is purely the compute pipeline (latency vs issue).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef ITERS
#define ITERS 200000u
#endif
#ifndef CHAIN
#define CHAIN 1
#endif

struct KernelArgs { __global float* out; };
static KernelArgs kernel_args;
static __global volatile float sink[256];

// SUBSTITUTE HERE
// SUBSTITUTE END

static inline uint32_t hart_id() {
  uint32_t id; asm volatile("csrr %0, mhartid" : "=r"(id)::"memory"); return id;
}
int main() {
  kernel_args = { (__global float*)sink };
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) { for (;;) {} }
  uint32_t code = (sink[0] == 1234.5f) ? 1u : 0u; // ~always 0
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
