// DRAM-bandwidth probe: stream a large BSS region (uninitialized -> no source bloat) that
// exceeds L1+L2, so reads miss to DRAM. Measures cyclotron-vs-RTL(DRAMSim2) memory bandwidth
// to calibrate cyclotron's DRAM model (gmem.toml). Correctness is irrelevant (we measure
// cycles); the kernel reduces the streamed words so the loads aren't dead-code-eliminated.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

// N words. 1<<20 = 1,048,576 words = 4 MB >> L2 (512 KB) -> DRAM-bound streaming.
#ifndef BW_WORDS
#define BW_WORDS (1u << 18)
#endif
static __global volatile uint32_t big[BW_WORDS]; // BSS: reserved, not embedded in source

struct KernelArgs { uint32_t n; };
static KernelArgs kernel_args;

// SUBSTITUTE HERE
// SUBSTITUTE END

static inline uint32_t hart_id() {
  uint32_t id; asm volatile("csrr %0, mhartid" : "=r"(id)::"memory"); return id;
}

int main() {
  kernel_args = { BW_WORDS };
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) { for (;;) {} }
  // accumulate a sentinel from the streamed region (prevents DCE); always "pass" (code 0).
  uint32_t code = (big[0] == 0xffffffffu) ? 1u : 0u; // effectively always 0
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
