// Value probe: compute mu_exp(PROBE_X) on the GPU and ecall the raw float BITS as the
// verdict code, so the sim prints "tohost=<bits>". Run on cyclotron vs RTL for the same
// PROBE_X and decode the bits to see exactly how/where RTL's FP diverges. Also probes the
// intermediate stages (polynomial p, 2^k) selectable via PROBE_STAGE.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef PROBE_X
#define PROBE_X -0.437583327293396f
#endif
#ifndef PROBE_STAGE
#define PROBE_STAGE 1
#endif

static inline uint32_t f2b(float f){ union{uint32_t i;float f;}c; c.f=f; return c.i; }

static inline uint32_t hart_id(){ uint32_t id; asm volatile("csrr %0, mhartid":"=r"(id)::"memory"); return id; }

static void kbody(void*, uint32_t, uint32_t, uint32_t) {}

int main() {
  mu_schedule(kbody, 0, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) { for(;;){} }

  const float x = PROBE_X;
  const float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
  float t = x * LOG2E;
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * LN2;
  float p = 1.0f + r*(1.0f + r*(0.5f + r*(0.16666667f + r*(0.041666668f + r*0.008333334f))));
  union { uint32_t i; float f; } s; s.i = (uint32_t)((k + 127) << 23);
  float full = p * s.f;

  uint32_t code;
#if PROBE_STAGE == 1
  code = f2b(p);
#elif PROBE_STAGE == 2
  code = f2b(s.f);
#elif PROBE_STAGE == 3
  code = f2b((float)k);
#else
  code = f2b(full);
#endif
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
