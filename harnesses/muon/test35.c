// GEMV softmax: probs[s] = softmax(scores)[s] over S (one decode row). SIMT.
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
struct KernelArgs { __global float *in, *out; uint32_t S; };
// SUBSTITUTE HERE
// SUBSTITUTE END
static KernelArgs kernel_args;
static inline float fabsf_(float x){return x<0.f?-x:x;}
static inline bool close_enough(float c,float g){return fabsf_(c-g)<=TOLERANCE_REL*fabsf_(g)+TOLERANCE_ABS;}
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}
int main(){
  kernel_args={x_raw,out_raw,SS};
  mu_schedule(kernel_body,&kernel_args,NUM_WARPS);
  mu_barrier(0,MU_NUM_CORES);
  asm volatile("vx_tmc %0"::"r"(1):"memory");
  if(hart_id()!=0){for(;;){}}
  uint32_t e=0; for(uint32_t i=0;i<VERIFY_COUNT;i++) if(!close_enough(out_raw[i],gold_raw[i])) e++;
  uint32_t code=e?((e<<1)|1u):0u; asm volatile(".insn i 0x73, 0, x0, %0, 0"::"r"(code):"memory"); return 0;
}
