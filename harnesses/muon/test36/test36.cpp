// Decode Q.K^T (GEMV): scores[s] = (K[s]·q)/sqrt(D). SIMT.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;
#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-4f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 1.0e-5f
#endif
#include "data"
struct KernelArgs { __global float *p, *V, *out; uint32_t S, D; };
// SUBSTITUTE HERE
// SUBSTITUTE END
static KernelArgs kernel_args;
static inline float fabsf_(float x){return x<0.f?-x:x;}
static inline bool close_enough(float c,float g){return fabsf_(c-g)<=TOLERANCE_REL*fabsf_(g)+TOLERANCE_ABS;}
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}
int main(){
  kernel_args={p_raw,v_raw,out_raw,SS,DD};
  mu_schedule(kernel_body,&kernel_args,NUM_WARPS);
  mu_barrier(0,MU_NUM_CORES);
  asm volatile("vx_tmc %0"::"r"(1):"memory");
  if(hart_id()!=0){for(;;){}}
  uint32_t e=0; for(uint32_t i=0;i<VERIFY_COUNT;i++) if(!close_enough(out_raw[i],gold_raw[i])) e++;
  uint32_t code=e?((e<<1)|1u):0u; asm volatile(".insn i 0x73, 0, x0, %0, 0"::"r"(code):"memory"); return 0;
}
