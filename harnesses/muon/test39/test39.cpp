// Verifying fp8 REQUANTIZER (QUANT_OUTPUT=true): kernel fp8 output vs requant golden C_out.
// Spec flags "data bytes correct, wrong order" -- a failure pattern here pinpoints it.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;
#include "data"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"
constexpr GemmConfig GEMM_CFG{ .TILE_M = MATMUL_M, .TILE_N = MATMUL_N, .TILE_K = MATMUL_K,
                               .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = true };
struct KernelArgs { uint8_t *C; uint32_t M, N, K; };
// SUBSTITUTE HERE
// SUBSTITUTE END
static KernelArgs kernel_args;
#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};
static void verify_body(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t e=0; for (uint32_t i=tid;i<VERIFY_COUNT;i+=tpb) if (C_raw[i]!=gold_raw[i]) e++;
  lane_errors[tid]=e; mu_fence();
}
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}
int main(){
  kernel_args={reinterpret_cast<uint8_t*>(reinterpret_cast<uint32_t>((uint8_t*)C_raw)),MATMUL_M,MATMUL_N,MATMUL_K};
  mu_schedule(kernel_body,&kernel_args,MX_NUM_WARPS);
  mu_barrier(0,MU_NUM_CORES); mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for(volatile uint32_t d=0;d<DRAIN_ITERS;d++) asm volatile("":::"memory");
  mu_schedule(verify_body,nullptr,1); mu_barrier(0,MU_NUM_CORES);
  if(hart_id()!=0){for(;;){}}
  mu_fence(); uint32_t t=0; for(uint32_t i=0;i<VERIFY_LANES;i++) t+=lane_errors[i];
  uint32_t code=t?((t<<1)|1u):0u; asm volatile(".insn i 0x73, 0, x0, %0, 0"::"r"(code):"memory"); return 0;
}
