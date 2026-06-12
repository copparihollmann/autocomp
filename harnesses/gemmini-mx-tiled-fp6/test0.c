#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp6_128x128x128.h"
#define DIM 16
#define BLK 64
#define BLK_OUTC (BLK/4)
#define BLOCKS_M (MATMUL_M/BLK)
#define BLOCKS_N (MATMUL_N/BLK)
#define BF16_PER_WORD 4
typedef uint8_t elem_t; typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
static elem_t A_hw[BLK/2][MATMUL_K];
static uint8_t A_s[MATMUL_K/32][BLK], B_s[MATMUL_K/32][BLK];
static out_t C_blk[BLK][BLK_OUTC];
static float C_acc[MATMUL_M][MATMUL_N];
static uint8_t A_lp[12], B_lp[12];
static void mx_pack_lut(const uint8_t *cc, uint8_t *o){uint64_t lo=0;uint32_t hi=0;for(int i=0;i<16;i++){int bit=i*6;uint64_t v=cc[i]&0x3F;if(bit+6<=64)lo|=v<<bit;else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}
static inline uint32_t fbits(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}
static inline float bf16f(uint16_t b){union{uint32_t u;float f;}v;v.u=((uint32_t)b)<<16;return v.f;}
#define OUTPUT_MATRIX_NAME C_acc
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/gemmini-mx-tiled-fp6/gold0.txt
int full_is_equal(float x[MATMUL_M][MATMUL_N], const float y[MATMUL_M][MATMUL_N]){(void)x;(void)y;return 1;}
static float gold[1][1];

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1
int main(){
#ifndef BAREMETAL
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror("mlockall");return 1;}
#endif
  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};
  memset(C_acc,0,sizeof(C_acc));
  mx_pack_lut((const uint8_t*)A_lut,A_lp); mx_pack_lut((const uint8_t*)B_lut,B_lp);
  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("GOLD_BEGIN\n");
  for(int i=0;i<MATMUL_M;i++)for(int j=0;j<MATMUL_N;j++)
    printf("%08x\n",fbits(C_acc[i][j]));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
