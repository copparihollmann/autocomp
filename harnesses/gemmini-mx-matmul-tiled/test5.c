#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp4_128x128x128.h"
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
#define OUT_COLS (MATMUL_N/BF16_PER_WORD)
static out_t C_hw[MATMUL_M][OUT_COLS];
#define OUTPUT_MATRIX_NAME C_hw
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/gemmini-mx-matmul-tiled/gold5.txt
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]){(void)x;(void)y;return 1;}
static out_t gold[1][1];

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1
int main(){
#ifndef BAREMETAL
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror("mlockall");return 1;}
#endif
  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};
  memset(C_hw,0,sizeof(C_hw));
  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("GOLD_BEGIN\n");
  for(int i=0;i<MATMUL_M;i++)for(int j=0;j<MATMUL_N/BF16_PER_WORD;j++)
    printf("%08x\n%08x\n",(unsigned)(C_hw[i][j]>>32),(unsigned)(C_hw[i][j]&0xffffffffu));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
