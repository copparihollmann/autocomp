#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp8_64x1024x3072.h"

#define DIM 16
#define BLK 64
#define BLK_OUTC (BLK / 4)
#define BLOCKS_M (MATMUL_M / BLK)
#define BLOCKS_N (MATMUL_N / BLK)
#define BLOCKS_K (MATMUL_K / BLK)
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
// MATMUL_M/K/N are PADDED to BLK (M,N) / DIM (K); REAL_* are the true dims. The block loop
// covers padded blocks; the last M/N block is partial and uses loop_ws pad to skip the rest.
#define REAL_M 1
#define REAL_N 3072
typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif

// All DMA-visible buffers at FIXED DRAM addresses so the capture and the
// optimization harness use bit-identical addresses regardless of binary layout
// (the spike MX model's behavior depends on buffer addresses).
#define A_blk          ((elem_t (*)[MATMUL_K])      0xA0000000UL)
#define B_blk          ((elem_t (*)[MATMUL_K])      0xA0100000UL)
#define A_s            ((uint8_t (*)[BLK])          0xA0200000UL)
#define B_s            ((uint8_t (*)[BLK])          0xA0280000UL)
#define C_blk          ((out_t (*)[BLK_OUTC])       0xA0300000UL)
#define scale_factors  ((uint32_t *)                0xA0400000UL)
#define C_acc          ((float (*)[MATMUL_N])       0xA0800000UL)
#define C_hw           ((out_t (*)[OUT_COLS])       0xA1000000UL)  // bf16-packed row-major output
#define SCALE_FACTORS_BYTES (1 << 22)
#define A_S_BYTES   ((MATMUL_K / 32) * BLK)
#define B_S_BYTES   ((MATMUL_K / 32) * BLK)   // requantizer scale stream; advances per launch -- keep large + zeroed

static inline float bf16f(uint16_t b) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)b) << 16; return v.f;
}
static inline uint32_t fbits(float f) {
  union { uint32_t u; float f; } v; v.f = f; return v.u;
}

#define OUTPUT_MATRIX_NAME C_hw
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/real-gemv-3072/gold0.txt
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {
  (void)x; (void)y; return 1;  // correctness checked vs GOLD_FILE by the runner
}
static out_t gold[1][1];

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  // SUBSTITUTE HERE
  // SUBSTITUTE END
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < REAL_M; i++)                 // dump only the REAL sub-block: padding writes by a
    for (int j = 0; j < REAL_N / BF16_PER_WORD; j++)  // candidate can't cause a false mismatch
      printf("%08x\n%08x\n", (unsigned)(C_hw[i][j] >> 32), (unsigned)(C_hw[i][j] & 0xffffffffu));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
