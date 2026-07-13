// MX-Gemmini fp8 64x64 WS matmul harness for autocomp.
// Adapted from gemmini-rocc-tests bareMetalC/matmul_tiled_fp8_64x64.c.
// Golden inputs/outputs come from the MxGen-generated header (A_in, B_in,
// A_scales_row, B_scales_col, C_out_bf16). The optimizable kernel goes between
// the SUBSTITUTE markers; autocomp injects cycle counting + a correctness check
// full_is_equal(OUTPUT_MATRIX_NAME, gold) around it.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#ifndef SPIKE_SIM
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

#define OUTPUT_MATRIX_NAME C_hw

// autocomp's injected timing wrapper calls fence(); gemmini-mx-cleanup exposes
// gemmini_fence() (it must wait for the accelerator before reading cycles).
#ifndef fence
#define fence() gemmini_fence()
#endif

// ---- Scale factor loader (RTL MMIO path; spike uses gemmini_mx_load_scales) ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
    }
  }
}

// Equality over the packed bf16 output words.
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], out_t y[MATMUL_M][OUT_COLS]) {
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < OUT_COLS; j++)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static out_t C_hw[MATMUL_M][OUT_COLS];
  static out_t gold[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};

  // Build the packed golden output from the MxGen reference (4 bf16 per word).
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      gold[i][j] = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                   ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                   ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                   ((uint64_t)C_out_bf16[i][j*4 + 0]);
    }
  }

  // ---- Tile dimensions / scratchpad layout (in scope for the kernel) ----
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;
  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  uint32_t acc_addr = (1u << (ADDR_LEN - 1));
  int SPAD_DEST = 128;

  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    memset(C_hw, 0, sizeof(C_hw));

    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }

  printf("Correct result\n");

#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
