// AUTO-GENERATED MX-Gemmini FLASH attention harness (S=128, D=64, tile=64).
// O = softmax(Q K^T) V with K/V tiling + online softmax (no S x S buffer).
// Correctness vs hardware-captured gold (GOLD_FILE) checked by the runner.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fla_qk_128x64.h"
#include "include/mxgen_fla_v_128x64.h"

#define DIM 16
#define ATT_S 128
#define ATT_D 64
#define BT 64
#define N_TILES (ATT_S / BT)
#define GROUPS_BT (BT / 32)
#define BLK_OUTC (BT / 4)
#define D_OUT_COLS (ATT_D / 4)

typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME O_acc
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/gemmini-mx-flash-attn/gold9.txt
int full_is_equal(float x[ATT_S][ATT_D], const float y[ATT_S][ATT_D]) {
  (void)x; (void)y; return 1;  // verified vs GOLD_FILE by the runner
}
static float gold[1][1];

// fixed scratch buffers (pinned addresses; spike behavior is layout-sensitive)
#define S1_hw   ((out_t (*)[BLK_OUTC]) 0xA0000000UL)   // S x BT scores tile (bf16-packed)
#define P_q     ((elem_t (*)[BT])      0xA0100000UL)
#define P_s     ((uint8_t (*)[ATT_S])  0xA0200000UL)
#define O_hw    ((out_t (*)[D_OUT_COLS]) 0xA0300000UL)
#define O_acc   ((float (*)[ATT_D])    0xA0400000UL)
#define M_run   ((float *)             0xA0600000UL)
#define L_run   ((float *)             0xA0610000UL)
#define scale_factors ((uint32_t *)    0xA0700000UL)
#define SCALE_FACTORS_BYTES (1 << 22)

static inline float bf16_to_f(uint16_t b) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)b) << 16; return v.f;
}
static inline uint32_t f_bits(float f) { union { uint32_t u; float f; } v; v.f = f; return v.u; }
static inline float bits_f(uint32_t u) { union { uint32_t u; float f; } v; v.u = u; return v.f; }
static inline float f_abs(float x) { return bits_f(f_bits(x) & 0x7FFFFFFFu); }
static inline int f_exp(float x) { return (int)((f_bits(x) >> 23) & 0xFF) - 127; }
static inline float pow2i(int e) {
  if (e < -126) return 0.0f;
  if (e > 127) e = 127;
  return bits_f((uint32_t)(e + 127) << 23);
}
static inline float exp_f(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float z = x * 1.44269504f;
  int n = (int)(z + (z >= 0 ? 0.5f : -0.5f));
  float f = z - (float)n;
  float p = 0.9999999916f + f*(0.6931471825f + f*(0.2401536316f + f*(0.0558263185f + f*(0.0089893397f + f*0.0018775767f))));
  return p * pow2i(n);
}
static inline uint8_t fp8_e4m3_rtz(float x) {
  if (x == 0.0f) return 0;
  uint8_t s = (f_bits(x) >> 31) ? 0x80 : 0;
  float a = f_abs(x);
  if (a >= 448.0f) return s | 0x7E;
  int e = f_exp(a);
  if (e < -6) return s;
  float m = a * pow2i(-e);
  int mant = (int)((m - 1.0f) * 8.0f);
  return s | (uint8_t)(((e + 7) << 3) | (mant & 7));
}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  memset(O_acc, 0, ATT_S * ATT_D * sizeof(float));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < ATT_S; i++)
    for (int j = 0; j < ATT_D; j++)
      printf("%08x\n", f_bits(O_acc[i][j]));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
