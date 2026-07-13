## matmul_tiled_fp6_128x128.c

SUMMARY: Demonstrates MX-format (fp6/fp8/fp4) matrix multiplication using Gemmini RoCC intrinsics, covering LUT/scale-factor loading, scratchpad tiling for A and B matrices, fused weight-stationary loop execution, and BF16-packed result readback from shared memory.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"

#define TILE 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4

// Memory-mapped control registers
#define GEMMINI_CTRL     0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

// LUT regions: weight (64 rows x 96b), activation, output
#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x380)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x680)

// Scale-factor SRAM regions
#define GEMMINI_SF_MEM   0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B  GEMMINI_SF_MEM

// Shared memory base for reading BF16 results
#define SMEM 0x40000000

// FP6 format selector (0=fp8, 1=fp6, 2=fp4)
#define GEMMINI_FORMAT 1

// Override fence to poll hardware busy register
#undef gemmini_fence
#define gemmini_fence() \
  { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

// Override RoCC instruction to use MMIO registers
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
  *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
  *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
  *((volatile uint32_t*) GEMMINI_INST_ADDR) =                                 \
      (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

typedef uint8_t out_t;

// Write byte-array scale factors into 64-bit-wide SRAM
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}
```

```c
// --- Configure Gemmini for MX weight-stationary mode ---

gemmini_flush(0);

// gemmini_config_ex equivalent: set format, strides, LUT enable, WS mode
ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
  ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
  | ((uint64_t)(1) << 16)       // A stride
  | (GEMMINI_FORMAT << 14)      // C format (fp6=1)
  | (GEMMINI_FORMAT << 12)      // B format
  | (GEMMINI_FORMAT << 10)      // A format
  | (0 << 9)                    // B transpose = false
  | (0 << 8)                    // A transpose = false
  | ((false) << 7)              // set_only_strides = false
  | ((USE_LUT) << 4)            // enable LUT-based dequant
  | ((0) << 3)                  // activation = none
  | ((WEIGHT_STATIONARY) << 2)
  | CONFIG_EX,
  ((uint64_t)(1) << 48)         // C stride
  | (0),
  k_CONFIG);

int tiles_I = MATMUL_M / 32;   // m-dimension tiles
int tiles_K = MATMUL_K / 16;   // k-dimension tiles
int tiles_J = MATMUL_N / 32;   // n-dimension tiles
int OUT_COLS = MATMUL_M / BF16_PER_WORD;

gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

// Configure requantizer/mvout: pass scale-factor buffer and tile counts
gemmini_mxquant_config_mvout(
  (uint64_t)scale_factors,      // output scale-factor destination
  tiles_I, tiles_J, tiles_K,
  0, 0,
  QUANT_LUT_UPDATE_GRANULARITY);

// Extended config: WS, no activation, identity scale, tile dims, no transpose
gemmini_extended3_config_ex(
  WEIGHT_STATIONARY, 0, 0,
  ACC_SCALE_IDENTITY,
  1, 1, 0, 0, false,
  1, 1, 3, 1);
```

```c
// --- Load LUTs (hardware path, pre-packed by lut_mapping_demo.py) ---
// Each LUT entry is 3 x uint32_t (96 bits) packed per quantization group.

// LUT0: weight (B) dequantization table
for (size_t i = 0; i < (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
  volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i;
  dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
}

// LUT1: activation (A) dequantization table
for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
  volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i;
  dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
}

// LUT2: output requantization table
for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
  volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i;
  dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
}

// Load per-block scale factors into dedicated SRAM banks
// sel=0 → A scales (row-wise), sel=1 → B scales (column-wise)
load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                   (uint8_t *) &A_scales_row, MATMUL_M * MATMUL_GK);
load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                   (uint8_t *) &B_scales_col, MATMUL_N * MATMUL_GK);
gemmini_fence();
```

```c
// --- Scratchpad layout and MVIN ---
// A_in_hw[MATMUL_M/2][MATMUL_K]: tiles_I m-tiles x tiles_K k-tiles,
//   each K_TILE hw-rows x DIM bytes
// B_in[MATMUL_K][MATMUL_N/2]:    tiles_K k-tiles x tiles_J n-tiles,
//   each K_TILE rows x DIM bytes

uint32_t a_base = 0;
uint32_t b_base = 8192 - tiles_K * tiles_J * K_TILE;

// MVIN A: row stride = full packed row width (MATMUL_K bytes)
gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
for (int i = 0; i < tiles_I; i++) {
  for (int k = 0; k < tiles_K; k++) {
    uint8_t *dram_ptr = (uint8_t *)A_in_hw
                        + i * DIM * MATMUL_K   // row-tile offset
                        + k * DIM;             // k-tile column offset
    uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    gemmini_fence();
  }
}

// MVIN B: row stride = packed column width (MATMUL_N/2 bytes, fp6 packs 2/byte)
gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
for (int k = 0; k < tiles_K; k++) {
  for (int j = 0; j < tiles_J; j++) {
    uint8_t *dram_ptr = (uint8_t *)B_in
                        + k * K_TILE * (MATMUL_N / VALUES_PER_BYTE)
                        + j * DIM;
    uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    gemmini_fence();
  }
}
```

```c
// --- Fused weight-stationary tiled matmul loop ---
// skip_flags = 0x38: skip ldA (0x08) | ldB (0x10) | ldD (0x20)
// Data already in scratchpad; only compute + store to accumulator.
//
// Bit map for skip flags:
//   bit 3 (0x08): ldA  – skip loading A from DRAM
//   bit 4 (0x10): ldB  – skip loading B from DRAM
//   bit 5 (0x20): ldD  – skip loading D/bias
//   bit 6 (0x40): ex   – skip compute
//   bit 7 (0x80): st   – skip acc→spad store

int SPAD_DEST = 128;
uint32_t acc_addr = (1u << (ADDR_LEN - 1));

gemmini_loop_ws_spad(
  tiles_I, tiles_J, tiles_K,
  0, 0, 0,          // pad_I, pad_J, pad_K
  a_base,           // A scratchpad base address
  8192,             // B scratchpad end address (b_base sentinel)
  0,                // D (bias) – none
  SPAD_DEST,        // C accumulator destination address
  false, false,     // A_transpose, B_transpose
  false, false, false, // full_C, low_D, ex_accumulate
  NO_ACTIVATION,
  0, 0,             // a_spad_id, b_spad_id (double-buffer IDs)
  false,            // is_resadd
  0x38);            // skip ldA | ldB | ldD; perform compute and store
```

```c
// --- Read BF16-packed results from shared memory ---
// Results are stored as packed BF16 words (4 bf16 per uint64_t).
// smem layout: base + SPAD_DEST*2 qwords, row-major, OUT_COLS qwords/row.

uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;

for (int i = 0; i < MATMUL_M; i++) {
  for (int j = 0; j < OUT_COLS; j++) {
    C_hw[i][j] = *(smem_start_addr + (i * OUT_COLS + j));
  }
}
gemmini_fence();
```

```c
// --- Verify BF16 output against golden reference ---
// C_out_bf16[MATMUL_M][MATMUL_N]: golden bf16 values (uint16_t each)
// C_hw[MATMUL_M][OUT_COLS]:       hardware output (4 bf16 packed per uint64_t)

int errors = 0;
for (int i = 0; i < MATMUL_M; i++) {
  for (int j = 0; j < OUT_COLS; j++) {
    uint64_t got = C_hw[i][j];

    // Pack 4 consecutive bf16 golden values into expected uint64_t
    uint64_t exp =
      ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
      ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
      ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
      ((uint64_t)C_out_bf16[i][j*4 + 0]);

    if (got != exp) {
      for (int lane = 0; lane < BF16_PER_WORD; lane++) {
        uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
        uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
        if (got_bf16 != exp_bf16) {
          printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                 i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
          errors++;
        }
      }
    }
  }
}
```

## matmul_ws_mx_generic.c

SUMMARY: Demonstrates MX-format (fp6/fp8/fp4) weight-stationary matmul using Gemmini RoCC intrinsics, covering LUT loading, scale factor loading, scratchpad tiling for A and B matrices, and the fused gemmini_loop_ws_spad execution with skip-mask control.

```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"

#define TILE 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4

#define GEMMINI_CTRL      0x40084000
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)   // weight LUT  (64 rows x 96b = 0x300 bytes)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x380)  // activation LUT
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x680)  // output LUT
#define GEMMINI_SF_MEM    0x40088000
#define GEMMINI_SF_MEM_A  (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B  GEMMINI_SF_MEM
#define SMEM              0x40000000

#undef GEMMINI_BUSY_ADDR
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

// FP6 format selector (0=fp8, 1=fp6, 2=fp4)
#define GEMMINI_FORMAT 1

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
  *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
  *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
  *((volatile uint32_t*) GEMMINI_INST_ADDR) =                                 \
      (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

typedef uint8_t out_t;

// ---------------------------------------------------------------------------
// Load an array of scale factors (byte-packed) into a memory-mapped SF region.
// sf_mem : pointer to the hardware scale-factor MMIO region
// scale_factors : source byte array
// n : total number of bytes to transfer
// ---------------------------------------------------------------------------
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
    uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
    for (size_t i = 0; i < n / 8; i++) {
        sf_mem[i] = dword_scale_factors[i];
    }
}

// ---------------------------------------------------------------------------
// Configure Gemmini for weight-stationary MX execution, load LUTs and scale
// factors, mvin A and B tiles to the scratchpad, run the fused WS loop, then
// read results back from shared memory.
// ---------------------------------------------------------------------------
void run_mx_matmul(uint64_t C_hw[][MATMUL_N / 8]) {

    // -----------------------------------------------------------------------
    // 1. Configure execution: MX format, USE_LUT, weight-stationary
    // -----------------------------------------------------------------------
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
        ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
        | ((uint64_t)(1) << 16)       // A stride
        | (GEMMINI_FORMAT << 14)      // C format
        | (GEMMINI_FORMAT << 12)      // B format
        | (GEMMINI_FORMAT << 10)      // A format
        | (0 << 9)                    // B transpose
        | (0 << 8)                    // A transpose
        | ((false) << 7)              // set-only-strides flag
        | ((USE_LUT) << 4)            // enable LUT-based dequant
        | ((0) << 3)                  // activation function
        | ((WEIGHT_STATIONARY) << 2)
        | CONFIG_EX,
        ((uint64_t)(1) << 48)         // C stride
        | (0),
        k_CONFIG);

    // -----------------------------------------------------------------------
    // 2. Tile counts
    //    A_in_hw[MATMUL_M/2][MATMUL_K] : tiles_I x tiles_K, each DIM rows x DIM bytes
    //    B_in   [MATMUL_K][MATMUL_N/2] : tiles_K x tiles_J, each K_TILE rows x DIM bytes
    // -----------------------------------------------------------------------
    int tiles_I = MATMUL_M / 32;
    int tiles_K = MATMUL_K / 16;
    int tiles_J = MATMUL_N / 32;

    gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

    static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(32))) = {0};
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
                                 tiles_I, tiles_J, tiles_K,
                                 0, 0,
                                 QUANT_LUT_UPDATE_GRANULARITY);

    // -----------------------------------------------------------------------
    // 3. Write pre-packed LUTs (3 x uint32 per group) to MMIO LUT banks
    //    B_lut  -> LUT0 (weight),     indexed by n-group
    //    A_lut  -> LUT1 (activation), indexed by m-group
    //    C_lut  -> LUT2 (output),     indexed by m-group
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
        volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i;
        dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
    }
    for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
        volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i;
        dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
    }
    for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
        volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i;
        dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
    }

    // -----------------------------------------------------------------------
    // 4. Load per-block scale factors into hardware SF memory
    //    SF_MEM_A : row scales for A  (MATMUL_M * MATMUL_GK bytes)
    //    SF_MEM_B : col scales for B  (MATMUL_N * MATMUL_GK bytes)
    // -----------------------------------------------------------------------
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                       (uint8_t *) &A_scales_row, MATMUL_M * MATMUL_GK);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                       (uint8_t *) &B_scales_col, MATMUL_N * MATMUL_GK);
    gemmini_fence();

    // -----------------------------------------------------------------------
    // 5. Scratchpad address layout
    //    a_base : starts at 0; A tiles packed as (tiles_I * tiles_K) x DIM rows
    //    b_base : top of 8 KiB spad minus space for all B tiles
    // -----------------------------------------------------------------------
    uint32_t a_base = 0;
    uint32_t b_base = 8192 - tiles_K * tiles_J * K_TILE;

    // -----------------------------------------------------------------------
    // 6. MVIN A
    //    Stride = MATMUL_K bytes (full packed row width of A)
    //    Each tile: DIM rows x DIM bytes from A_in_hw[i*DIM : (i+1)*DIM][k*DIM : (k+1)*DIM]
    // -----------------------------------------------------------------------
    gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            uint8_t  *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
            uint32_t  sp_addr  = a_base + (i * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
            gemmini_fence();
        }
    }

    // -----------------------------------------------------------------------
    // 7. MVIN B
    //    Stride = MATMUL_N/2 bytes (full packed row width of B, 2 values/byte)
    //    Each tile: K_TILE rows x DIM bytes from B_in[k*K_TILE : ...][j*DIM : ...]
    // -----------------------------------------------------------------------
    gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
    for (int k = 0; k < tiles_K; k++) {
        for (int j = 0; j < tiles_J; j++) {
            uint8_t  *dram_ptr = (uint8_t *)B_in
                                 + k * K_TILE * (MATMUL_N / VALUES_PER_BYTE)
                                 + j * DIM;
            uint32_t  sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
            gemmini_fence();
        }
    }

    // -----------------------------------------------------------------------
    // 8. Fused weight-stationary loop over scratchpad tiles
    //    skip_mask = 0x38 : bits 3(ldA) | 4(ldB) | 5(ldD) set
    //      -> skip DRAM loads for A, B, D (already in spad); perform compute + store
    //
    //    Skip-mask bit reference:
    //      bit 3 (0x08) : ldA  - skip loading A from DRAM
    //      bit 4 (0x10) : ldB  - skip loading B from DRAM
    //      bit 5 (0x20) : ldD  - skip loading D/bias
    //      bit 6 (0x40) : ex   - skip compute
    //      bit 7 (0x80) : st   - skip acc->spad store
    // -----------------------------------------------------------------------
    int SPAD_DEST = 128;
    gemmini_loop_ws_spad(
        tiles_I, tiles_J, tiles_K,
        0, 0, 0,          // pad_I, pad_J, pad_K
        a_base,           // A scratchpad base address
        8192,             // B scratchpad end address (tiles addressed downward)
        0,                // D (bias) address - none
        SPAD_DEST,        // C accumulator destination address
        false, false,     // A_transpose, B_transpose
        false, false, false, // full_C, low_D, ex_accumulate
        NO_ACTIVATION,    // activation
        0, 0,             // a_spad_id, b_spad_id (double-buffer IDs)
        false,            // is_resadd
        0x38);            // skip_mask: skip ldA + ldB + ldD, run compute + store

    // -----------------------------------------------------------------------
    // 9. Read BF16-packed results from shared memory
    //    smem layout: each output row occupies MATMUL_N/8 uint64 words
    //    SPAD_DEST row offset in smem = SPAD_DEST * 2 uint64 words
    // -----------------------------------------------------------------------
    uint64_t *smem_start_addr = ((uint64_t *)SMEM) + SPAD_DEST * 2;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < MATMUL_N / 8; j++) {
            C_hw[i][j] = *(smem_start_addr + (i * MATMUL_N / 8 + j));
        }
    }
    gemmini_fence();
}
```

## matmul_tiled_fp4_64x64.c

SUMMARY: Demonstrates MX-Gemmini matmul kernel API usage including scale factor loading, tiled mvin for A/B matrices, fused weight-stationary loop execution, and BF16-packed result readback from shared memory using Gemmini RoCC intrinsics.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp4_64x64.h"

#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;
typedef uint64_t out_t;
```

```c
// Scale factor loader: fills SF memory with 0x7f (identity scale) for n/8 words
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = 0x7f7f7f7f7f7f7f7f;
  }
}
```

```c
// Gemmini MX setup: configure MX format and load per-block scale factors
gemmini_flush(0);
gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 3, 0);

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0); // sel=0 -> A
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1); // sel=1 -> B
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, 1024);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, 1024);
#endif
```

```c
// MVIN A: load tiles (i,k) into scratchpad at a_base + (i*tiles_K + k)*DIM
// Stride = MATMUL_M * sizeof(elem_t) bytes (row-major, full matrix width)
gemmini_config_ld(MATMUL_M * sizeof(elem_t));

for (int i = 0; i < tiles_I; i++) {
  for (int k = 0; k < tiles_K; k++) {
    elem_t *dram_ptr = ((elem_t*)A_in_hw) + i * DIM * MATMUL_M + k * DIM;
    uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}
```

```c
// MVIN B: load tiles (k,j) into scratchpad at b_base + (k*tiles_J + j)*DIM
// Stride = MATMUL_N * sizeof(elem_t) / 2 bytes (fp4 packing: 2 elems per byte)
gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);

for (int k = 0; k < tiles_K; k++) {
  for (int j = 0; j < tiles_J; j++) {
    elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N / 2 + j * DIM;
    uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}
```

```c
// Configure output stride and requantizer, then run fused WS loop
int SPAD_DEST = 128;
uint32_t scale_factors[512] __attribute__((aligned(32))) = {0};

gemmini_config_st(OUT_COLS * sizeof(out_t));
gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

gemmini_loop_ws_spad(
    tiles_I, tiles_J, tiles_K,
    0, 0, 0,
    a_base,
    BANK_NUM * BANK_ROWS,  // b_base (end of scratchpad)
    0,
    SPAD_DEST,
    false, false,
    false, false, false,
    NO_ACTIVATION,
    0, 0,
    false,
    0x38);
```

```c
// Read BF16-packed results from shared memory (non-SPIKE path)
// Each smem word holds 4x BF16; SPAD_DEST*2 offsets into uint64_t* smem base
#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      C_hw[i][j] = *(smem_start_addr + (i * OUT_COLS + j));
    }
  }
#endif

gemmini_fence();
```

```c
// Verify BF16-packed output against golden reference (4 bf16 lanes per uint64_t word)
for (int i = 0; i < MATMUL_M; i++) {
  for (int j = 0; j < OUT_COLS; j++) {
    uint64_t got = C_hw[i][j];

    uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                   ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                   ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                   ((uint64_t)C_out_bf16[i][j*4 + 0]);

    if (got != exp) {
      for (int lane = 0; lane < BF16_PER_WORD; lane++) {
        uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
        uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
        if (got_bf16 != exp_bf16) {
          printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                 i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
        }
      }
    }
  }
}
```

## matmul_tiled_fp8_64x64.c

SUMMARY: Demonstrates MX-Gemmini FP8 weight-stationary tiled matrix multiplication using Gemmini RoCC intrinsics, covering scale factor loading, scratchpad tiling/layout, fused loop execution, and BF16-packed result readback.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// Load per-block scale factors into Gemmini MX scale factor memory.
// INDIM: number of rows (A) or cols (B); K: reduction dimension.
// Scale factors are packed 8 per uint64_t word.
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
    for (size_t k = 0; k < K / 32; k++) {
        for (size_t i = 0; i < INDIM / 8; i++) {
            sf_mem[k * INDIM / 8 + i] = ((uint64_t *)scale_factors)[k * INDIM / 8 + i];
        }
    }
}
```

```c
// MX FP8 weight-stationary tiled matmul kernel (SPIKE_SIM path shown in comments).
// Assumes A_in, B_in, A_scales_row, B_scales_col, C_out_bf16 are provided by header.

void run_mx_fp8_matmul(out_t C_hw[MATMUL_M][OUT_COLS]) {
    int tiles_I = MATMUL_M / DIM;
    int tiles_J = MATMUL_N / DIM;
    int tiles_K = MATMUL_K / DIM;

    uint32_t a_base   = 0;
    uint32_t b_base   = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    int      SPAD_DEST = 128;

    uint32_t scale_factors[512] = {0};

    // 1. Flush and configure MX FP8 weight-stationary mode
    gemmini_flush(0);
    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 0, 0, 3, 0);

    // 2. Load per-block scale factors (sel=0 for A, sel=1 for B)
#ifdef SPIKE_SIM
    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
    load_scale_factors((volatile uint64_t *)GEMMINI_SF_MEM_A,
                       (uint8_t *)&A_scales_row, MATMUL_M, MATMUL_K);
    load_scale_factors((volatile uint64_t *)GEMMINI_SF_MEM_B,
                       (uint8_t *)&B_scales_col, MATMUL_N, MATMUL_K);
#endif

    // 3. MVIN A tiles: layout (i, k) -> scratchpad row a_base + (i*tiles_K + k)*DIM
    gemmini_config_ld(MATMUL_M * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t  *dram_ptr = ((elem_t *)A_in) + i * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *)dram_ptr, sp_addr, DIM, DIM);
        }
    }

    // 4. MVIN B tiles: layout (k, j) -> scratchpad row b_base + (j*tiles_K + k)*DIM
    for (int j = 0; j < tiles_J; j++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t  *dram_ptr = ((elem_t *)B_in) + j * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr  = b_base + (j * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *)dram_ptr, sp_addr, DIM, DIM);
        }
    }

    // 5. Configure output stride and MX requantizer/mvout parameters
    gemmini_config_st(OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout(
        (uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

    // 6. Fused weight-stationary loop over all tiles
    gemmini_loop_ws_spad(
        tiles_I, tiles_J, tiles_K,
        0, 0, 0,
        a_base,
        BANK_NUM * BANK_ROWS,
        0,
        SPAD_DEST,
        false, false,
        false, false, false,
        NO_ACTIVATION,
        0, 0,
        false,
        0x38);

    // 7. Read BF16-packed results from shared memory into output buffer
#ifdef SPIKE_SIM
    gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
    uint64_t *smem_start_addr = ((uint64_t *)SMEM) + SPAD_DEST * 2;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            C_hw[i][j] = *(smem_start_addr + (i * OUT_COLS + j));
        }
    }
#endif

    gemmini_fence();
}
```

```c
// Verify BF16-packed hardware output against golden reference.
// C_out_bf16[M][N] holds per-element uint16_t golden values;
// C_hw[M][OUT_COLS] holds 4 BF16 lanes packed into each uint64_t.
int verify_mx_output(out_t C_hw[MATMUL_M][OUT_COLS]) {
    int errors = 0;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            uint64_t got = C_hw[i][j];
            uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                           ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                           ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                           ((uint64_t)C_out_bf16[i][j*4 + 0]);
            if (got != exp) {
                for (int lane = 0; lane < BF16_PER_WORD; lane++) {
                    uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
                    uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
                    if (got_bf16 != exp_bf16) {
                        errors++;
                    }
                }
            }
        }
    }
    return errors;
}
```

## matmul_tiled_fp8_64x64_requant.c

SUMMARY: Demonstrates MX-Gemmini fp8 tiled matrix multiplication using weight-stationary dataflow, covering scale factor loading, scratchpad tiling/addressing, fused loop execution, and BF16-packed result readback from shared memory.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define DIM 16
#define ADDR_LEN 32

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// Load scale factors into MX scratchpad memory (8 bytes per word, value 0x7f = 1.0 in fp8)
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = 0x7f7f7f7f7f7f7f7f;
  }
}
```

```c
// Tiled MX fp8 weight-stationary matmul kernel
// Assumes MATMUL_M, MATMUL_N, MATMUL_K defined; DIM=16 systolic array tile size

static out_t C_hw[MATMUL_M][MATMUL_N/8];
uint32_t scale_factors[512] __attribute__((aligned(32))) = {0};
memset(C_hw, 0, sizeof(C_hw));

int tiles_I = MATMUL_M / DIM;
int tiles_J = MATMUL_N / DIM;
int tiles_K = MATMUL_K / DIM;

uint32_t a_base = 0;
uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
uint32_t acc_addr = (1u << (ADDR_LEN - 1));

// Configure Gemmini for weight-stationary MX fp8
gemmini_flush(0);
gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

// Load per-block scale factors for A (sel=0) and B (sel=1)
#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, 1024);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, 1024);
#endif

// MVIN A tiles: tile (i,k) -> scratchpad address a_base + (i*tiles_K + k)*DIM
gemmini_config_ld(MATMUL_M * sizeof(elem_t));
for (int i = 0; i < tiles_I; i++) {
  for (int k = 0; k < tiles_K; k++) {
    elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
    uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}

// MVIN B tiles: tile (k,j) -> scratchpad address b_base + (j*tiles_K + k)*DIM
for (int j = 0; j < tiles_J; j++) {
  for (int k = 0; k < tiles_K; k++) {
    elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
    uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}

int SPAD_DEST = 128;

// Configure output stride and requantizer (scale_factors buf, tile counts, format flags)
gemmini_config_st(1 * sizeof(out_t));
gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

// Fused weight-stationary loop over all tiles; results written to SPAD_DEST in shared mem
gemmini_loop_ws_spad(
    tiles_I, tiles_J, tiles_K,
    0, 0, 0,
    a_base,
    BANK_NUM * BANK_ROWS,
    0,
    SPAD_DEST,
    false, false,
    false, false, false,
    NO_ACTIVATION,
    0, 0,
    false,
    0x38);

// Read BF16-packed results from shared memory into output buffer
#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N / 2);
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < MATMUL_N / 8; j++) {
      C_hw[i][j] = *(smem_start_addr + (i * MATMUL_N / 8 + j));
    }
  }
#endif

gemmini_fence();
```

```c
// Elementwise correctness check: compare fp8 output bytes against golden reference
static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}

// Usage after kernel execution:
int errors = 0;
int diff1 = 0, diff2 = 0, diff3plus = 0;
uint8_t *hw_bytes = (uint8_t *)C_hw;

for (int i = 0; i < MATMUL_M; i++) {
  for (int j = 0; j < MATMUL_N; j++) {
    uint8_t got = hw_bytes[i * MATMUL_N + j];
    uint8_t exp = C_out[i][j];
    if (got != exp) {
      errors++;
      int bits = popcount8(got ^ exp);
      if      (bits == 1) diff1++;
      else if (bits == 2) diff2++;
      else                diff3plus++;
    }
  }
}
```

## autocomp_harness_test0.c

SUMMARY: Demonstrates the MX-Gemmini harness structure for fp8 64x64 weight-stationary matmul, including MMIO-based scale factor loading, packed BF16 output comparison, and the scratchpad/accumulator address layout used by the optimizable kernel region.

```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16
#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// Scale factor loader (RTL MMIO path; spike uses gemmini_mx_load_scales)
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
    }
  }
}

// Equality check over packed bf16 output words
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], out_t y[MATMUL_M][OUT_COLS]) {
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < OUT_COLS; j++)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

// Build packed golden output from MxGen reference (4 bf16 per word)
void build_gold_output(out_t gold[MATMUL_M][OUT_COLS]) {
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      gold[i][j] = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                   ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                   ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                   ((uint64_t)C_out_bf16[i][j*4 + 0]);
    }
  }
}

// Scratchpad/accumulator address layout for tiled WS matmul
void compute_spad_layout(int *tiles_I, int *tiles_J, int *tiles_K,
                         uint32_t *a_base, uint32_t *b_base,
                         uint32_t *acc_addr) {
  *tiles_I = MATMUL_M / DIM;
  *tiles_J = MATMUL_N / DIM;
  *tiles_K = MATMUL_K / DIM;
  *a_base  = 0;
  *b_base  = BANK_NUM * BANK_ROWS - (*tiles_K) * (*tiles_J) * DIM;
  *acc_addr = (1u << (ADDR_LEN - 1));
}
```

## mx_format_test.c

SUMMARY: Demonstrates MX format configuration for the Gemmini accelerator using `gemmini_extended3_config_ex`, showing how to set activation, weight, and output MX formats (FP8=0, FP6=1, FP4=2) and the `uselut` bit across various format combinations.

```c
#include <stdint.h>
#include "include/gemmini_testutils.h"

// MX format encoding: FP8=0, FP6=1, FP4=2

// Set all formats to FP8 (0)
// gemmini_extended3_config_ex(dataflow, act, sys_act, acc_scale, relu6_shift, stride,
//                              pool_stride, pool_size, pool_ceil_dim,
//                              act_mx_format, wt_mx_format, out_mx_format, uselut)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

// Set all formats to FP6 (1)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 1, 1);

// Set all formats to FP4 (2)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 2, 0);

// Mixed formats: act=FP8(0), wt=FP6(1), out=FP4(2)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 1, 3, 0);

// Mixed formats: act=FP4(2), wt=FP8(0), out=FP6(1)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 0, 1, 0);

// FP8 formats with uselut=1 enabled
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 1);
```

```c
// Memory-mapped RoCC instruction dispatch for Gemmini
#define GEMMINI_CTRL      0x40084000
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)

#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x200)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x380)

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                          \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12)  \
        | (1 << 15) | (2 << 20) | ((funct) << 25);                              \
}
```

## autocomp_baseline_sol0.c

SUMMARY: Demonstrates the MX-Gemmini fp8 weight-stationary matmul kernel API, covering configuration, scale factor loading, scratchpad tiling for A/B matrices, fused WS compute loop, and BF16 result readback using Gemmini RoCC intrinsics.

```c
// Configure execution: weight-stationary, MX requantize to bf16 output.
gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

// Load per-group scale factors into the scale-factor memory (A=0, B=1).
gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

// MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM
gemmini_config_ld(MATMUL_M * sizeof(elem_t));
for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
        elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
        uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
        gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
}

// MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM
for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
        elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
        uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
        gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
}

// Configure store + MX requantizer for the mvout, then run the WS compute loop.
gemmini_config_st(OUT_COLS * sizeof(out_t));
gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
gemmini_loop_ws_spad(
    tiles_I, tiles_J, tiles_K,
    0, 0, 0,
    a_base,
    BANK_NUM * BANK_ROWS,
    0,
    SPAD_DEST,
    false, false,
    false, false, false,
    NO_ACTIVATION,
    0, 0,
    false,
    0x38);

// Read the bf16-packed result out of shared memory into C_hw.
gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
gemmini_fence();
```