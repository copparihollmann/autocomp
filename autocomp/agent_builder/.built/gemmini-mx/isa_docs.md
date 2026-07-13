## Configuration and Setup

### gemmini_extended3_config_ex

// RS1: [63:32] acc_scale | [31:16] a_stride | [15:14] out_mx_fmt | [13:12] wgt_mx_fmt | [11:10] act_mx_fmt | [9] b_transpose | [8] a_transpose | [7] set_only_strides | [6] spacer | [5] uselut | [4:3] activation | [2] dataflow | [1:0] cmd_type
// RS2: [63:48] c_stride | [47:32] relu6_shift | [31:0] in_shift
#define gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides, act_mx_fmt, wgt_mx_fmt, out_mx_fmt, uselut) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
        ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | \
        ((uint64_t)(A_stride) << 16) | \
        ((uint64_t)(out_mx_fmt) << 14) | \
        ((uint64_t)(wgt_mx_fmt) << 12) | \
        ((uint64_t)(act_mx_fmt) << 10) | \
        ((uint64_t)(B_transpose) << 9) | \
        ((uint64_t)(A_transpose) << 8) | \
        ((uint64_t)(set_only_strides) << 7) | \
        ((uint64_t)(uselut) << 5) | \
        ((uint64_t)(sys_act) << 3) | \
        ((uint64_t)(dataflow) << 2) | \
        CONFIG_EX, \
        ((uint64_t)(C_stride) << 48) | (uint64_t)(uint32_t)(sys_shift), \
        k_CONFIG)

### gemmini_extended3_config_ex

1. **Configure execution** — set dataflow + the MX format fields:
   ```c
   gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
       /*C_stride*/1, /*A_stride*/1, /*A_transpose*/0, /*B_transpose*/0,
       /*set_only_strides*/false,
       /*act_mx_fmt*/0, /*wgt_mx_fmt*/0, /*out_mx_fmt*/3, /*uselut*/0);
   ```
   `act_mx_fmt`/`wgt_mx_fmt`/`out_mx_fmt` are 2-bit fields (config_ex RS1 bits
   [11:10],[13:12],[15:14]). For fp8 the example uses act=0,wgt=0,out=3 (BF16
   output). **For fp6/fp4 use the values shown in the corresponding
   `matmul_tiled_fp6_*.c` / `matmul_tiled_fp4_*.c` example — do not guess the
   encoding.** Set `uselut=1` and load a LUT (below) when the format requires it.

---

### gemmini_extended2_config_ex

#define gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false, 0, 0, 0, 0)

---

### gemmini_extended_config_ex

#define gemmini_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

---

### gemmini_extended5_config_ld

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

---

### gemmini_extended4_config_ld

#define gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

---

### gemmini_extended3_config_ld

#define gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

---

### gemmini_extended2_config_ld

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(stride, scale, shrunk, 0)

---

### gemmini_extended_config_ld

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

---

### gemmini_extended2_config_st

#define gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

---

### gemmini_extended_config_st

#define gemmini_extended_config_st(stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

---

### gemmini_mxquant_config_mvout

#define gemmini_mxquant_config_mvout(dram_addr, i_bound, j_bound, k_bound, scale_act_sel, scale_w_sel, lut_update_granularity) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   ((uint64_t)(scale_w_sel) << 61) | ((uint64_t)(scale_act_sel) << 60) |  ((uint64_t)(k_bound) << 51) | ((uint64_t)(j_bound) << 42) | ((uint64_t)(i_bound) << 33) | (uint64_t)(dram_addr), \
    (uint64_t)(lut_update_granularity) & 0xFFFF, CONFIG_SCALE_MEM)

### gemmini_mxquant_config_mvout

5. **Configure the requantized mvout** — programs the requantizer + output scale
   memory and bounds:
   ```c
   gemmini_config_st(OUT_COLS * sizeof(out_t));
   gemmini_mxquant_config_mvout((uint64_t)scale_factors,
       /*i_bound*/tiles_I, /*j_bound*/tiles_J, /*k_bound*/tiles_K,
       /*scale_act_sel*/0, /*scale_w_sel*/0, /*lut_update_granularity*/1);
   ```

---

### MXQUANT_CONFIG_MVOUT

| 26 | `MXQUANT_CONFIG_MVOUT`    | `[32:0]` = `scale_dram` (low 33 bits), `[33:42]` = `tiles_I`, `[42:51]` = `tiles_J`, `[51:60]` = `tiles_K`, `[60]` = `scale_act_sel`, `[61]` = `scale_wgt_sel` | `[15:0]` = `lut_update_granularity` (G) | Configures the requant write-back. Per-row, per-N-group e8m0 scale codes are stored at `scale_dram + m*N_blocks + bi`. |

---

### CONFIG_EX MX format fields

### CONFIG_EX changes (funct 0, CONFIG_EX subtype, `rs1[1:0] == 0b00`)

`gemmini_extended3_config_ex` now packs four MX format fields into
previously-unused bits of `rs1`; the macro signature changed to take
`(act_fmt, wgt_fmt, out_fmt, uselut)` as the last four arguments. The
spike handler reads:

| `rs1` bits | Field | Meaning |
|------------|-------|---------|
| `[5]`     | `uselut`  | When 1, the MX kernel takes the FP6 LUT path. |
| `[11:10]` | `act_fmt` | A-side / activation format: 0 = FP8 E4M3, 1 = FP6 E3M2 (LUT-indexed), 2 = FP4 E2M1. Selects the matmul kernel. |
| `[13:12]` | `wgt_fmt` | B-side / weight format (same encoding). |
| `[15:14]` | `out_fmt` | Output format: 0 = FP8 E4M3 packed, 1 = FP6 (4-bit LUT indices), 2 = FP4 E2M1 packed, 3 = BF16. Selects whether a requant post-pass runs and which projection it uses. |

All other config-EX bits (dataflow, sys_act, strides, transposes, sys
shifts) keep their pre-existing meanings.

## Control Flow and Synchronization

### gemmini_flush

#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

---

### gemmini_fence

#define gemmini_fence() asm volatile("fence")

## Data Types and Formats

### MX format table

## Supported MX formats

| Format | Enc | Notes |
|--------|-----|-------|
| FP4 (E2M1) | 4-bit | 2 elements packed (dual) |
| FP6 E2M3 | 6-bit | single |
| FP6 E3M2 | 6-bit | dual |
| FP8 E4M3 | 8-bit | single — the most common input format |
| FP8 E5M2 | 8-bit | dual |

Each MX block shares one `fpe8m0` (power-of-two) scale factor. Scale factors are
grouped: the golden headers expose `A_scales_row[GK][M]` and `B_scales_col[GK][N]`
where `GK` is the number of scale groups along K.

---

### gemmini_state_t MX fields

### State added to `gemmini_state_t`

`mx_act_fmt`, `mx_wgt_fmt`, `mx_out_fmt`, `mx_use_lut`,
`mx_lut_update_granularity`, `mx_scale_dram`, `mx_tiles_{I,J,K}`,
`mx_scale_{act,wgt}_sel`, `mx_loop_{a,b,c}_spad`, `mx_loop_skips`,
`mx_loop_spad_marker`, plus the buffers `mx_scale_a_mem`,
`mx_scale_b_mem`, `mx_smem`, and per-format LUTs `mx_lut_{a,b,c}` (each
16 codes × max LUTs). All cleared on `reset()`.


## Memory Load Operations

### gemmini_extended_mvin

4. **MVIN tiles** of A and B into the scratchpad. Tiles are DIM×DIM (DIM=16):
   ```c
   gemmini_config_ld(MATMUL_M * sizeof(elem_t));
   gemmini_extended_mvin(dram_ptr, spad_addr, DIM, DIM);
   ```
   Lay A tiles at `a_base + (i*tiles_K + k)*DIM`, B tiles at
   `b_base + (j*tiles_K + k)*DIM` where `b_base = BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`.

### gemmini_extended_mvin

#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

---

### gemmini_extended_mvin2

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

---

### gemmini_extended_mvin3

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

---

### gemmini_block_mvin

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

## Memory Store Operations

### gemmini_extended_mvout

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

---

### gemmini_extended_mvout_spad

#define gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

---

### gemmini_mvout_spad

#define gemmini_mvout_spad(dst_addr, src_addr) \
  gemmini_extended_mvout_spad(dst_addr, 1, src_addr, DIM, DIM)

## MX Scale and LUT Management

### gemmini_mx_load_scales

2. **Load scale factors** into scale-factor memory (sel 0 = activations/A,
   sel 1 = weights/B):
   ```c
   gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
   gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
   ```
   (RTL builds instead MMIO-write the scale-factor memory at GEMMINI_SF_MEM;
   spike uses the intrinsic. Optimize for the spike path.)

### gemmini_mx_load_scales

#define gemmini_mx_load_scales(dram_addr, len, sel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(sel) << 32) | ((uint64_t)(len) & 0xFFFFFFFFu), \
    k_MX_LOAD_SCALES)

---

### gemmini_mx_load_lut

// Load num_luts × (16 6-bit FP6 codes) from DRAM (3 LE uint32 per LUT).
// sel: 0 = B (weight), 1 = A (activation), 2 = C (output)
#define gemmini_mx_load_lut(dram_addr, num_luts, sel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(sel) << 32) | ((uint64_t)(num_luts) & 0xFFFFFFFFu), \
    k_MX_LOAD_LUT)

### gemmini_mx_load_lut

3. **(Optional) Load a quant LUT** with `gemmini_mx_load_lut(dram_addr, num_luts, sel)`
   for formats/requantization that need it (see `matmul_data_mx_lut_hw.h` users).

---

### MX_LOAD_SCALES

| 27 | `MX_LOAD_SCALES`          | DRAM byte address of scale block | `[32]` = `sel` (0 = A-row scales, 1 = B-col scales), `[31:0]` = `len` (bytes) | Streams `len` e8m0 codes from DRAM into `mx_scale_a_mem` / `mx_scale_b_mem`. |

---

### MX_LOAD_LUT

| 29 | `MX_LOAD_LUT`             | DRAM byte address | `[33:32]` = `sel` (0 = B, 1 = A, 2 = C), `[31:0]` = `num_luts` | Loads `num_luts × (3 LE uint32 = 96 bits)` from DRAM and unpacks 16 × 6-bit FP6 E3M2 codes per LUT into `mx_lut_a / b / c`. |

## MX Shared Memory Operations

### gemmini_mx_read_smem

#define gemmini_mx_read_smem(dram_addr, smem_off_bf16, num_bf16) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(num_bf16) << 32) | ((uint64_t)(smem_off_bf16) & 0xFFFFFFFFu), \
    k_MX_READ_SMEM)

### gemmini_mx_read_smem

7. **Read the result** out of shared memory (BF16, packed 4 per uint64 word):
   ```c
   gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST*16, MATMUL_M*MATMUL_N);
   gemmini_fence();
   ```

---

### MX_READ_SMEM

| 28 | `MX_READ_SMEM`            | DRAM byte address | `[63:32]` = `num_u16`, `[31:0]` = `smem_word_offset` | Copies `num_u16` 16-bit words from the MX shared memory (`mx_smem`) to DRAM. Used to drain BF16 outputs or packed FP4 / FP6 codes after the matmul. |

## Compute and Systolic Array Operations

### gemmini_extended_compute_preloaded

#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

---

### gemmini_extended_compute_accumulated

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

---

### gemmini_extended_preload

#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

---

### gemmini_preload

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

---

### gemmini_preload_zeros

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

## Fused Loop and Tiled Matmul

### gemmini_loop_ws

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

---

### gemmini_loop_ws_spad

#define gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_SPAD_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((uint64_t)(C) << 32) | 0x200U | (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

### gemmini_loop_ws_spad

6. **Compute** the weight-stationary tiled matmul over the scratchpad:
   ```c
   gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0,0,0,
       a_base, BANK_NUM*BANK_ROWS, 0, SPAD_DEST,
       false,false, false,false,false, NO_ACTIVATION, 0,0, false, 0x38);
   ```
   Results land in shared memory at `SPAD_DEST`, requantized to BF16.

---

### LOOP_WS_CONFIG_SPAD_AB

| 24 | `LOOP_WS_CONFIG_SPAD_AB`  | A spad base address | B spad end address | Sets `mx_loop_a_spad`, `mx_loop_b_spad`. Also marks the next `LOOP_WS` as the MX (`mx_loop_ws_spad`) variant. |

---

### LOOP_WS_CONFIG_SPAD_C

| 25 | `LOOP_WS_CONFIG_SPAD_C`   | C spad base address | – | Sets `mx_loop_c_spad`. |

---

### LOOP_WS (MX variant)

The MX matmul itself reuses `LOOP_WS` (funct 8). When the preceding
instruction was `LOOP_WS_CONFIG_SPAD_AB`, the spike handler dispatches
to `mx_loop_ws_spad` instead of the legacy CISC loop; `rs1` is unused
and in `rs2` the upper 32 bits are the C-side spad word offset, while
`rs2[5:0]` is the existing skip-mask byte.

## Reference Documentation and Workflows

### MX intrinsics summary

## Key MX intrinsics (from gemmini.h)

- `gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides, act_mx_fmt, wgt_mx_fmt, out_mx_fmt, uselut)`
- `gemmini_mx_load_scales(dram_addr, len, sel)` — sel 0=A, 1=B
- `gemmini_mx_load_lut(dram_addr, num_luts, sel)`
- `gemmini_mxquant_config_mvout(dram_addr, i_bound, j_bound, k_bound, scale_act_sel, scale_w_sel, lut_update_granularity)`
- `gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips)`
- `gemmini_mx_read_smem(dram_addr, smem_off_bf16, num_bf16)`
- Standard Gemmini intrinsics also apply: `gemmini_extended_mvin/mvin2/mvin3`,
  `gemmini_extended_mvout`, `gemmini_preload`, `gemmini_compute_preloaded/accumulated`,
  `gemmini_config_ld/st`, `gemmini_fence`, `gemmini_flush`.

---

### Kernel workflow (weight-stationary fp8 matmul)

## The kernel workflow (weight-stationary fp8 matmul)

A correct MX matmul kernel performs these steps (see
`matmul_tiled_fp8_64x64.c` for the canonical example):

1. **Configure execution** — set dataflow + the MX format fields:
   ```c
   gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
       /*C_stride*/1, /*A_stride*/1, /*A_transpose*/0, /*B_transpose*/0,
       /*set_only_strides*/false,
       /*act_mx_fmt*/0, /*wgt_mx_fmt*/0, /*out_mx_fmt*/3, /*uselut*/0);
   ```
   `act_mx_fmt`/`wgt_mx_fmt`/`out_mx_fmt` are 2-bit fields (config_ex RS1 bits
   [11:10],[13:12],[15:14]). For fp8 the example uses act=0,wgt=0,out=3 (BF16
   output). **For fp6/fp4 use the values shown in the corresponding
   `matmul_tiled_fp6_*.c` / `matmul_tiled_fp4_*.c` example — do not guess the
   encoding.** Set `uselut=1` and load a LUT (below) when the format requires it.

2. **Load scale factors** into scale-factor memory (sel 0 = activations/A,
   sel 1 = weights/B):
   ```c
   gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
   gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
   ```
   (RTL builds instead MMIO-write the scale-factor memory at GEMMINI_SF_MEM;
   spike uses the intrinsic. Optimize for the spike path.)

3. **(Optional) Load a quant LUT** with `gemmini_mx_load_lut(dram_addr, num_luts, sel)`
   for formats/requantization that need it (see `matmul_data_mx_lut_hw.h` users).

4. **MVIN tiles** of A and B into the scratchpad. Tiles are DIM×DIM (DIM=16):
   ```c
   gemmini_config_ld(MATMUL_M * sizeof(elem_t));
   gemmini_extended_mvin(dram_ptr, spad_addr, DIM, DIM);
   ```
   Lay A tiles at `a_base + (i*tiles_K + k)*DIM`, B tiles at
   `b_base + (j*tiles_K + k)*DIM` where `b_base = BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`.

5. **Configure the requantized mvout** — programs the requantizer + output scale
   memory and bounds:
   ```c
   gemmini_config_st(OUT_COLS * sizeof(out_t));
   gemmini_mxquant_config_mvout((uint64_t)scale_factors,
       /*i_bound*/tiles_I, /*j_bound*/tiles_J, /*k_bound*/tiles_K,
       /*scale_act_sel*/0, /*scale_w_sel*/0, /*lut_update_granularity*/1);
   ```

6. **Compute** the weight-stationary tiled matmul over the scratchpad:
   ```c
   gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0,0,0,
       a_base, BANK_NUM*BANK_ROWS, 0, SPAD_DEST,
       false,false, false,false,false, NO_ACTIVATION, 0,0, false, 0x38);
   ```
   Results land in shared memory at `SPAD_DEST`, requantized to BF16.

7. **Read the result** out of shared memory (BF16, packed 4 per uint64 word):
   ```c
   gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST*16, MATMUL_M*MATMUL_N);
   gemmini_fence();
   ```

---

### Tiling parameters and scratchpad layout

DIM = 16 (the systolic array / tile dimension). Tiling counts are
`tiles_I = M/DIM`, `tiles_J = N/DIM`, `tiles_K = K/DIM`. `BANK_NUM`/`BANK_ROWS`
come from `gemmini_params.h`.

---

### Optimizer tuning knobs

## What an optimizer can tune

- **Tiling / loop order** of I/J/K and how tiles are laid out in the scratchpad
  (`a_base`/`b_base`, `SPAD_DEST`), to maximize scratchpad/accumulator reuse.
- **MVIN strategy**: batching, strides (`gemmini_config_ld`), which of
  mvin/mvin2/mvin3 ports to use, overlapping loads with compute.
- **Using `gemmini_loop_ws_spad`** (a fused hardware loop) vs hand-rolled
  preload/compute sequences.
- **Skips / flags** in `loop_ws_spad` (last arg is a skip bitmask, e.g. 0x38).
- Avoiding redundant scale/LUT reloads across tiles.