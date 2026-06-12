# MX-Gemmini Programming Model (microscaling fp8/fp6/fp4)

MX-Gemmini is a microscaling (MX) extension of the Gemmini systolic-array
accelerator. It performs low-precision floating-point matmul where each block of
elements shares a power-of-two scale factor (the "microscale"). Inputs are MX
floats (fp8/fp6/fp4); the systolic array accumulates in higher precision and the
result is requantized (typically to BF16) on the way out.

This document plus the example `matmul_tiled_*.c` test kernels and the Gemmini
intrinsics header (`gemmini.h`) define the kernels an optimizer works with. All
kernels here target the **spike functional model** (binaries built with
`-DSPIKE_SIM`), which provides the `gemmini_mx_*` intrinsics.

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

DIM = 16 (the systolic array / tile dimension). Tiling counts are
`tiles_I = M/DIM`, `tiles_J = N/DIM`, `tiles_K = K/DIM`. `BANK_NUM`/`BANK_ROWS`
come from `gemmini_params.h`.

## What an optimizer can tune

- **Tiling / loop order** of I/J/K and how tiles are laid out in the scratchpad
  (`a_base`/`b_base`, `SPAD_DEST`), to maximize scratchpad/accumulator reuse.
- **MVIN strategy**: block-mvin batching (load `batch` DIM-tiles per `gemmini_extended_mvin`,
  `cols = batch*DIM`) and strides (`gemmini_config_ld`). Validated envelope: batch=2 (cols=32) is the
  sweet spot; keep `cols <= 48` (batch <= 3) — the RTL `num_cols` field is only 6 bits, so `cols=64`
  truncates to 0 and the load is ILLEGAL. **Do NOT use `mvin2`/`mvin3` (port != 0) for `loop_ws_spad`
  inputs**: the RTL routes non-zero ports to a scratchpad region the loop_ws read ports cannot see, so
  the kernel is spike-correct but RTL-wrong. Use single-port `gemmini_extended_mvin` for loop_ws A/B.
- **Using `gemmini_loop_ws_spad`** (a fused hardware loop) vs hand-rolled
  preload/compute sequences.
- **Skips / flags** in `loop_ws_spad` (last arg is a skip bitmask, e.g. 0x38).
- Avoiding redundant scale/LUT reloads across tiles.

## Correctness

Output is compared against an MX "golden model" (PyTorch-simulated quantization,
generated by MxGen / `golden_model.py`) stored as BF16 in the test headers
(`C_out_bf16`). A kernel is correct only if the BF16-packed shared-memory result
matches the golden output exactly. Optimizations must preserve this.
