# MX-Gemmini Hardware Architecture Summary

## What the Hardware Is

MX-Gemmini is a microscaling (MX) floating-point matrix multiplication accelerator integrated as a RISC-V RoCC coprocessor. It extends the baseline Gemmini systolic array with native support for MX-format operands (FP8 E4M3, FP6 E3M2, FP4 E2M1) and per-block e8m0 scale factors, producing BF16-precision outputs. The programming model is instruction-driven: the host CPU issues RoCC custom-2 (`XCUSTOM_ACC`) instructions via C intrinsics to configure formats, load data and scales into on-chip memories, execute the fused hardware matmul loop, and drain results to DRAM. All kernels run on the Spike functional simulator (`-DSPIKE_SIM`), which provides bit-exact BF16 output matching the MX golden model.

## Programming Model Sequence

Every matmul invocation must follow this ordering or correctness is violated:

1. **`gemmini_extended3_config_ex`** — sets MX format fields (`act_mx_fmt`[11:10], `wgt_mx_fmt`[13:12], `out_mx_fmt`[15:14]), `uselut`[5], dataflow (weight-stationary=1), transposes, strides, and activation function in CONFIG_EX RS1/RS2.
2. **`gemmini_mx_load_scales(dram_addr, len, sel)`** (funct 27) — streams e8m0 scale codes from DRAM into `mx_scale_a_mem` (sel=1) or `mx_scale_b_mem` (sel=0). Must be issued for both A and B before compute.
3. **`gemmini_mx_load_lut(dram_addr, num_luts, sel)`** (funct 29) — loads FP6 LUT banks (16 × 6-bit codes per LUT, packed as 3 × uint32 per LUT). Required only when `uselut=1`. LUT banks persist until overwritten, so reloading is unnecessary when the same weight block is reused across tiles.
4. **`gemmini_config_ld` + `gemmini_extended_mvin{,2,3}`** — configure DRAM stride and DMA tiles into scratchpad (A via mvin, B via mvin2) or accumulator (D/bias via mvin3).
5. **`gemmini_config_st` + `gemmini_mxquant_config_mvout`** (funct 26) — configure output stride and the requantizer: `scale_dram`, `i_bound`/`j_bound`/`k_bound` (9-bit tile counts), `scale_act_sel`/`scale_w_sel` (which loaded scale buffer feeds requant), and `lut_update_granularity` G (controls how often the output LUT refreshes during quantization; larger G = fewer refreshes = faster).
6. **`gemmini_loop_ws_spad`** — issues funct 24 (`LOOP_WS_CONFIG_SPAD_AB`, sets one-shot `mx_loop_spad_marker`) then funct 8 (`LOOP_WS`). The marker is consumed on execution; funct 24 must be re-issued before every invocation. Executes the entire tiled I×J×K weight-stationary matmul in hardware.
7. **`gemmini_mx_read_smem(dram_addr, smem_off_bf16, num_bf16)`** (funct 28) + **`gemmini_fence()`** — drains BF16-packed results from `mx_smem` to DRAM. Results are packed 4 BF16 values per uint64 word. The fence must complete before the host reads output.

## Memory Hierarchy

### Scratchpad (SRAM)
- **Size:** 4 banks × 2048 rows = 8192 rows total; each row is `DIM × sizeof(elem_t)` = 16 × 1 = **16 bytes**; total **128 KB**
- **Addressing:** word offsets (not byte addresses); programmer-managed layout
- **A tile layout:** `a_base + (i*tiles_K + k)*DIM` row offset
- **B tile layout:** `b_base + (j*tiles_K + k)*DIM`; B is placed at the **top** of the scratchpad: `b_base = BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`, growing downward to avoid overlap with A
- **DMA limits:** `MAX_BLOCK_LEN = 4` — up to 4 consecutive DIM-row tiles per single mvin command (MAX_BYTES=64, elem_t=1 byte → 64/(16×1)=4)
- **Double-buffering:** scratchpad is logically split into two halves (`BANK_NUM*BANK_ROWS/2` rows each); `a_spad_id`/`b_spad_id` (0=no reuse, 1=first buffer, 2=second buffer) control operand reuse in `gemmini_loop_ws_spad`

### Accumulator
- **Size:** 256 rows × 128 bytes/row = **32 KB**; row width = `DIM × sizeof(acc_t)` = 16 × 8 bytes
- **DMA limits:** `MAX_BLOCK_LEN_ACC = 1` — only 1 accumulator tile per mvin command (128-byte rows exceed MAX_BYTES=64); accumulator mvins are 4× more command-intensive than scratchpad mvins
- **Address space:** high bit set (`1 << (ADDR_LEN-1)`) for D/bias input; bits [ADDR_LEN-1:ADDR_LEN-2] = `11` for C output; double-buffer partition at `ACC_ROWS/2 = 128` rows
- **Maximum in-flight output tiles:** 256 (single-buffer) or 128 (double-buffer); must mvout before overflow

### MX Shared Memory (`mx_smem`)
- Holds BF16-packed outputs (or packed FP4/FP6 codes) after `gemmini_loop_ws_spad` completes
- Addressed in 16-bit word units; C spad word offset = `SPAD_DEST * 16`
- Drained exclusively via `MX_READ_SMEM`; `MVOUT_SPAD` (funct 23) is a no-op in Spike

### Scale Memory
- Two independent buffers: `mx_scale_a_mem` (activation row scales) and `mx_scale_b_mem` (weight column scales)
- Scale format: **e8m0** (8-bit exponent only, pure power-of-two); one scale per MX block along K
- DRAM layout: `scale_dram + m * N_blocks + bi` (row-major, block-indexed)
- Persist in hardware state until overwritten; reloading is only necessary when the scale block changes

### LUT Memory
- Three banks: `mx_lut_a`, `mx_lut_b`, `mx_lut_c`; each entry = 16 × 6-bit FP6 E3M2 codes
- DRAM packing: 3 × LE uint32 per LUT (96 bits per LUT)
- Persist until overwritten; amortize across all tiles sharing the same weight block

### Off-Chip DRAM
- Source for all tile data, scales, and LUTs; destination for BF16 results via `MX_READ_SMEM`
- No hardware cache between DRAM and scratchpad; all data movement is explicit DMA

## Compute Unit

- **Type:** 16×16 weight-stationary systolic array
- **Tile dimension:** DIM = 16; all tiling must use multiples of 16 in M, N, K
- **Input element type:** `elem_t = uint8_t` (fp8/fp6/fp4 packed into 1-byte slots); `elem_t_max=4095`, `elem_t_min=0`
- **Accumulator type:** `acc_t = uint64_t`; BF16-class precision (8 exponent bits, 8 significand bits)
- **Output scale type:** `acc_scale_t = float` (IEEE 754 float32)
- **Fused hardware loop:** `gemmini_loop_ws_spad` executes the entire tiled I×J×K matmul in hardware without software loop overhead; tile counts `tiles_I = M/DIM`, `tiles_J = N/DIM`, `tiles_K = K/DIM`
- **Transpose support:** A-transpose (bit 8) and B-transpose (bit 9) in CONFIG_EX RS1
- **`skips` bitmask** in `gemmini_loop_ws_spad` RS2[5:0]: suppresses sub-operations (e.g., redundant preloads or bias adds); tunable for performance without affecting correctness when the skipped operation is genuinely unnecessary
- **Throughput:** one 16×16 MAC array; peak utilization requires keeping the systolic array fed with back-to-back tiles, which demands overlapping mvin with compute and minimizing configuration re-issue

## Key Constraints Affecting Optimization

**Tiling constraints:**
- Scratchpad must hold all A and B tiles simultaneously: `(tiles_I*tiles_K + tiles_J*tiles_K)*DIM ≤ 8192` rows (single-buffer) or `≤ 4096` (double-buffer)
- Accumulator must hold all C tiles: `tiles_I*tiles_J*DIM ≤ 256` rows (single-buffer) or `≤ 128` (double-buffer)
- Tile dimension fields are 16-bit; I, J, K tile counts must each be < 65535
- M, N, K must be exact multiples of DIM=16 (no partial-tile hardware padding in MX path)

**Scale and LUT amortization:**
- `gemmini_mx_load_scales` and `gemmini_mx_load_lut` are DRAM loads with non-trivial latency; they must be amortized across all tiles that share the same scale block or LUT
- `lut_update_granularity` G in `gemmini_mxquant_config_mvout` controls requantizer LUT refresh frequency; set G as large as correctness allows to minimize refresh overhead
- Scale selectors (`scale_act_sel`, `scale_w_sel`) choose between two loaded scale buffers, enabling ping-pong scale loading while compute proceeds

**Instruction ordering and statefulness:**
- `mx_loop_spad_marker` is one-shot: funct 24 must be re-issued before every `gemmini_loop_ws_spad` call
- CONFIG_EX, CONFIG_LD, and CONFIG_ST fields persist until changed; avoid redundant re-issue when format, stride, and scale are unchanged across consecutive tiles
- `set_only_strides` flag (CONFIG_EX RS1 bit 7) allows updating A/C strides without full reconfiguration
- `gemmini_fence()` is a full memory barrier; use only when necessary (after `gemmini_mx_read_smem`, before consuming results)

**Bandwidth and scheduling:**
- Three independent mvin ports (mvin/mvin2/mvin3) can overlap A, B, and D loads with each other and with compute
- Accumulator mvin is 4× more command-intensive than scratchpad mvin (MAX_BLOCK_LEN_ACC=1 vs. 4); minimize partial accumulator writes
- B matrix must be placed at the top of the scratchpad (`BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`) before tiling begins; verify no overlap with A before issuing mvins
- `MVOUT_SPAD` (funct 23) is a no-op in Spike; all output must flow through `mx_smem` → `MX_READ_SMEM` → DRAM

**Correctness:**
- Output must be bit-exact BF16 matching the MX golden model; no approximations are permitted
- Scale memory layout in DRAM must match the expected `scale_dram + m*N_blocks + bi` indexing exactly
- LUT banks are indexed by the `sel` field (0=B, 1=A, 2=C); loading to the wrong bank silently produces incorrect results