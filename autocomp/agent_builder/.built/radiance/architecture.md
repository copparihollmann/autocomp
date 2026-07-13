## Hardware Architecture Summary: Radiance Muon + MX-Gemmini Cluster

### System Overview

The target is a single Radiance cluster consisting of two Muon SIMT cores and one MX-Gemmini systolic array accelerator. The SIMT cores drive the accelerator via MMIO stores into shared address space; the accelerator scratchpad is physically identical to the cluster shared memory. All matrix computation must go through the accelerator (~100× SIMT FP throughput; SIMT-side matmul is both slower and produces different rounding). The verdict is bit-exact against hardware MX accumulation semantics.

### SIMT Cores

Each Muon core runs 2 warps of 16 lanes (32 threads/core), giving 4 warps and 64 total threads per threadblock across both cores. The ISA is RV32IM + Zfinx (floats in integer registers; no separate FPU register file). Warp width is fixed at 16 lanes. Issue is single-wide: 1 warp-instruction/cycle maximum. FP32 throughput is 0.5 FLOP/thread/cycle (8 FP32 PEs); INT32 is 1 FLOP/thread/cycle (16 INT PEs).

The physical register file is 16 KiB shared across all active warp slots. At 2 warps/core (the safe operating point given `MX_NUM_WARPS=2`), each thread has up to 128 architectural registers available (128 regs × 16 threads × 4 bytes = 8 KiB/warp × 2 warps = 16 KiB). Exceeding 8 warps total overflows the physical register file. RF read bandwidth is 192 bytes/cycle (no-conflict minimum for 3 operands); 4-operand instructions (256 bytes) cause stalls.

Available intrinsics: `mu_schedule(kernel_body, args, NUM_WARPS)`, `mu_barrier(id, num_warps)`, `mu_fence`, `mu_exp(float)`. Forbidden: `expf`/`exp` stdlib, direct `mxgemmini`/tensor ISA from SIMT (use MMIO instead), `main`/`KernelArgs`/`mu_exp` redefinition, printing from kernels.

### Memory Hierarchy

**Shared Memory / Accelerator Scratchpad (SMEM):** 128 KiB per cluster, base address `0x0` in GPU-local shared space. This is simultaneously the cluster SMEM and the MX-Gemmini scratchpad — the same physical SRAM. Organized as 4 banks × 2048 rows × 16 bytes/row. SMEM bandwidth is 64 bytes/cycle for SIMT (one full-warp store saturates it); aggregate 256 bytes/cycle with simultaneous read+write ports. Bank arbitration is round-robin between cores; Gemmini gets lowest priority (programmer must tile to avoid starvation). SIMT stores to SMEM are the mechanism for staging operands and scales before issuing accelerator commands.

**L0 Data Cache:** 16 KiB/core, direct-mapped, write-through, 64 bytes/cycle bandwidth, 3-cycle read latency. Does not apply to SMEM accesses.

**L1 Cache:** 64 KiB/cluster, 4-way set associative, write-through, no write-allocate, 32 bytes/cycle bandwidth, 2 banks. Shared across both cores; multi-core accesses serialize. C move-out from SMEM to global memory is bounded by L1 bandwidth.

**L2 Cache:** 512 KiB, 8-way, coherent, 32 bytes/cycle bandwidth (SBus = 256-bit), ~20-cycle hit latency.

**DRAM:** ~100-cycle latency (GPU-local). GPU DRAM is CPU-addressable at `0x1_0000_0000`. GPU address translation appends bit 33 on L1 egress.

**Fixed SMEM Address Map (GPU-local):**
- `0x00000000`: 128 KiB cluster SMEM / Gemmini scratchpad
- `0x00084000` (`GEMMINI_CTRL`): Gemmini MMIO command block
- `0x00084080` / `0x84380` / `0x84680`: LUT0 (weights), LUT1 (activations), LUT2 (output)
- `0x00088000` (`SF_MEM_B`): e8m0 block scales for B (weights)
- `0x0008A000` (`SF_MEM_A`): e8m0 block scales for A (activations)
- `SF_MEM_* + 0x800`: double-buffer offset for odd K-tiles
- `0x00040000` (`GEMMINI_REQUANT`): 256 KiB requantization region

### MX-Gemmini Accelerator

A 16×16 weight-stationary systolic array. Tile dimension DIM=16 for all operations; all matrix dimensions must be multiples of 16. Scratchpad layout: 4 banks × 2048 rows = 8192 rows total; A tiles start at row 0, B tiles at `BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`, C accumulator at row 256 (byte offset 4096). Accumulator is 256 rows × 128 bytes/row = 32 KiB, addressed separately (ADDR_LEN=32, high bits distinguish SPAD vs ACC).

**Supported formats:** FP8 E4M3 (8-bit, 1 element/slot), FP6 E3M2 (6-bit, 2 elements/slot, LUT path), FP4 E2M1 (4-bit, 2 elements/slot). Scale format: e8m0 (8-bit exponent-only), one scale per 32-element K-block. The tapeout config (`MxConfig.mxGemmini`) supports only same-format pairings (modes 0, 4, 8): FP4×FP4 (4 outputs/PE/cycle), FP6×FP6 (4 outputs/PE/cycle), FP8×FP8 (1 output/PE/cycle). Cross-format pairings are not supported. Act and weight formats must match per tile.

**PE throughput by format:** FP8 E4M3 × FP8 E4M3 = mode 8, 1 output/PE/cycle; FP6 E3M2 × FP6 E3M2 = mode 4, 4 outputs/PE/cycle; FP4 E2M1 × FP4 E2M1 = mode 0, 4 outputs/PE/cycle.

**Accumulation:** Anchor-aligned (MX semantics), not IEEE sequential FP addition. Results are BF16-packed in scratchpad (4 BF16 per uint64 word) at `SPAD_DEST`. Accumulation is order-sensitive; K must be traversed ascending. The `ex_accumulate` bit in `LOOP_WS` RS1 must be set for every K-tile after the first to ADD into existing C rather than overwrite.

**MVIN constraints:** `cols` field is 6-bit in RTL — maximum 48 (batch ≤ 3); `cols=64` truncates to 0 and is illegal. Validated sweet spot: batch=2 (cols=32). For `loop_ws_spad` inputs, use only port-0 `gemmini_extended_mvin`; `mvin2`/`mvin3` route to scratchpad regions invisible to `loop_ws` read ports (spike-correct but RTL-wrong, silent failure).

### MMIO Command Interface

Base address `0x84000`. Register map:

| Offset | Register | Direction |
|--------|----------|-----------|
| `+0x00` | INST | Write (triggers dispatch) |
| `+0x08` | READY | Read (nonzero = ready) |
| `+0x10` | RS1 | Write (latch before INST) |
| `+0x18` | RS2 | Write (latch before INST) |
| `+0x20` | BUSY | Read (nonzero = in-flight) |
| `+0x28` | OCCUPANCY | Read (queue depth) |

Issue sequence is strictly ordered: write RS1 → write RS2 → write INST. INST write is the dispatch trigger. Back-to-back MMIO stores go through the LSU; the instruction buffer provides elasticity but ordering must be respected. Minimum 3 cycles per command (3 stores), plus LSU latency.

Instruction word: `inst = 0x7B | (funct << 25)`.

Key funct codes: 0=CONFIG_EX, 7=FLUSH, 8=LOOP_WS (MX dispatch when preceded by funct 24), 9=LOOP_WS_CONFIG_BOUNDS, 10=LOOP_WS_CONFIG_ADDRS_AB, 12=LOOP_WS_CONFIG_STRIDES_AB, 24=LOOP_WS_CONFIG_SPAD_AB, 25=LOOP_WS_CONFIG_SPAD_C, 26=MXQUANT_CONFIG_MVOUT, 27=MX_LOAD_SCALES, 28=MX_READ_SMEM, 29=MX_LOAD_LUT.

`LOOP_WS` RS2 `skips` bits enable independent pipeline stage control: `skip_lda`, `skip_ldb`, `skip_ldd`, `skip_ex`, `skip_stc`. Load-only pass: `skip_ex|skip_stc`. Compute-only pass: `skip_lda|skip_ldb`. This is the mechanism for K-tile software pipelining.

Fence variants: `gemmini_fence()` spins on BUSY (full drain, blocking); `gemmini_fence_ready()` spins on READY (backpressure only, allows SIMT overlap); `gemmini_fence_waitcount(n)` spins until OCCUPANCY ≤ n (partial drain for pipelining).

### Warp Specialization Model

The 4 warps are partitioned: one Gemmini-manager warp issues all MMIO commands (`LOOP_WS_CONFIG_*`, `LOOP_WS`, `gemmini_fence`/`gemmini_fence_ready`) and polls READY/BUSY/OCCUPANCY; the remaining SIMT worker warps stage A/B operand tiles and e8m0 scales into SMEM via stores, and execute C move-out after fence. `mu_fence` and `mu_barrier` coordinate SMEM writes between warps before MMIO command issue. The primary pipeline overlap is: manager issues compute on tile K while workers stage tile K+1 into SMEM and load next scales. Scale double-buffering (even tiles at `SF_MEM_A/B` base, odd tiles at `SF_MEM_A/B + 0x800`, toggled via `rs1[60]`/`rs1[61]` in MXQUANT_CONFIG_MVOUT funct=26) enables this overlap without aliasing.

### Key Optimization Constraints

- **No SIMT matmul.** Accelerator is ~100× faster and rounds differently; bit-exact verdict requires hardware MX accumulation.
- **TILE_K must be a multiple of 32** to align with per-32-K-block e8m0 scale granularity.
- **SMEM is shared** between SIMT LSU and Gemmini DMA/compute; bank contention between staging writes and accelerator reads must be managed by layout and tiling.
- **SMEM bandwidth ceiling:** 64 bytes/cycle for SIMT; a single full-warp store saturates it. Double-buffering both SMEM regions concurrently will contend.
- **C move-out** (`copy_smem_to_gmem_simt`): all lanes issue contiguous 32-bit stores from SMEM after `gemmini_fence()`; bounded by L1 bandwidth (32 bytes/cycle, serialized across cores).
- **Register pressure:** `mxgemm_lib` is heavily inlined; `MX_NUM_WARPS=2` is the safe operating point. Do not increase warp count.
- **LUT reuse:** avoid redundant `MX_LOAD_LUT` calls across tiles when format is unchanged.
- **Scale staging is SIMT stores, not DMA** — `MX_LOAD_SCALES` (funct 27) is the DRAM path; in the Radiance kernel, SIMT workers write scales directly into `SF_MEM_A/B` via shared stores before the manager issues the dependent `LOOP_WS`.
- **FP6 LUT path** requires `uselut=1` in CONFIG_EX and LUT loaded before compute; FP8 E4M3 does not use LUT.
- **Accumulation order:** K ascending, `ex_accumulate=0` for first tile, `ex_accumulate=1` for all subsequent tiles.
- **`mu_exp`** must be used instead of `expf`/`exp`; sigmoid is `1.0f / (1.0f + mu_exp(-x))`.