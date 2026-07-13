# Whole-Radiance: driving MX-Gemmini from Muon SIMT over MMIO

This is the bridge that the pure-Muon and pure-Gemmini docs both omit: on the **whole
Radiance config**, the Muon SIMT cores do NOT do the matrix arithmetic. They *orchestrate*
an on-cluster **MX-Gemmini** matrix accelerator by writing its command registers over MMIO,
staging operands/scales into shared memory, and moving results out. The accelerator's
scratchpad **IS** the cluster shared memory, so operands, the C accumulator, the scale
SRAMs and the MMIO command block all share one address space.

> IMPORTANT: This OVERRIDES any rule that says "mxgemmini/tensor instructions are not
> available to SIMT" or "do not write to the Gemmini MMIO block." On the combined config
> that MMIO path is exactly how kernels reach the matrix unit — it is the whole point.
> The accelerator is ~100× the SIMT FP throughput; hand-rolling a matmul in SIMT is both
> wrong (different rounding) and far slower.

## Config

`RadianceSingleClusterConfig` = `WithRadianceMxGemmini(InCluster(0), dim=16)` +
`WithMuonCores(2)`. One threadblock = 2 cores × 2 warps (`MX_NUM_WARPS`, see registers) ×
16 lanes. DIM = 16 (systolic array is 16×16). SMEM = BANK_NUM(4) × BANK_ROWS(2048) rows ×
16 B/row = 128 KiB. A scratchpad *row address* r maps to SMEM **byte** `r * DIM`.

## MMIO command protocol

`GEMMINI_CTRL = 0x00084000` in the GPU-local **shared** space (see `mxgemmini_mmio.h`).
To issue a RoCC command, a SIMT thread does three shared stores:

```
store rs1 -> GEMMINI_CTRL + 0x10        (RS1)
store rs2 -> GEMMINI_CTRL + 0x18        (RS2)
store inst-> GEMMINI_CTRL + 0x00        (INST word triggers the command)
   inst = 0x7B | (funct << 25)
```

Only the INST-word store fires the command; RS1/RS2 just latch. Status registers (read to
poll / fence):

```
READY     @ +0x08   (accepts next command)
BUSY      @ +0x20   (array still draining)
OCCUPANCY @ +0x28   (queue depth)
```

`gemmini_fence()` spins on BUSY until the array drains (blocking). `gemmini_fence_ready()`
waits only until the unit can accept the *next* command — use it to overlap SIMT work
(scale staging, previous tile's C move-out, next-tile DMA) underneath accelerator compute.

## RoCC funct table (the ones the MX datapath uses)

| funct | name | purpose |
|------:|------|---------|
| 0  | CONFIG_EX          | set A/B/C formats (fp8/fp6/fp4/full), uselut |
| 7  | FLUSH              | reset accelerator state between matmuls (clears the C accumulator) |
| 8  | LOOP_WS            | run the loop FSM (DMA and/or compute per the `skips` bits) |
| 9  | LOOP_WS_CONFIG_BOUNDS | PE-tile loop bounds I/J/K |
| 10 | LOOP_WS_CONFIG_ADDRS_AB | A/B DRAM base addresses |
| 12 | LOOP_WS_CONFIG_STRIDES_AB | A/B DRAM row strides |
| 24 | LOOP_WS_CONFIG_SPAD_AB | A/B scratchpad (SMEM) tile addresses |
| 25 | LOOP_WS_CONFIG_SPAD_C  | C scratchpad address (rs2[63:32]) |
| 26 | MXQUANT_CONFIG_MVOUT | output requant + **scale double-buffer selects** |
| 27 | MX_LOAD_SCALES     | (DRAM path only; the Radiance kernel stages scales via SIMT stores instead) |
| 28 | MX_READ_SMEM       | drain the bf16 C accumulator from SMEM to DRAM |
| 29 | MX_LOAD_LUT        | FP6 dequant LUTs |

`skips` field of LOOP_WS rs2 (bits, `loop_matmul_skips`): lda, ldb, ldd, ex, stc. The
Radiance kernel issues LOOP_WS **twice per K-tile**: first **load-only** (`skip_ex|skip_stc`)
so the accelerator DMAs A/B tiles DRAM→scratchpad, then **compute-only**
(`skip_lda|skip_ldb`) to run the systolic array and bf16-accumulate C in SMEM. The
`ex_accumulate` bit (set for every K-tile after the first) makes compute ADD into the
existing C rather than overwrite — this is how K is reduced across multiple K-tiles.

## Numerics (microscaling MX)

fp8 e4m3 (16×16 tiles, 1 code/byte); fp4 e2m1 & fp6 e3m2 (32×32 tiles, nibble-packed — A
along M, B along N; fp6 nibbles index a 16-entry LUT of e3m2 codes). Accumulation is a
16-deep systolic array with a per-column precision schedule (acc_e/acc_m), one **e8m0 block
scale per 32-element K group**. Reassociating the K loop or changing accumulation order/
precision changes the bit-exact result — keep k ascending and let the accelerator do the math.

## Scale-factor staging + double-buffering (subtle, easy to get wrong)

Per-32-K-block e8m0 scales are staged into SMEM by SIMT stores (NOT DMA'd):
`SF_MEM_B = 0x88000`, `SF_MEM_A = 0x8A000`. For software-pipelined K-tiles the kernel
**double-buffers** scales: odd K-tiles are staged at `SF_MEM_* + 0x800`
(`GEMMINI_SF_MEM_BUFFER_OFFSET`), and the accelerator is told which buffer to read via the
`scale_act_sel`/`scale_wgt_sel` bits (rs1[60]/rs1[61]) of `gemmini_mxquant_config_mvout`.
FP6 LUTs likewise stage into SMEM at LUT0=0x84080 (weights/B), LUT1=0x84380 (activations/A),
LUT2=0x84680 (output).

## The driver: `mxgemm_lib.hpp`

`mxgemm<GemmConfig>(...)` computes a single TILE_M×TILE_N output tile and loops K internally
(software-pipelined: DMA next tile / compute this tile / stage next scales / fence). Per
output tile it: `configure_mxgemmini` (CONFIG_EX + bounds) → stage e8m0 scales into SMEM →
LOOP_WS load-only (DMA A/B) → LOOP_WS compute (systolic, accumulate C in SMEM at
`SPAD_DEST=256` i.e. SMEM byte 4096) → `gemmini_fence` → SIMT move-out of C (SMEM→GMEM,
`copy_smem_to_gmem_simt`, vectorized 32-bit stores). `GemmConfig` levers: TILE_M/TILE_N/
TILE_K (TILE_K multiple of 32), DATATYPE (FP8/FP6/FP4), QUANT_OUTPUT.

## Optimization levers (this is an ORCHESTRATION problem, not an inner-loop problem)

- **Overlap:** use `gemmini_fence_ready()` instead of `gemmini_fence()` so SIMT work
  (scale staging, previous-tile C move-out, next-tile DMA) runs under the ~thousands of
  cycles the accelerator is busy.
- **Warp specialization:** one warp issues accelerator commands while the others stage
  scales and move C out (see `example_warpspec_simt_contention.cpp`). Beware SMEM bank
  contention between them.
- **Tiling:** larger K per LOOP_WS amortizes command overhead; smaller tiles allow
  double-buffering A/B against the previous tile's compute. Larger output tiles cut the
  fixed per-tile overhead but must fit the C accumulator without overlapping operands in SMEM.
- **Move-out:** make every lane contribute contiguous 32-bit stores in `copy_smem_to_gmem_simt`.

## Register constraint

`mxgemm_lib` is heavily inlined. `NUM_WARPS` is already 8 in `VX_config.h`; launching 8
warps' worth of live registers blows the 256-entry physical register file (RTL Rename
assert / cyclotron `globalOverSubscription`). Use `MX_NUM_WARPS = 2` — do NOT reuse the
name `NUM_WARPS`.

## Fast evaluation

The combined kernel runs in cyclotron (fast) with `CYCLOTRON_MXGEMMINI=1`, which activates
the MX co-model that reads/writes the same SMEM the Muon LSU uses (so accelerator results
become visible to subsequent SIMT warps). Correctness is bit-exact against a golden that
encodes the hardware accumulation semantics (validated vs real spike libgemmini). RTL
`RadianceSingleClusterConfig` is the final gate. Verdict is BIT-EXACT (no FP tolerance):
the idealized numpy/torch model disagrees with the hardware on ~90% of elements.
