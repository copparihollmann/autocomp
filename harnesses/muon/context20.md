MX-Gemmini FP8 matmul C[64,64] (bf16) = A[64,64] @ B[64,64] (fp8 e4m3), on the WHOLE
Radiance config: Muon SIMT cores + the MX matrix accelerator, not SIMT alone.

The warps do NOT do the arithmetic. They drive the accelerator over its MMIO command block
and move data. `mxgemm_lib.hpp` is already included and provides the driver; the baseline
just calls `mxgemm<GemmConfig>(...)`. One threadblock: 2 cores x MX_NUM_WARPS(2) warps x 16
lanes. The accelerator's scratchpad IS cluster SMEM, so operands, the C accumulator, the
e8m0 scale SRAMs (0x88000 / 0x8A000) and the MMIO block (0x84000) all share one address
space.

What the driver does per output tile:
  configure (config_ex: fp8 in, full/bf16 out)
  stage per-32-K-block e8m0 scale factors into SMEM with SIMT stores
  loop_ws #1 with skip_ex|skip_stc  -> accelerator DMAs A/B tiles DRAM->scratchpad
  loop_ws #2 with skip_lda|skip_ldb -> systolic compute, bf16-accumulate into C in SMEM
  gemmini_fence()                    -> spin on the BUSY reg until the array drains
  SIMT move-out of C from SMEM to GMEM

Optimization levers (this is an orchestration problem, not an inner-loop problem):
 - Overlap: the accelerator is busy for thousands of cycles. Use `gemmini_fence_ready()`
   (accepts the next command) instead of `gemmini_fence()` (waits for completion) so SIMT
   work -- scale staging, C move-out of the previous tile, next-tile DMA -- runs underneath.
 - Warp specialization: one warp issues accelerator commands while the others stage scales
   and move C out (see `mxgemm.simt_contention.cpp` / `mxgemm.flash_contention.cpp` in
   radiance-kernels for the pattern). Beware SMEM bank contention between the two.
 - Tiling: TILE_M/TILE_N/TILE_K in GemmConfig. TILE_K must be a multiple of 32 (the MX
   block-scaling group). Larger K per loop_ws amortizes command overhead; smaller tiles
   allow double-buffering A/B against the compute of the previous tile.
 - Move-out: `copy_smem_to_gmem_simt` is vectorized (32-bit stores); make sure every lane
   contributes and addresses are contiguous per lane.
 - Do NOT hand-roll the matmul in SIMT. The accelerator is ~100x the SIMT FP throughput;
   an "optimization" that computes in SIMT will be both wrong (different rounding) and slow.

Constraints:
 - Register file: 256 physical regs across 8 warp slots. mxgemm_lib is heavily inlined; if
   you add per-thread state you can trip `globalOverSubscription` (an RTL assert).
   MX_NUM_WARPS is 2 for this reason.
 - Do not rename MX_NUM_WARPS to NUM_WARPS: VX_config.h already defines NUM_WARPS as 8.

Verification is BIT-EXACT against a golden that encodes the hardware's exact accumulation
semantics (16-deep systolic accumulator with a per-column precision schedule, one e8m0
scaling per 32-element K group), validated against real spike libgemmini. There is no FP
tolerance: reassociating the K loop, or accumulating in a different order/precision, changes
the result and will be rejected. Keep k ascending and let the accelerator do the math.
