# Radiance kernel-design wiki (tapeout-330)

KDA-style single source of hardware constraints + optimization patterns for the Radiance SoC. Purpose:
(1) hand-optimization checklist, (2) inject into the autocomp agent prompt so it stops rediscovering the
same constraints. Every claim here is RTL-grounded (Verilator, tapeout-330) unless marked. Keep terse.

## 1. The machine
- **2 Muon SIMT cores** × 8 warps × 16 lanes. Scalar `fmadd.s` → 32 MAC/cyc/core = **64 flop/cyc/core**
  peak (SIMT is NOT a tensor engine — it's for norms/activations/RoPE/residual/softmax/decode-GEMV).
- **MX-Gemmini**: 16×16 systolic tensor core. **fp8 = 256 MAC/cyc; fp6/fp4 = 512 MAC/cyc (2×)** via
  packing (fp6/fp4 use 32×32 PE tiles, fp8 16×16). Every matmul → MX.
- **128 KB shared SMEM** (4 banks × 2048 rows × 16 B). Guard: `BANK_NUM*BANK_ROWS*DIM==128KiB` — do NOT
  build against the mxgemmini dev header (4096 rows = 256KB rocket/spike config → silent corruption).
- **Timing-accurate memory**: tiles @500MHz, uncore @200MHz (2.5× crossing); L2 InclusiveCache 512KB
  8-way `outerLatencyCycles=40`, 32-B blocks; DRAM miss ≈100 tile cyc. → **latency-hiding (SMEM reuse,
  fusion) is a REAL lever**, not a no-op.

## 2. Hard constraints (violate → wrong results or aborts)
- **l0d has NO landing pads** (`MuonTile.scala:242`, `makeLandingPads` defaults false). Sustained GMEM
  streaming trips `TLNBDCache "response must be ready!"` assert → mid-kernel abort. Occupancy-drop
  (NW=2/1) avoids it but perf craters (1.8M+ cyc). **The real fix is FUSION** — make the elementwise op
  SMEM-resident (operands staged by an adjacent MX matmul) so there is no sustained GMEM stream. Do NOT
  edit the DUT (tapeout fidelity). Standalone streaming (final residual, embedding) is inherently
  l0d-limited → flag as HW constraint.
- **256 physical-register Rename wall** (`globalOverSubscription`). Heavy inlining + 8 warp slots all
  running main() overflow → abort mid-kernel (truncated trace, 0 output stores). Rein in with
  `-mllvm -inline-threshold=256` per-kernel. A cycle count from an aborted run is MEANINGLESS — always
  check termination first (§5).
- **MX requant (fp8 output) is BROKEN on RTL** — writes ¼ correct / ¾ zero (wrong output scale in
  `MxRequantizer.scala`); bit-exact on co-model. It's the tapeout team's flagged datapath WIP, NOT
  kernel-fixable (both SIMT and DMA readout agree; version/ISA confirmed correct). See
  `REQUANT_INVESTIGATION.md`. **Workaround: keep GEMM output bf16 and quantize activations in SIMT.**
- **MX ISA is our own fork**: opcodes `CONFIG_SCALE_MEM=26`, `MVOUT_SPAD=23`; **NO** `MX_READ_SMEM=28`/
  `MX_LOAD_SCALES=27`/`MX_LOAD_LUT=29` (those are a newer chipyard-mx fork). Do NOT copy chipyard-mx or
  spike examples verbatim — they emit illegal opcodes here. RTL = gemmini@69a1c03 + radiance@f2755a3;
  kernel header = mxgemmini@62c4f85.

## 3. Correctness status of the kernel set (RTL, trace-verified)
- ✅ **All MX GEMMs bf16-output**: fp8, fp6 (0/16384 bit-exact), fp4 (0/4096 bit-exact). Workhorse path.
- ✅ **SIMT**: RMSNorm (0/32768), decode Q·Kᵀ/P·V (0-fail), GEMV-softmax, RoPE — all compute-correct.
- ❌ **Requant fp8-output** — the one real bug (§2). Bypass in SIMT.
- ⛔ **l0d-blocked at full shape**: standalone SwiGLU/ResAdd/decode-O-proj GEMV streaming (§2) → fuse.

## 4. Optimization levers (ranked by measured impact)
1. **MX ≫ SIMT for matmul** (~260× throughput). SIMT only for non-matmul.
2. **Precision: fp6/fp4 = 2× MACs/cyc, and PROVEN correct on RTL** (was mislabeled "team-blocked" — it's
   not). Use lowest precision each matmul tolerates. FFN/large error-tolerant → fp4; scores/sensitive → fp8.
3. **Overlap SIMT with MX** (today 1.0×, serialized; SIMT ~95% idle during MX compute). Warp-specialize:
   manager warp(s) drive `mxgemm<C>` while other warps run the *adjacent* tile's SIMT epilogue with NO
   barrier between them (pattern: `gemm_mxgemmini/mxgemm.simt_contention.cpp`). Biggest win is LAYER-level
   (~2×); single-op overlap is modest (epilogue ≪ matmul). Cyclotron is BLIND to this — measure on RTL.
4. **Amortize MX fixed overhead** (config + DMA + move-out ≈ 66% at small K). Bigger K (K=128→512 banked
   1.68×), fuse Q+K+V into one GEMM (concat weights), larger tiles.
5. **Fuse elementwise into matmul epilogue/prologue** (no DRAM round-trip + clears l0d): RMSNorm→proj
   input-stage, RoPE→QKV epilogue (`autocomp_fused33` pattern, verified), SwiGLU→FFN down-proj input,
   ResAdd→proj output.
6. **Coalesce** lane-consecutive GMEM access for the SIMT kernels that must stream.

## 5. Methodology (non-negotiable)
- **Report only REAL numbers = RTL (Verilator/VCS).** Cyclotron is a fast functional pre-check only; it
  mis-ranks memory/SMEM/overlap — never quote its cycles as a result.
- **Verify via offline trace-db reconstruction, NOT in-kernel tohost read-back** (SIMT-store visibility
  artifact gives false fails). Reconstruct stores from the `inst` table (store-PC-filtered) or read final
  DRAM from the `dmem` LOAD log. Scripts: `scratchpad/simt_out_verify.py`, `mx_verify2.py`, `rq_dma_verify.py`.
- **Check TERMINATION before trusting cycles**: grep sim.log for `Aborting|Assertion failed|$stop` and
  confirm the kernel completed. Aborted (reg-wall / l0d) runs give truncated, meaningless cycle counts.
- **Metric = total kernel-region cycles** (`hw_util_phased.py` "kernel span"), never IPC (gameable).
- **Sim hygiene**: NO `+verbose` (190-260MB logs, I/O-bound); ≤2 concurrent Verilator sims; one run
  ≈ 20-25 min; DRAIN modest (2-20K). Trace-db is `+verbose`-independent. Beware `pgrep -f` matching your
  own monitor script (self-reference) and multiple sims racing one trace-db file.

## 6. Reuse map (build on these, don't reinvent)
- l0d in-flight knobs: `kernels/{saxpy,vecadd}` (`impl<ILP>`, `__mu_num_warps`, `unroll(disable)`).
- SIMT tiling: `kernels/gemm_simt` (BM/BN/BK/TM/TN + double-buffer), `kernels/sgemm_muon` (4×4 microtile).
- Fusion/overlap: `kernels/autocomp_fused33` (MX→SMEM→SIMT), `kernels/gemm_mxgemmini/mxgemm.simt_contention.cpp`
  (warp-spec), `kernels/flash_attention_virgo` (fused online-softmax).
- MX precision: `kernels/gemm_mxgemmini/mxgemm_lib.hpp` `GemmConfig{.DATATYPE=FP8|FP6|FP4}`; data headers
  `mxgemm.data.fp{8,6,4}.mMnNkK.h`.
- Reductions/norms: `kernels/{softmax,rmsnorm,gemv}`. Fences: `lib/include/mu_intrinsics.h`.

## 7. Open perf work (this campaign)
- Precision A/B on RTL (fp8 vs fp4 vs fp6 @ matched shape) — quantify lever #2. [in progress]
- Warp-specialized overlap testbed (lever #3) — needs multi-tile to show. [planned]
- SIMT bf16→fp8 activation-quantize (requant replacement + fusion) — proves model correctness. [planned]
