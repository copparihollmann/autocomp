# Radiance LLM-inference kernels — detailed measurement notes (internal)

Companion to `KERNELS.md`. This is the reference for questions from the RTL / tapeout authors: how each
number was obtained, the debugging behind the kernels that were hard to bring up, and the results we
explored but did not ship. Not part of the PR description.

---

## 1. Measurement setup

- **Simulator:** Verilator, `RadianceSingleClusterConfig` (1 MX-Gemmini 16×16 mesh + 2 Muon cores).
  Run with `+max-cycles=20000000 +trace-db=<x>.sqlite +loadmem=kernel.soc.elf`, `ulimit -s unlimited`.
- **Correctness oracle:** the functional co-model (cyclotron), which halts with `tohost=0` on a bit-exact
  match against the hardware golden. Kernels are validated there before an RTL run. RTL `tohost` read-back
  is not used as the correctness signal (the on-device verify path has its own artifacts); the RTL run is
  for cycles.
- **Peaks (from RTL source):**
  - MX MAC/cyc = `numOutputs` × 256 PEs. `MxParameters.scala`: mode8/fp8 numOutputs = 1 → 256;
    mode0/fp4 and mode4/fp6 numOutputs = 4 → 1024. MX_peak_FLOP = 2× = 512 (fp8) / 2048 (fp4·fp6).
  - Muon FP pipe: `FPPipe.scala` numFP32Lanes = 8, numFP16ALULanes = 16, bf16-native. Peak = 16 × 2 cores
    × 2 FLOP/FMA = 64 FLOP/cyc bf16 (fp32 half = 32). Issue peak = 2 (one slot per core).
  - Config (`RadianceConfigs.scala`): numWarps 8, numLanes 16, numCores 2; 2-cluster config is
    `RadianceTapeoutSimConfig`.

## 2. How the cycle window is extracted

Two extraction paths, matched to how a kernel drains its output. Getting this wrong is the single biggest
source of mis-reads, so both are pinned to the RTL trace.

- **`max_cycle`** — `max(cycle)` over the trace `inst` table. Correct for **mesh-output kernels** (GEMM,
  fused): the Muon manager warp finishes and drains early (~20–40k) while the mesh keeps computing, so the
  full kernel = last mesh instruction.
- **`net_kernel_cycles`** — drain hand-off, in `scripts/muon/rtl_kernel_cycles.py`. Correct for
  **SIMT-output kernels** (weight-stationary, decode), where the Muon does the final store. The terminal
  drain is the highest-hit PC among the top-8 PCs (hit-count ≥ 50% of the max); its first cycle is the
  kernel end. This deliberately excludes the internal barrier spin during the matmul (an earlier extractor
  latched onto that barrier and under-reported decode by ~4×).
- Trace `cycle` is the Muon **core** clock = 2× the tile clock.

**SIMT sub-metrics** (`scripts/muon/lane_eff.py`, over the trace `inst` table): lane-eff =
mean(popcount(lane_mask & 0xFFFF))/16; IPC = instrs ÷ cycle-span (both cores); issue-util = IPC/2.
`scripts/muon/hw_util_phased.py` computes the same IPC plus the phased MX/SIMT/whole split when given the
per-kernel MAC/FLOP counts. (Its `PEAK_MX` constant was corrected fp4/fp6 512 → 1024 to match the RTL.)

## 3. Per-section notes

### GEMM ladder (§A)
Whole-kernel MX util climbs 32% → 95% as K goes 512 → 5632: the per-tile scale-load/config is a fixed
overhead amortized by deeper K (the mesh runs ~95–100% inside the K-loop). There is no native input-scale
DMA, so this overhead is a hardware floor. 128×128 is the hardware-max output tile — the Gemmini
accumulator (256 rows = one 128×128 fp8 tile) and the SPAD bf16 C-staging both cap at 128×128; 256² is
rejected at compile (`C_FITS_IN_SPAD`). So deeper K is the only amortization lever, and 128×128 fullout
requires TILE_K ≤ 128.

Low precision (fp6/fp4) reads low on util (24–39%) because the mesh is 4× denser (2×2 packing → 1024
MAC/cyc) but the single Muon manager warp cannot feed operands that fast — at 4× density the mesh drains
each K-tile faster than the manager feeds it, so it idles. This is **orchestration/operand-feed-bound**,
not memory-bound (arithmetic intensity is high) and not scale-load-bound (the per-tile scale-load is
already double-buffered behind compute). The reachable lever is cutting Muon orchestration — chiefly the
DMA move-out (§5). fp6 and fp4 share the same peak, so use fp6 wherever accuracy matters at no throughput
cost.

### Weight-stationary (§B)
Read-once vs re-stream = 1.33× at K=2048 (216,514 vs 288,355), cutting DRAM weight reads 4× → 1×. The fp4
down-proj (K=5632, #9) is measured at 293,134, but its fp8 read-once/re-stream counterparts **hang on RTL**
(~500k, no `$finish`), so no fp4-vs-fp8 multiplier is claimed. Ceiling: the SPAD forces TN ≤ 64 at TM=256,
plus the read-once weight-move floor.

### Attention (§C)
8-head GQA + causal at d=64, fp8, flash online-softmax. Retires on RTL at 495,661 (all 8 warps, `$finish`,
188,577 Muon instructions). It is **SFU-exp-latency-bound**: the cores stall ~81% of cycles on `exp`, so
issue-util is 19.1% at lane-eff 92.6% (lanes well-packed, issue starved). The d=64 matmuls under-fill the
128×128 mesh, so MX util is low and not the binding metric.

**Bring-up.** The straightforward variant deadlocks: at occupancy that fits enough warps for the
`mu_barrier`, the ~57 registers/warp overflow the 256-entry rename pool, and the barrier waits on warps
that never load. The shipped kernel is the **thread-per-row softmax variant at occupancy 2 (4 warps)** —
its lighter register footprint fits 4 warps, so the barriers complete and it retires. Occupancy 3/4 (6/8
warps) overflow the register file and deadlock (both measured to hang). So occupancy 2 is the hardware
ceiling; beating 19.1% needs more warps (blocked by the register file) or a vector-exp unit — silicon,
not a kernel change.

**Accuracy** (offline golden vs fp32, `fp6_attn_relerr.py`, now also reporting cosine): fp8 cosine
0.99859 / rel-err 5.33%; fp6 0.99448 / 10.55%; fp4 0.97074 / 24.0%.

### Batched decode (§D)
Batch M rows through the Gemmini DMA (bypasses the l0d). Reported as full-kernel latency (drain hand-off,
includes the SIMT C move-out) → fp4-M128 is 3.65×/token vs fp8-M32. In a real autoregressive loop the C
move-out overlaps the next token's compute, so amortized throughput approaches the GEMM-end window
(~4.19×); the ranking (fp4-M128 fastest) holds either way. M=128 is the SPAD ceiling. Requires the
non-square-tile fix (M ≠ N).

### Fused epilogues (§E)
SIMT element-wise fused into the matmul, SMEM-resident. Measured 67,211 / 106,147 / 35,645. Matmul dims
(from the committed data headers): rmsnorm_qkv_fused fp8 64×64×256, rope_qkv_fused fp4 64×64×2048, requant
fp8 128×128×128 → the MX-util column. They are SIMT-bound (the element-wise stage dominates; low IPC
0.13–0.14 shows they are load/SFU-latency-bound, not issue-bound). They retire individually; chaining
several into one megakernel deadlocks on RTL (SFU fence-stall under l0d back-pressure), so fusion is
per-op, not whole-layer. No unfused single-kernel baseline is committed, so no fusion multiplier is
claimed. The old draft's "1.25× vs unfused" and a "201,686" requant figure were from workloads not in the
committed set and are dropped.

### Coverage ops (§F)
All 9 run to completion on RTL and are bit-exact; precision is carried by the GEMM paths, so these are
bandwidth-bound element-wise ops where a near-full lane-eff is the ceiling. They are run on plain Verilator
(no lockstep difftest); the reported RTL cycle is the compute window (`net_kernel_cycles`), which excludes
the in-kernel serial verify loop. Measured compute-window cycles / lane-eff / IPC: qkv_bias 15,232 / 93.5%
/ 0.06, geglu 7,656 / 87.0% / 0.11, logit_softcap 28,195 / 92.8% / 0.04, rmsnorm_gemma 17,315 / 91.7% /
0.06, embed_scale 17,506 / 91.4% / 0.06, gemma_4norm 20,687 / 87.6% / 0.06, layernorm 20,934 / 92.0% /
0.05, patch_embed 43,670 / 99.7% / 0.56, bias_add 10,525 / 88.8% / 0.08. (The earlier cyclotron functional
counts of 2.6–4.7 M were far off and are not used.) Two fixes are contributions: `geglu` (a flat
2-load/1-store streamer stalled the l0d, cleared by lowering `KERNEL_OCCUPANCY` 4 → 2) and `patch_embed`
(weight thrash, SMEM-staged with a lane-to-patch remap, ~16× less weight traffic).

### Precision fidelity vs fp32
Cosine similarity / relative error of the quantized result against an fp32 reference (`gemm_fidelity.py`
for the GEMM path, `fp6_attn_relerr.py` for attention). GEMM 128×128×2048: fp8 0.99917 / 4.07%, fp6
0.99706 / 7.67%, fp4 0.98708 / 16.07%. Attention 8-head GQA d=64: fp8 0.99859 / 5.33%, fp6 0.99448 /
10.55%, fp4 0.97074 / 24.02%. The matmul kernels are bit-exact against their same-precision golden, so this
table is the separate measure of what the datatype itself costs.

## 4. Held / not-retiring items — detail

**`flash_attention_mx_gemma`** retires on RTL after two D=256 SMEM-layout fixes inherited from the D=128
template: (1) PVOUT overflowed the 128 KB SMEM → TileLink PutFull stop; (2) the 32 KB PV C-output collided
with the mesh's V operand at the top of SMEM, corrupting O. Fix: relocate PVOUT to the never-used odd SPAD
halves — kernel-only, shared `mxgemm_core.hpp` untouched. The mesh-written O is **on-device-unverifiable**
by two independent tooling limits: the co-model emits NaN for mesh-O (confirmed on the pristine upstream
attention too), and the trace-db captures only SIMT dmem stores while O is written by the MX move-out DMA.
What is proven: the soft-cap-fold + sliding-window softmax logic verifies `tohost=0` on the mesh-free twin;
offline golden vs fp32 error is 11.35%.

**`flash_attention_mx_fp6`** had two RTL deadlocks. #1 (fixed): the fp6-LUT min-select / fixed-point
ternaries lowered to divergent branches (104 in the finder); the warp failed to reconverge and hung
silently at cycle 490,585. Fixed with an opaque inline-asm mask-select (divergent branches 104 → 0). #2
(remains): the quantizer's final tile stalls on a single 16-byte coalesced global store stuck in the l0d
(the silent streaming variant, below the `TLNBDCache` assert threshold), so it never reaches the mesh.
Proper fix: keep the quantized operands SMEM-resident — a re-architecture. Cyclotron-correct; cosine
0.99448, rel-err 10.55%.

## 5. Explored but not shipped

| item | result | why not shipped | path to shipping |
|---|---|---|---|
| DMA move-out (`SIMT_GMEM_MOVE_OUT`) | RTL fp4 1.20× / fp8 1.08× (prior run) | single-output-tile only; not re-measured this pass | multi-tile version, re-measure vs next-tile mvin |
| MX‖SIMT overlap | ~1.28×, fragile (deadlocks when pushed) | ~17.5% both-active ceiling | raise both-active — partly silicon |
| thread-per-row softmax | now the shipped attention (§C) | — | promoted |
| single-stream GEMV | 3 orders behind batched MX | l0d-aliasing workaround | needs l0d landing-pads — silicon |
| single-tile LM-head | only cross-validates the fp4 GEMM | — | fold into `gemm_mxgemmini` as a multi-output-tile shape |
| fused FFN / whole-layer megakernel | does not retire (SFU + l0d deadlock) | — | needs l0d landing-pads + SFU fence-skid — silicon |

**Dual-cluster** (`RadianceTapeoutSimConfig`, 2 clusters): a data-parallel GEMM was written (output M-tiles
split across clusters, single-cluster-validated) but the 2-cluster Verilator host-boot is too slow to
measure in-session. Analytical projection ≈ 1.3–1.5× (not the naive 2×): the two mesh compute-windows could
~2× if DRAM feeds both, but ~half of the single-cluster runtime is SIMT C move-out contending on the shared
L2→DRAM path, which does not scale — reinforcing DMA move-out as the first lever to pull.

**Separate exploration branch** (`chipyard-mx`, `RadianceGemminiOnlyConfig`, scored on spike): a different
config and metric, not comparable to these tables. Salvageable for a separate spike-model PR: RTL-legality
guards, the faithful-cycle model, an `mx_iexp` shift-UB fix, and non-flash-attention / MX-conv coverage.
Its large spike speedups (3–5×) are inflated or RTL-illegal and are not reported.

## 6. Migration status

Staged locally on `feat/radiance-autocomp` as one commit, not pushed. Net diff: 23 kernel dirs + the
`lib/mxgemmini` submodule bump (`6fc8ec7 → 62c4f85`) — a required build fix (`gemmini.h` `abs()` on
`uint64_t acc_t` is ambiguous once real libc headers are on the include path; the old pointer fails to
compile the MX kernels, verified). No scratch dirs or build artifacts are committed. `KERNELS.md` (the PR
description) and this file are drafts kept in `autocomp/`, not committed.
