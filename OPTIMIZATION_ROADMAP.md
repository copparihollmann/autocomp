# Best-utilization roadmap per kernel (MX any-precision + SIMT), + realistic gap closure

## What I can close vs what I can't (be honest)
| Gap | Closeable by me? | How |
|---|---|---|
| ⚠️ RMSNorm/Softmax/fused PASS-unconfirmed | **Yes** | large-DRAIN re-run (write-drain, not a real fail) — running now |
| ⛔ l0d-assertion (SwiGLU/ResAdd/decode-GEMV) | **Yes, kernel-side** | **coalesce** access + reduce in-flight requests (reg8 cleared it by cutting loads). Demonstrating on ResAdd. The *DUT* fix (l0d landing pads) is the team's call; the *kernel* fix (coalescing) is mine |
| ❌ decode-attention RTL mismatch | **Investigate** | drain-invariant → real cyclotron-vs-RTL; root-cause FP-order/SFU or a kernel bug. May be closeable or a co-model/team item |
| 🚧 fp6/fp4/requant RTL precision | **No (team)** | co-model bit-exact; RTL variable-acc-precision / byte-order is the tapeout team's flagged WIP |
| ❌ fused MX flash-attn v4 | **Buildable** | team is writing it; can build from the test33 MX+SIMT fusion pattern if needed |

## The utilization levers (ranked by impact, from the RTL analysis)
1. **MX ≫ SIMT for matmul (~260× effective throughput).** Every matmul → MX. SIMT only for what can't
   go on the systolic array (norms, activations, RoPE, residual, softmax reductions, decode-GEMV).
2. **Lower MX precision = 2× MACs/cyc** (fp6/fp4 = 512 vs fp8 256). Use the lowest precision each matmul
   tolerates. (Blocked until the team's fp6/fp4 RTL precision lands.)
3. **Overlap SIMT with MX (today 1.0×; SIMT is 95% idle during MX compute).** Schedule the SIMT-side ops
   of the *adjacent* matmul concurrently with MX → ~2× at the fused-layer level.
4. **Amortize MX fixed overhead (up to 66%: config+DMA+move-out).** Bigger/fused tiles; DMA move-out
   (`SIMT_GMEM_MOVE_OUT=false`) reclaims the ~26% SIMT move-out; K=128→512 already showed 20%→34%.
5. **Coalesce + cut l0d pressure for SIMT streaming kernels** (also closes the l0d gap) → max memory BW.
6. **Fuse elementwise into matmul epilogues/prologues** → hide their latency behind MX + kill the DRAM
   round-trip.

## Per-kernel best strategy
**Projections (Q/K/V/O/Up/Gate/Down) — MX GEMM.** (a) Fuse Q+K+V into ONE GEMM (concat weights) and
Up+Gate into one → amortize config once. (b) fp6/fp4 where accuracy holds → 2×. (c) DMA move-out →
reclaim 26%. (d) Overlap input-RMSNorm + output-RoPE on SIMT during the matmul. Target: whole-kernel
34%→~60-70% + up to 2× from precision. *Compute-window is already ~92% — the win is all overhead+precision+overlap.*

**Prefill attention — fused MX flash-attn v4** (the team's kernel): Q·Kᵀ & P·V on MX, softmax on SIMT
*overlapping* the next MX tile, scores kept in SMEM (no DRAM round-trip). fp8 scores, fp6/fp8 P·V.

**Decode attention (GEMV) — memory-bound, keep on SIMT.** M=1 GEMV → the systolic array would sit
mostly idle, so MX is the wrong tool *unless you batch heads/steps to raise M* (then MX wins). Levers:
(a) coalesce the KV$ reads (clears l0d, maxes BW); (b) batch decode across heads/tokens → M>1 → route to
MX; (c) fix the RTL mismatch. Compute util is irrelevant here — optimize memory-BW utilization.

**RMSNorm / RoPE / ResAdd / SwiGLU — SIMT, memory/latency-bound.** They will *never* have high compute
util (they aren't compute). Best strategy = make their cost ~free: (a) **coalesce** (clears l0d, maxes
BW — ResAdd demo); (b) **fuse into the adjacent matmul**: RMSNorm into the proj input-staging, SwiGLU
into the FFN down-proj input, ResAdd into the proj output, RoPE into the QKV epilogue (test33 pattern) →
run *during* the MX matmul (overlap) and skip the DRAM round-trip.

## The meta-strategy (layer-level)
Schedule a transformer layer so **MX runs matmuls back-to-back while SIMT does the norms/activations/
RoPE/residual of the neighbouring matmul concurrently.** That converts today's 47%-idle, 1.0×-overlap,
per-op-launch layer into a pipeline where MX approaches its 92% and the SIMT work is hidden — so the
*layer* utilization approaches the tensor core's, not the SIMT lanes'. Precision knob on top: fp8 for
accuracy-sensitive matmuls (scores, some projections), fp6/fp4 for the large error-tolerant ones
(FFN) → another up-to-2×. Net realistic target: **layer throughput ~2× (overlap) × up-to-2× (precision)
× the overhead-amortization**, gated on (1) team fp6/fp4 RTL, (2) l0d coalescing, (3) a real fused/
overlapped layer scheduler (the biggest software lift).

## A1 UPDATE — l0d workaround findings (measured) + methodology fix

**Occupancy-drop avoids the l0d assert but is NOT a viable standalone fix.** Measured on ResAdd 128×512:
- NW=4 (baseline): trips the l0d assert, aborts at ~18K cyc.
- NW=2 and NW=1: **do NOT assert** (lower occupancy holds the l0d off) but perf **craters** — 1.8M+
  simulated cycles and still climbing (vs the assert at 18K). Single-lane (u_t35_gmvsm `if(tid!=0)return`)
  passes but is the extreme.
- Coalescing alone (sol28_coalesced) still tripped it.
=> For a **pure-streaming** SIMT kernel, the only l0d-safe configs starve concurrency so hard that
throughput collapses. Occupancy/ILP tuning is a dead-end for standalone streaming on this l0d.

**The real fix is FUSION, not occupancy.** The l0d assert is a *sustained GMEM-streaming* phenomenon.
Make the elementwise op **SMEM-resident** (read operands staged by an adjacent MX matmul, write back to
SMEM / into the next matmul) so there is no sustained GMEM stream → no l0d pressure. Proven by the
flash-attention kernel: its SIMT online-softmax reads S from SMEM (the QK result tile) and runs clean at
2 warps. So SwiGLU/ResAdd/RoPE should be **fused into the matmul epilogue/prologue** (Phase B.2/B.4),
NOT run as standalone streaming kernels. Standalone streaming (final residual, embedding) that genuinely
can't be fused is inherently l0d-limited on this silicon — flag as a HW constraint.

**Methodology fix (important):** the RTL sweep ran the sim with `+verbose`, producing 190–260 MB
stdout logs per run that made 4 concurrent sims catastrophically I/O-bound (~150 cyc/s). The trace-db
`.sqlite` (what we parse) is independent of `+verbose`. **Drop `+verbose` from the sim command** in
`sweep_util_trace.sh` for all campaign RTL runs → far faster, no giant logs. Also cap concurrency
(≤2 heavy Verilator sims) and keep DRAIN modest (verify PASS needs ~2–20K, not 300K).
