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
