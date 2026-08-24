# Whole-network TinyLlama headroom — the DRAM weight-BW floor bounds everything

## The master reframe (quantified)
Per-layer weights (TinyLlama-1.1B, dim 2048, 32Q/4KV×64, FFN 5632): Q 4.19M + K 0.52 + V 0.52 + O 4.19 +
gate 11.53 + up 11.53 + down 11.53 = **~44M params/layer**. fp4 = **22 MB/layer**, fp8 = 44 MB.
L2 = 512 KB, SMEM = 128 KB. **22 MB ≫ 512 KB → no weight ever survives in L2/SMEM across ops; each weight
matrix is read from DRAM exactly ONCE per layer.**

- **DRAM BW ≈ 4–8 B/cyc** (docs disagree: HW_UTILIZATION 4, MUON_RESULTS 8 — PIN THIS, it sets the ridge/floor).
- **Weight-move floor/layer (fp4): 22 MB ÷ (4–8 B/cyc) = 2.75M–5.5M cyc/layer — a HARD floor no on-chip trick beats.**
- **Roofline ridge: M ≈ 30–60** (compute = weight-move). Below → weight-BW-bound; above → compute-bound.

**This partitions the problem:**
- **DECODE (M=1, batched M≤32): weight-BW-bound, AT the floor.** Matches batched-decode data (M=32 = 39% mesh
  peak = the ridge). No on-chip trick escapes the 22 MB read; only levers = fewer weight bytes (fp4 done) +
  raise M past the ridge (done) + hide latency. **Decode is essentially optimized to its floor.**
- **PREFILL (M=256–2048): compute-bound, deep past the ridge.** Here weight-prefetch/overlap CAN hide the DRAM
  stream, and amortization/precision/weight-reuse pay off. **This is where the remaining room is.**

## Ranked remaining levers (for a full fused PREFILL layer)
1. **Larger-M weight-stationary reuse (1.37×→~2×)** — HIGHEST. Attacks the first-order weight-move. Built A/B
   pair `autocomp_gemm_wstationary/` RTL-gate-pending (cyclotron-blind → RTL is arbiter).
2. **Fix layer-overlap on RTL (1.06×→~1.3×)** — HIGH ceiling, hardest (open regression). Root cause = residual
   GMEM-stream + SMEM-bank contention cyclotron can't see. Being root-caused (`autocomp_overlap_layer_*`).
3. **Cross-layer weight prefetch** — HIGH for prefill (hide the 22 MB stream behind compute), ZERO for decode.
   Idiom: `mxgemm_prefetch_tile` — extend to stream layer N+1's first weight tile during layer N's tail.
4. **Persistent single-launch-per-layer** — MEDIUM. Fused FFN today = 7 mu_schedule+barrier+fence+drain launches
   (~100k+ cyc pure overhead); collapse to 1 → removes barrier/fence serialization + enables 2/3/6.
5. **RMSNorm-prologue into the FFN block** — MEDIUM, idiom proven (`autocomp_fused_rmsnorm_qkv`); removes the Xn
   GMEM round-trip. Not yet ported into `ffn_block_fp4`.
6. **Activation double-buffer / keep Xn resident across gate+up** — MEDIUM. gate+up are the ONLY consecutive ops
   sharing an operand (both consume Xn_fp4, 64KB@M=64, fits SMEM). QKV/O/down share nothing.
7. **Cross-layer residual + next-norm fusion** — MEDIUM-LOW (activation traffic, second-order).
8. **Unblock HID=2048** — REQUIRED prerequisite (K-loop DMA/barrier backpressure at 32 K-tiles; same class as the
   RMSNorm-prologue write-drain fix). Until fixed, the fast fused block only exists at toy HID=512.

## Downgraded (NOT throughput levers)
- **L2 weight residency across a layer**: physically impossible (22 MB ≫ 512 KB). Only prefetch helps, compute-bound only.
- **fp4-weights + fp8-activations mix**: does NOT cut weight bytes (already fp4), raises activation bytes → an
  ACCURACY knob (fp6/fp8 for score-sensitive ops), not perf.
- **fp4 beyond 1.74×**: fixed multiplier (grows slightly with K), not remaining headroom.

## Key nuance the ledger missed
The fused-block intermediates (Xn_fp4 64KB, h_fp4 180KB @M=64) FIT in L2. Their "GMEM round-trip" is an L2
round-trip (cheap) — but cyclotron is L2-blind so it looks expensive. **RTL may already recover this** (same
blind-spot that hid the 1.37× weight-reuse). Measure the intermediate round-trip on RTL.

## Bottom line
Biggest UNREALIZED wins: (1) larger-M weight-reuse→2× [RTL-gating], (2) fix layer-overlap on RTL→~1.3× [open],
(3) cross-layer weight prefetch [prefill only]. Rest is real but second-order (touches activation traffic, not
the 22 MB weight stream). **Decode is at its DRAM floor (done); prefill has ~2× of realistic room left, all
requiring RTL to measure (cyclotron is blind to L2 residency, DRAM latency, bank contention, overlap).**
Every remaining lever except RMSNorm-round-trip-removal needs RTL as the arbiter.
