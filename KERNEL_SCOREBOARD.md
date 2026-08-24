# TinyLlama Kernel Scoreboard — RTL-final, correctness-verified kernels

**One question this answers:** have we finished iterating on the TinyLlama kernels, and how do
they compare to the radiance-kernels baselines? Every row is grouped by op class.

## How to read this (see METHODOLOGY.md)
- **RTL-ONLY.** Every latency/util/speedup here is Verilator (tapeout-330). **Cyclotron numbers are
  NOT results** (§0) — kernels that only have a cyclotron number are listed in a separate "NO RTL
  NUMBER — not final" block, never mixed into the RTL rows.
- **Two RTL cycle proxies exist and are NOT interchangeable** (§2): `net_kernel_cycles` (drain-keyed,
  for GEMMs built with a DRAIN spin) and `trace_max`/dmem-span (full span, for fused blocks built
  DRAIN=0). Ratios are only trustworthy **within a matched family**; cross-family absolute cycles are not.
- **Completeness gate** (§3): a cycle count counts only if `dmem store-count == baseline's` (a truncated
  run yields a plausible-but-wrong span — this trap already burned the "persist 1.264×" claim, corrected below).
- **Utilization** = achieved/peak (§5): *MX whole-kernel util* (M·N·K/cyc /peak, includes fixed overhead —
  the amortization metric), *MX mesh-in-loop util* (~95-100% at real K), *SIMT util*, *both-active%* (§6).
  Per row I give the one that applies. Peaks (§1): MX 256 MAC/cyc fp8, 512 fp4/fp6; SIMT 32/64 flop/cyc.
- **Correctness** (§8): GEMM/FFN = cyclotron `verify_body` tohost=0 (bit-exact vs the MX golden the co-model
  generates); MX-mesh attention FP = Python golden **rel-err** (on-device verify is platform-blocked). The
  GEMM "verify-fails" flagged as *read-back artifact* are the stale-golden/SIMT-read-back class (cycles valid,
  control flow data-independent) — not compute bugs. The ONE genuine correctness bug is the **requantizer**
  (RTL datapath diverges from co-model — team WIP, not kernel-fixable) → all deployed decode/GEMM paths use
  `QUANT_OUTPUT=false` (bf16 out) to dodge it.
- **STILL-ITERATING** = an open agent is still working the lever: weight-stationary #58, coalesced-decode #63,
  scale-load #64, algorithmic-softmax #65, decode-ridge #66. Everything else is **FINISHED** (RTL-final).

---

## 1. Projection / FFN GEMM primitives (fp8 / fp6 / fp4, real K) — the tensor-core path
These are the QKV/O/gate/up/down GEMMs. All run the upstream `gemm_mxgemmini` mesh lib; the levers layered on
are precision (fp8→fp6→fp4) and amortization (deeper K). 128²×K unless noted. Metric = `net_kernel_cycles`.

| kernel (radiance-kernels/kernels/) | op / shape | RTL cyc | MX whole-kernel util | cyc/MAC | correctness | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|---|
| gemm fp8 128²×512 | proj GEMM K=512 | **105,831** | 31.0% (mesh-in-loop 91%) | 0.0126 | read-back artifact; cycles valid | — (operating-point ref) | 1.00× ref | FINISHED | overhead-bound (small K) |
| `autocomp_gemm ... fp8` K=2048 | gate/up/QKV K=2048 | **209,116** | 62.7% | 0.0062 | " | fp8@512 | 2.0×/MAC (amortization) | FINISHED | no — K-scaling |
| fp8 down-proj K=5632 | down-proj K=5632 | **450,101** | **80.1%** (compute-win 95%) | 0.00488 | " | fp8@512 | 2.54×/MAC | FINISHED | **YES** (mesh at peak) |
| fp6 @K=2048 | accuracy-sensitive GEMM | **142,011** | 46.1% (236 MAC/cyc) | ~0.0043 | tohost=0 (fp6 0/16384) | fp8@2048 | 1.47× | FINISHED | overhead-fraction |
| `autocomp_gemm_fp4_k2048` | FFN GEMM K=2048 | **126,969** | 51.6% (264 MAC/cyc) | 0.0038 | tohost=0 (fp4 0/4096) | fp8@2048 | 1.65× (precision) | FINISHED | near (overhead-frac) |
| `autocomp_gemm_fp4_k2048_tk256` | K=2048, TILE_K=256 | **132,803** (whole-sim) | — | — | tohost=0, PASS | tk128 133,942 (whole-sim) | 1.0085× (deeper tile) | FINISHED | YES (2nd-order lever) |
| `autocomp_gemm_fp4_k5632` | down-proj K=5632 | **258,059** | 69.8%* (**358 MAC/cyc**) | **0.0028** | tohost=0 | fp8 down K5632 | **1.74×** (precision) | FINISHED | **YES** (fastest primitive) |

\*fp4 util reads "lower" only because it does 2× MAC/cyc; **cyc/MAC 0.0028 is the honest signal — fp4 is the
fastest GEMM primitive.** Decomposition (#654): overhead clusters at ~60-90k cyc regardless of compute size →
**mesh runs ~95-100% of peak in the K-loop; the whole-kernel gap is scale-load-dominated fixed overhead, NOT
mesh idle.** Deeper TILE_K only buys ~1% (fill/drain already amortized). RTL precision ladder @K2048:
**fp4 126,969 < fp6 142,011 < fp8 209,116.**

---

## 2. FFN block (fused, fp4-MX, real dims)
RMSNorm→gate/up→SwiGLU→down→ResAdd chained; matmuls on the fp4 mesh, all glue fused. Metric = `trace_max` (DRAIN=0).

| kernel | op | RTL cyc | util | cyc/MAC | correctness | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|---|
| ffn_serial (128²×512+SiLU) | per-op FFN-GEMM | **104,174** | 31% | — | read-back artifact; cycles valid | — (our per-op) | 1.00× ref | FINISHED | — |
| **ffn_fused4w** (128²×512+SiLU) | GEMM+SiLU, SMEM epilogue | **83,369** | 39.3% | 0.0099 | correct-by-construction | ffn_serial | **1.25×** (fusion) | FINISHED | fusion ceiling (#1082) |
| combined stack fp4@K2048 + fused SiLU | GEMM+epilogue | **105,132** | 62% | **0.0031** | fp4 tohost=0 + SiLU cbc | fp8-serial@K512 (0.0124) | **4.0×**/MAC (stack) | FINISHED | — |
| **`autocomp_ffn_block_fp4`** (M64/HID512/FN64/DN512) | full fused FFN block | **553,879** (trace_max 556,759) | — | ~0.37 (est) | cyclotron tohost=0, **bit-exact** | matched fp8-MX (`autocomp_ffn_block_fp8`) = **STRUCTURALLY-UNMEASURABLE** (l0d deadlock, see below) | *(context: at this block's small K≤512, the RTL precision ladder has fp8 FASTER than fp4 — fp4@128²×512 217,455 vs fp8 105,831 — so fp4 is likely NOT a win at these dims; fp4 only wins at K≥2048)* | FINISHED | — |
| `autocomp_ffn_block_noxnrt` | +Xn-round-trip-elim | *matched RTL deferred* | — | — | tohost=0, bit-exact | ffn_block_fp4 | cyclotron −20.4%; **RTL says Xn-elim UNREALIZABLE** (deadlock/SFUPipe assert) | FINISHED (negative) | — |

Fusion building blocks (all RTL, all cyclotron tohost=0 bit-exact; cyclotron over-counted these 2.4-9.7×):

| block | RTL cyc | speedup vs cyclotron self | status |
|---|---|---|---|
| RMSNorm-prologue (`fused_rmsnorm_qkv`) | **64,400** | 9.7× | FINISHED |
| RoPE-epilogue (`fused_qkv_rope_d64`) | **103,446** | 2.4× | FINISHED |
| activation-quantize (`fused_matmul_quant`) | **201,686** | 2.4× (1.85× vs standalone quant) | FINISHED |

---

## 3. Attention (flash, head_dim=64 — one real TinyLlama head)
Fused MX QKᵀ/PV + SIMT online-softmax. Fork of upstream `flash_attention_mx_yrh` (d128→d64). Metric = core-cyc trace_max.

| kernel | op / shape | RTL cyc | util | correctness | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|
| **`autocomp_fa_d64`** | single head Sq64/Sk256/d64 | **217,099** (215-217k across runs) | MX-mesh **0.4%**, SIMT-worker 75%, both-active 0.14% | golden rel-err **4.16%** (in-band fp8) | *(no matched-shape upstream RTL — GAP)* | — (baseline) | FINISHED (#65 tpr won 1.11×) | softmax-bound |
| **`autocomp_fa_d64_tpr`** | +thread-per-row single-exp softmax (0 per-row fences) | **195,328** (both cores $finish) | softmax phase −18.8% (fence-drain reclaimed) | bit-identical P/scales, rel-err **4.16%** (unchanged) | fa_d64 | **1.11× WIN** (#65) | FINISHED (RTL-final) | SFU-exp latency-bound |
| `autocomp_fa_d64_occ3` | +occ=3 | 223,592 | LSU 2.77 | rel-err unchanged | fa_d64 | 0.99× (worse) | FINISHED (negative) | — |
| `autocomp_fa_bcfree` | bank-conflict-free softmax | 223,471 | LSU 1.14 (=1.15 base) | bit-exact, rel-err 4.16% | fa_d64 | **+3.5% SLOWER** | FINISHED (negative) | — |
| `autocomp_fa_bcfree2` (uint16-skew) | lower-overhead bcfree | *cyclotron parity; RTL parity expected* | — | bit-exact | fa_d64 | ~1.0× | FINISHED (negative) | — |
| `autocomp_fa_gqa_causal` | **8-head GQA+causal** | **809,035** (completeness-verified: 16384 O-stores = 8/8 heads, both cores $finish) | MX-mesh softmax-bound | golden rel-err **6.09-6.10%** | 8× single-head fa_d64 (1,736,792) | **2.15×** (causal block-skip + GQA KV-share) | FINISHED (RTL-final) | softmax-bound |
| `autocomp_fa_gqa8_tpr` | +thread-per-row softmax on 8-head GQA | **991,720** (16384/16384 O-words ✅, both cores $finish) | fewer insts (388k vs 452k) but +cyc: SFU-latency exposed | bit-identical P/scales, rel-err **6.10%** (unchanged) | fa_gqa_causal (809,066) | **0.816× REGRESSION** (#66) | FINISHED (negative) | tpr does NOT survive to 8-head |
| `autocomp_fa_gqa8_tpr_occ3` | +occ=3 (recovery attempt) | **RTL-UNRUNNABLE** (GP-faults host ~30s; register-wall 256/256 = Rename.scala:123) | — | — | — | occ3 can't rescue tpr (#66) | FINISHED (negative) | — |
| `autocomp_fa_d64_overlap` / `_ovl2` | async QKᵀ‖softmax | **DEADLOCK on RTL** (l0d backpressure) | — | — | — | DEAD (DUT-structural) | FINISHED (negative) | — |

RTL phase attribution (MARK trace): softmax **64.6%** · PV 13.2% · prologue-QK 10.1% · rescale 9.7% · finalize 2.4%.
Softmax cost is **SFU-exp latency (64 mu_fexp/row) + fence/barrier drains — NOT bank conflicts** (bcfree refuted it,
net-negative). All SMEM-layout / occ / precision levers are exhausted (~1.0× stack). The **ALGORITHMIC** lever
(cut fence count via thread-per-row softmax) RESOLVED: **#65 single-head fa_d64 = 1.11× WIN** (0 per-row fences,
softmax phase −18.8%), but **#66 8-head GQA-causal = 0.816× REGRESSION** — removing fences exposes SFU-exp latency
that the cooperative version's fence/barrier warp-sync had been hiding; the win is problem-shape-specific and does
NOT survive to the deployable 8-head kernel (occ3 can't rescue: unrunnable register-wall + structurally idle at 64
rows). Deployable attention stays cooperative (`fa_gqa_causal` 809,035). Remaining softmax lever: a fence-COUNT trim
that keeps 16-lane column-splitting (latency-hiding), or a cheaper exp approx (gated on rel-err). Attention is
**not** a both-engines-busy win at d=64 (per-block matmul ~100 cyc vs softmax ~100× longer → concurrency ceiling
~0.4%, and overlap deadlocks).

---

## 4. Overlap (warp-specialized MX‖SIMT co-execution)
Metric = core-0 inst span / dmem-span. Baselines are our own serial (barrier-fenced) twins.

| kernel | op | RTL cyc | both-active% | correctness | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|
| `autocomp_overlap2` | 2-tile warp-spec | **99,239** | ~18% | tohost=0 | overlap2_serial 117,258 | **1.18×** | FINISHED | — |
| single-tile fp8 GEMM+SiLU overlap | 1-tile | 100,091 | 17.9% | tohost=0 | serial 118,044 | 1.18× | FINISHED | — |
| **`autocomp_overlap_layer_banked`** | 3-tile layer, prefetch-once | **171,698** (171,931) | **14.0%** (of 17.5% hard cap = 80%) | tohost=0 | serial 220,056 | **1.28×** | FINISHED | **YES** (overlap ceiling) |
| `autocomp_overlap_layer` (broken re-DMA) | 3-tile, re-DMA/tile | 206,650 | ~0 | — | serial 220,056 | 1.06× (regression) | FINISHED (superseded) | — |
| `autocomp_overlap_heavy` (W4) | heavy epilogue | 291,501 | — | tohost=0 | serial 294,729 | 1.011× (backfires) | FINISHED (negative) | — |
| `autocomp_overlap_heavy8` (W8) | heavier epilogue | 366,454 | — | tohost=0 | serial 361,618 | **0.987×** (backfires) | FINISHED (negative) | — |

**1.28× is the confirmed overlap ceiling** for single-core warp-spec (#712, #969): heavier concurrent SIMT starves
the manager warp's mesh-issue port → monotonically worse. Root cause of the earlier layer regression = operand
re-DMA contending with epilogue writes; fixed by prefetch-once (banked). Spatial core-specialization (pin manager to
own core) does NOT beat 1.28× (register-wall + SFUPipe assert + wrong bottleneck). Bound by FLOP-asymmetry
(SIMT epilogue <10% of layer FLOPs).

---

## 5. Batched decode → MX mesh (weight-stationary M-batching)
out[M,128]=X[M,2048]@W, K=2048, one N=128 tile, bf16 out (dodges requantizer). Metric = net_kernel_cycles. All tohost=0 (bf16 golden).

| kernel | op | RTL cyc | cyc/token | cyc/MAC | eff weight-BW | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|---|
| `autocomp_dec_batched` fp8 **M32** | decode proj, 32 tok | **47,314** | 1,479 | 0.00564 | ~7.1 B/cyc (at DRAM peak) | M=1 GEMV (infeasible) | — | STILL-ITERATING (#66) | weight-BW |
| `autocomp_dec_batched_fp8_m64` **M64** | decode proj, 64 tok | **48,673** | **760** | **0.00290** | — | fp8 M32 | **1.95×/token** | STILL-ITERATING (#66) | no — push M=96/128 |
| `autocomp_dec_batched_fp4` M32 | fp4 twin | 96,797 | 3,025 | 0.01154 | — | fp8 M32 | **2.05× SLOWER** | FINISHED (negative) | — |
| `..._fp4_m64` | fp4 M64 | 110,626 | 1,728 | 0.00659 | — | fp8 M64 | 2.27× SLOWER | FINISHED (negative) | — |
| `autocomp_dec_gemv_2048` M=1 SIMT GEMV | single-stream decode | **l0d assert — cannot run** | — | — | — | — | — | FINISHED (DUT-blocked) | l0d wall |

**fp4 LOSES for decode** (small-M → fp4 per-projection overhead swamps byte-halving; TILE_K-invariant). The decode
floor is **fp8**, not fp4. fp8 M=32≈M=64 (+2.9% cyc for 2× tokens) → **push M higher** = open agent #66.

Decode GEMV latency probe (single-stream, refutes "DRAM-BW floor" → it's an **l0d-backpressure floor**):

| kernel | RTL cyc | out stores | eff BW | verdict |
|---|---|---|---|---|
| `autocomp_bw_17` (ceiling) | **250,033** | n/a | **2.10 B/cyc** | DRAM streaming ceiling |
| `autocomp_gemvlat_base` (uncoalesced 4W ILP2) | **1,443,950** | 128/128 ✅ | 0.363 B/cyc (17%) | completing decode floor |
| `autocomp_gemvlat_ilp4` | 1,779,134 | 128/128 ✅ | 0.295 | 23% SLOWER (reg pressure) |
| `autocomp_gemvlat_ilp8` | 1,713,415 | 64/128 ✗ | — | l0d deadlock (incomplete) |
| `autocomp_gemvlat_coal` (coalesced [K,N]) | 166,591 | 64/128 ✗ | — | l0d deadlock (span < floor = impossible) |

Completing GEMV = 17% of BW ceiling → **latency-bound, not BW-bound**; but every closing lever (coalescing, deeper
ILP) DEADLOCKS on the l0d no-landing-pads wall (DUT, forbidden). Coalesced [K,N] is open agent #63.

---

## 6. Weight-stationary reuse (read-once dataflow)
Same C, weight residency varied. Metric = net_kernel_cycles. Both tohost=0, both RTL-complete (store-count checked).

| kernel | op | RTL cyc | cyc/MAC | weight DRAM reads | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|
| **`autocomp_ws_after`** (read-once) | C[256,64] fp8 K=2048 | **216,514** | 0.00645 | 32 tiles / 128KB / **1×** | ws_before | **1.332×** | STILL-ITERATING (#58) | — |
| `autocomp_ws_before` (M-outer re-stream) | same | **288,355** | 0.00859 | 128 tiles / 512KB / **4×** | — | 1.00× ref | STILL-ITERATING (#58) | — |
| `autocomp_redma_probe` R=0/1/3 | L1-resident re-DMA probe | **343,783** (bit-identical all R) | — | 1×/2×/4× (dead scratch) | — | re-reading L1-resident weights is **FREE** | FINISHED | — |

**Proper read-once weight-stationary = 1.33× on RTL** (refutes the earlier "1.013×, can't be fixed" — that was a
drain-polluted trace_max on an A/B that never varied residency). L1-hit weight re-reads are free; L1-evicted
(block-separated) re-reads cost real L2/DRAM. Needs non-square tall tiles (TM=256 → TN≤64 SPAD fit).

---

## 7. Full fused TinyLlama LAYER
attention block + FFN block chained, residual flowing. Metric = trace_max (DRAIN=0 both sides). M64/H512/D64/FN64.

| kernel | op | RTL cyc | stores | correctness | baseline | speedup | status | at ceiling? |
|---|---|---|---|---|---|---|---|---|
| **`autocomp_layer`** (16-launch, epilogue-fused) | full layer | **445,650** | 24,349 ✅ | cyclotron tohost=0 bit-exact | — (per-op layer) | 1.00× (FASTEST realizable) | FINISHED | **YES** (fusion tops 1.25×) |
| `autocomp_layer_persist` (1-launch) | launch-collapse | **457,900** | 17,211* | tohost=0 | autocomp_layer | **0.973× (2.7% SLOWER)** | FINISHED (negative) | — |
| `autocomp_layer_maxstack` / `_mega` (+Xn-elim) | max stack | **DEADLOCK / TRUNCATED — INVALID** | 1,739 | — | — | — | FINISHED (negative) | — |
| `autocomp_layer_xnelim` (multi-launch +Xn-elim) | Xn-elim | **SFUPipe assert @126,664** | — | — | — | fails | FINISHED (negative) | — |

\*CORRECTED: the earlier "persist 1.264×" was a **TRUNCATED** run (17,211 of 24,349 stores = 71%). The complete
persist run is 457,900 = **0.973× (slower)**. Launch-collapse and Xn-elim are both neutral-to-negative or
deadlock on RTL. **The epilogue-fused per-op layer (autocomp_layer, 445,650) is the fastest realizable layer** —
the whole-layer megakernel does NOT beat it. Cross-layer weight prefetch is the one untested layer-scale lever
(real-dim/weight-bound only; unmeasurable on this RTL — toy dims see <0.1%, real dims don't retire).

---

## 8. NO RTL NUMBER — not final (cyclotron-functional only; EXCLUDED from results per §0)
These are correctness-verified (cyclotron tohost=0 / golden) but have **no valid RTL cycle count** → not final.

| kernel | op | cyclotron cyc (functional, NOT perf) | correctness | why no RTL |
|---|---|---|---|---|
| `autocomp_lmhead_multitile` | LM-head, 8 N-tiles | 1,928,904 | tohost=0 | multitile RTL not run (≈8× single-tile ≈ ~1.0M; single-tile now RTL-gated below) |
| **fp4 LM-head tile K=2048 (RTL)** | LM-head GEMM primitive (128²×2048) | **128,322 RTL** (net, output-complete; 8192/8192 stores) | cyclotron tohost=0 bit-exact (258,290); RTL verify = read-back artifact (cycles valid) | ✅ RTL-gated (`autocomp_lmhead_st`, `-DNUM_TILES=1`). Cross-validates fp4 GEMM primitive 126,969. |
| `autocomp_ffn_block_fp4_hid2048` | real HID=2048 FFN | 131,072 outs, 0 mismatch | bit-exact | ELF lacks tohost/fromhost → won't retire (subsumed by e2e) |
| `autocomp_attn_block` | fused attention block | 4,604,197 | tohost=0 bit-exact | never RTL-gated |
| `autocomp_tinyllama_e2e` | full forward ×2 layers | 4,009,936 | **bit-exact** (0/16384 logits) | e2e doesn't retire on RTL (l0d/timing) |
| `autocomp_tinyllama_e2e_real` | e2e at real hidden=2048 | 8,487,951 | **bit-exact** tohost=0 | " (real-dim doesn't retire) |
| `autocomp_fa_gqa_causal` | 8-head GQA attention | 805,958 | golden rel-err 6.09% | ✅ NOW RTL-FINAL: **809,035** (§3, 8/8 heads verified) — moved to attention table row above |
| `autocomp_dec_gemv_2048` | decode GEMV @2048 | 33.5M | tohost=0 | l0d assert on RTL (can't run) |

**e2e cyclotron cycles (4.0M / 8.49M) are CORRECTNESS evidence, NOT performance** (§0). No RTL cycle count exists for
the whole model; a real e2e timing number requires per-layer RTL ×22 summed.

---

## GAPS — missing upstream-baseline comparisons (do NOT fabricate)
The precision/amortization/fusion speedups above are measured against **our own** fp8 operating point or serial
twins — solid within-family. What is genuinely **missing** is a matched-shape comparison to the **upstream
radiance-kernels** baselines, and RTL numbers for a few finished-on-cyclotron kernels:

1. ✅ **FILLED — 8-head GQA-causal now has a clean, completeness-verified RTL number: 809,035 cyc**
   (`autocomp_fa_gqa_causal`, private-binary Verilator run, 16384 O-stores = 8/8 heads, both cores $finish;
   matches cyclotron 805,958 within 0.4%). Real speedup: vs a naive 8× independent single-head fa_d64
   (8×217,099 = 1,736,792) the causal block-skip (2 of 4 key blocks) + GQA KV-sharing give **2.15×**.
   *(Still open, secondary: a matched-shape fa_d64-vs-upstream comparison — upstream 243,650 is d128, ours is d64.
   Not run: would need either the upstream kernel rebuilt at d64 or ours at d128; lower value than the GQA number.)*
2. ✅ **FILLED — LM-head fp4 GEMM primitive now RTL-gated: 128,322 cyc** (`autocomp_lmhead_st` `-DNUM_TILES=1`,
   net output-complete; 8192/8192 C_multi stores). Cross-validates the fp4 GEMM primitive (126,969). The 8-N-tile
   `lmhead_multitile` (cyclotron 1,928,904) was not RTL-run (≈8× the single-tile primitive, ~1.0M cyc, long); the
   single-tile primitive is the measurement the gap asked for ("fp4 single-tile and/or multitile").
3. ⛔ **STRUCTURALLY-UNMEASURABLE — matched fp8-MX FFN baseline (`autocomp_ffn_block_fp8`) does NOT retire on RTL.**
   Built a faithful matched baseline: identical fused chain / M64-HID512 dims, only the 3 MX matmuls' precision
   differs — fp8-e4m3 (upstream native format) vs fp4-e2m1 — with bit-exact runtime quantizers lifted verbatim
   from `autocomp_simt_quant` (proven Python↔Muon) and a finite, non-degenerate mx_golden (fmt=0). BUT it
   **deterministically deadlocks on RTL at cycle 559,486** (reproduced on 2 independent Verilator runs): warp 1
   stalls in `rmsnorm_body`'s store-drain (PC 0x10031880) while warp 0 spins at the `mu_schedule` barrier — the
   documented **l0d-no-landing-pads store-drain backpressure wall**, tripped by fp8's larger memory footprint
   (2× quant buffers + 2× weight bytes shift the map). The byte-identical fp4 twin retires fine (553,879), and
   the fp8 hang lands in RMSNorm **before any fp8 matmul executes** (out_raw = 0/32768). Independently, the
   CYCLOTRON_MXGEMMINI co-model errors on the param-lib+fp8 path (the fp4 control passes tohost=0), so there is
   no cyclotron fallback either. This is DUT-structural (l0d, forbidden to touch from kernel code) — same class
   as the async-overlap deadlock and decode-GEMV l0d wall. **Not faked.** Nearest measured context: the RTL
   precision ladder shows fp8 BEATS fp4 at K≤512 (fp8 105,831 < fp4 217,455 @128²×512), so a matched fp8 FFN at
   these small-K block dims would likely be FASTER than ffn_block_fp4 — i.e. fp4 is probably the wrong precision
   here (fp4 only wins at K≥2048), but this cannot be turned into a clean × without the fp8 block retiring.

(Secondary: `ffn_block_noxnrt` matched-RTL delta was deferred — but RTL later showed Xn-elim is unrealizable
anyway, so this gap is effectively closed as a negative.)

**Update (RTL runs landed, quiet window):** Gaps 1 & 2 are FILLED with completeness-verified private-binary Verilator
runs (GQA-causal 809,035 @ 8/8 heads; LM-head fp4 tile 128,322 @ 8192/8192 stores). Gap 3's matched fp8-MX FFN
(`autocomp_ffn_block_fp8`) was built + RTL-run but is **structurally-unmeasurable** — it deterministically deadlocks
in the RMSNorm store-drain on the l0d wall (see §2 / gap 3), so no matched × exists (not faked). Method: each sim quiet-gated
(load < 8), niced, run from a uniquely-named private copy of the Verilator binary (siblings pkill `simulator-chipyard`),
completeness-verified by dmem store-count before any number is credited.
