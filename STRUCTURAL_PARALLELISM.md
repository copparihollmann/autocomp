# Structural (graph-level) parallelism map — Radiance dual-engine co-execution

**Purpose.** A static, schedule-independent map of every region of each priority model's compute
DAG whose ops carry **no data dependency** between them and can therefore run **concurrently** on
Radiance's two engines (MX-Gemmini tensor core + the 2 Muon SIMT cores), or be pipelined. This is
the "what CAN legally run in parallel" complement to the runtime bubble-elimination effort. Other
agents (esp. the whole-layer co-scheduler) exploit this map; this file does **not** build kernels.

Ground truth: op dataflow read directly from `model2MLIR/workloads/tiny_llama/tiny_llama.mlir`
(Q/K/V = matmul_1/2/3 all consume the RMSNorm output; gate/up = matmul_7/8 both consume the
post-attn norm output; down = matmul_9 consumes SwiGLU) and the four weight manifests
(`tiny_llama`, `smolvla`, `gemma2_2b`, `deepseek_qwen_1_5b`). All four are **plain serial
transformers**: attn → FFN, no MoE, no PaLM/GPT-J parallel attn+FFN block. So the *intra-layer*
independence structure is common; only dims and a few elementwise ops differ.

---

## 0. The engine asymmetry that sets the mapping (read this first)

The two engines are **not** symmetric, so "independent" does not automatically mean "run one on
each engine and win":

| Engine | Peak | Best at |
|---|---|---|
| **MX-Gemmini** (16×16 systolic) | 256 MAC/cyc (fp8), 512 (fp6/fp4) | dense GEMM / GEMV (matmul, QKᵀ, PV, LM-head) |
| **2× Muon SIMT** | ≈128 FLOP/cyc combined | elementwise, reductions, transcendentals (exp/SiLU/GELU/rsqrt/tanh), gather, quantize, masking |

Two consequences that drive the ranking:

1. **matmul ‖ matmul is NOT free concurrency.** Two independent GEMMs (e.g. gate ‖ up) both want
   MX; running one on SIMT is ~2–4× slower per FLOP. So matmul-pair independence is exploited by
   **pipelining on MX** (or splitting a big op's tiles), *not* by "GEMM-on-SIMT". It raises
   both-engines-active% only if the SIMT half is small enough that its slower rate still finishes
   inside the MX half's time.
2. **The natural dual-engine fit is `MX GEMM ‖ independent SIMT elementwise`** — warp-specialised
   epilogue/prologue overlap. But note the **FLOP imbalance**: elementwise is <1% of model FLOPs,
   so a single fused activation is ~100× cheaper than the GEMM it hides behind and leaves SIMT
   ~99% idle. To get SIMT *meaningfully* busy concurrent with MX you need one of the few
   **S²-scaling** or **large-N** SIMT regions (attention softmax; LM-head/vocab), or to hand SIMT
   a whole *small* GEMM (K/V proj) or a slice of a big one.

**Net:** both-engines-active% is bounded low by pure per-op elementwise overlap; the real leverage
sits in (a) **flash-attention** (softmax is inherently SIMT and interleaves with the QKᵀ/PV
matmuls, and both grow as S²), (b) **LM-head vocab-tiling** (huge N, splittable MX‖SIMT), and (c)
**aggregating** every layer's elementwise into a steady SIMT pipeline that runs across the whole
MX GEMM chain so SIMT is never fully drained.

Regime matters: **prefill / VLA-vision = compute-bound** (weights reused across S tokens) → overlap
wins; **decode = weight-BW-bound GEMV** (S=1) → the independence is used to overlap **weight-DMA**
with compute, and elementwise is negligible.

---

## 1. TinyLlama-1.1B (TOP PRIORITY) — per-layer independence

Dims: d=2048, FFN F=5632, 32 Q / 4 KV heads (GQA 8×), head_dim 64, 22 layers, vocab 32000.
Per-layer GEMM work in **MACs** (prefill seq S), and the SIMT elementwise between them:

| Op | shape (M×N×K) | MACs | MX cyc @256 | engine-natural |
|---|---|---|---|---|
| Q proj | S×2048×2048 | 4.19M·S | 16.4k·S | MX |
| K proj | S×256×2048 | 0.524M·S | 2.0k·S | MX or SIMT |
| V proj | S×256×2048 | 0.524M·S | 2.0k·S | MX or SIMT |
| QKᵀ (32 h) | S×S×64 ×32 | 2048·S² | 8·S² | MX |
| softmax | 32·S×S | ~5·32·S² flop | — | **SIMT** |
| PV (32 h) | S×64×S ×32 | 2048·S² | 8·S² | MX |
| O proj | S×2048×2048 | 4.19M·S | 16.4k·S | MX |
| gate | S×5632×2048 | 11.53M·S | 45.0k·S | MX |
| up | S×5632×2048 | 11.53M·S | 45.0k·S | MX |
| SiLU(gate) | S×5632 | ~10·S·5632 flop | — | **SIMT** |
| SwiGLU mul | S×5632 | S·5632 flop | — | **SIMT** |
| down | S×2048×5632 | 11.53M·S | 45.0k·S | MX |
| RMSNorm ×2 | S×2048 | ~5·S·2048 flop | — | **SIMT** |
| RoPE(Q,K) | S×2304 | ~6·S·2304 flop | — | **SIMT** |
| residual ×2 | S×2048 | S·2048 flop | — | **SIMT** |
| **LM-head** (final) | S×32000×2048 | 65.5M·S | 256k·S | MX (N-tileable) |

FFN (gate+up+down = 34.6M·S) dominates per-layer MX time; attn projections (QKVO = 9.4M·S) are
next; attention QKᵀ/PV scale as S² and overtake at long seq (S≳1500 QKᵀ = O-proj).

### 1.1 Ranked independent regions (TinyLlama)

| # | Region | Independent ops | Independence argument (why no dep) | MX / SIMT mapping | Leverage |
|---|---|---|---|---|---|
| **T1** | **Flash-attention inner loop** | QKᵀ(block) · online-softmax(max/exp/sum/rescale) · PV(block), across KV blocks & 32 heads | Heads are fully independent (block-diagonal). Within a head, online softmax of KV-block *j* depends only on QKᵀ(j); PV(j) on softmax(j); but block *j+1*'s QKᵀ is independent of block *j*'s softmax/PV → classic 3-stage pipeline. | MX: QKᵀ(j+1) and PV(j) ‖ SIMT: softmax+rescale(j). Softmax is **S²**-scaling SIMT work that fills MX time at real seq. | **Highest at prefill/long-ctx.** Both engines inherently busy; softmax grows S² so both-active% rises with sequence. Decode (S=1): negligible. |
| **T2** | **FFN gate ‖ up, + SiLU/SwiGLU/quant pipeline** | gate, up (both read post-attn RMSNorm output); SiLU(gate); SwiGLU=gate·up; down; down-residual; quantize | gate ⟂ up (same input, disjoint outputs). SiLU(gate) needs only gate. SwiGLU needs both. down needs SwiGLU. → gate→SiLU can run **while MX computes up**. | MX streams gate→up→down (pipelined on MX). SIMT: SiLU(gate) hidden behind up; SwiGLU-mul + fp8-quantize of down input hidden behind down; **down's residual-add + next-layer input-RMSNorm hidden behind down's own tail / next Q-proj**. | **High.** Largest MX block/layer (34.6M·S). SIMT per-op cheap but the chain keeps SIMT fed continuously; enables the layer-fused megakernel's steady overlap. |
| **T3** | **Q ‖ K ‖ V projections (+ RoPE)** | Q, K, V (all read the pre-attn RMSNorm output); RoPE(Q), RoPE(K) | Three GEMMs, one shared input, disjoint weight/outputs → mutually independent (MLIR matmul_1/2/3 all consume %285/%291). RoPE(Q) ⟂ V-proj, RoPE(K) ⟂ Q-proj. | **Split:** Q on MX (16.4k·S) ‖ K+V on SIMT (1.05M·S → 8.2k·S, finishes inside Q). Then RoPE(Q,K) on SIMT ‖ MX starts QKᵀ or (in fused) O-proj weight prefetch. GQA: K,V are 8× narrower (256 vs 2048) → cheap, ideal SIMT candidates. | **Medium.** K/V small enough that SIMT covers ~50% of Q's MX window → real both-active. RoPE overlap is a clean bonus. |
| **T4** | **LM-head vocab N-tiling** (final token(s) only) | logits = h·Wₑᵀ, N=32000 split into independent N-tiles | Output columns (vocab tiles) are fully independent — no reduction across N. 32000/128 = 250 independent tiles. | Split N across MX and SIMT: MX does most tiles, SIMT does a fraction concurrently; or overlap argmax/softmax(logits) (SIMT) with remaining MX tiles. | **Medium-high (decode).** 65.5M·S MACs, the single biggest GEMM; in decode it's *the* work and is trivially splittable → direct both-active. |
| **T5** | **Cross-op tail overlap: residual+norm ‖ next matmul** | residual-add, input-RMSNorm of layer/sub-block *k+1* ⟂ the matmul whose output they don't consume | residual(attn-out) + RMSNorm feed FFN gate/up; they do **not** depend on the FFN weights → can start as O-proj tiles land, overlapping O-proj tail. Same at layer boundary. | SIMT: residual + RMSNorm(k+1) ‖ MX: O-proj tail / first FFN GEMM prologue (weight DMA). | **Low-medium** individually; **valuable as the glue** that keeps SIMT busy at every op boundary (kills the ~14 between-op DRAM round-trips/layer). |
| **T6** | **KV-cache append ‖ current-token compute** (decode) | write K_t,V_t to cache ⟂ Q_t·Kcacheᵀ over *past* positions | The new token's K/V append is a copy/quantize (SIMT) independent of the attention read over cached positions for already-appended history. | SIMT: quantize+store K_t,V_t ‖ MX: begins the score GEMV against cache. | **Low** (tiny per step) but free; matters in long-decode (DeepSeek-style). |
| **T7** | **Cross-layer weight prefetch** (prefill) | layer N+1 weight DMA ⟂ layer N compute | Weights are read-only constants; N+1's gate/up/down weights have no dep on N's activations → DMA can begin during N's tail. | DMA engine / SIMT-issued prefetch ‖ MX computes layer N. | **Low-medium**; a latency-hiding lever, weights are 4–42MB/layer streaming from DRAM. Not "both-compute-engines" but overlaps the memory wall. |

**TinyLlama verdict:** coverage-complete → pure performance. The overlap ladder is
**T1 (attention, prefill) > T2 (FFN pipeline) > T4 (LM-head, decode) > T3 (QKV split) > T5/T6/T7**.

---

## 2. DeepSeek-R1-Distill-Qwen-1.5B — same skeleton + QKV bias + long decode

Dims: d=1536, FFN 8960, 28 layers, 12 Q / 2 KV heads (GQA 6×), head_dim 128, vocab 151936.
Structure identical to TinyLlama (SwiGLU/RMSNorm) with two differences that add parallelism:

- **QKV bias** (Qwen2): each of Q/K/V has an independent bias-add. The bias-add of Q ⟂ K-proj GEMM,
  etc. → three independent `GEMM(MX)+bias(SIMT)` epilogues; SIMT bias-add of Q hidden behind K/V MX.
- **Reasoning model → very long decode (CoT).** Decode is the dominant regime and is **weight-BW-
  bound GEMV**. Here the top lever is **T7-style weight-DMA ‖ GEMV compute** and **T4 LM-head split**
  (N=151936 → 1187 tiles, the biggest vocab of the four). KV-cache grows large → **T6** append
  ‖ compute becomes non-trivial over a long context.
- FFN is proportionally the largest (8960/1536 = 5.8× vs TinyLlama's 2.75×) → **T2** gate‖up pipeline
  is even more dominant.

Ranking (decode-weighted): **T4 LM-head split ≈ T7 weight-DMA overlap > T2 FFN > T1 attention (long
ctx) > QKV bias epilogues (T3′) > T6 KV append.**

---

## 3. Gemma-2-2b — extra norms + soft-cap + sliding window (more SIMT to overlap)

Dims: d=2304, FFN **9216**, 26 layers, **8 Q / 4 KV** heads (GQA 2×), head_dim **256**, vocab
**256000**. GeGLU (GELU) FFN. Confirmed from manifest: **4 RMSNorms/layer** (input,
post_attention, pre_feedforward, post_feedforward) — Gemma2's pre+post-norm sandwich.

Gemma2 is the model where **SIMT has the most independent work to overlap** — good for both-active%:

| Extra SIMT op | Independence / overlap |
|---|---|
| **4 norms/layer** (vs 2) | each (1+w)·RMSNorm ⟂ the matmul it doesn't feed; post_attention & post_feedforward norms overlap the *next* GEMM's prologue. 2× the norm-overlap opportunities of TinyLlama. |
| **GELU (erf)** in GeGLU | transcendental, ~2× the SIMT flops of SiLU → fills more of the up-proj MX window (T2 with a heavier SIMT half — better balance). |
| **Logit soft-cap** `c·tanh(x/c)` | applied to attention scores (⟂ next head's QKᵀ) **and** final logits (⟂ remaining LM-head N-tiles). tanh is SIMT; fuses into softmax / LM-head epilogue and overlaps MX. |
| **Sliding-window mask** (alt layers) | masking is SIMT, ⟂ the QKᵀ of other heads/blocks; **and** windowed layers have *bounded* KV span → their attention is cheaper, changing the T1 balance per-layer but not the independence. |
| **head_dim 256, only 8 heads** | fewer, fatter heads → each QKᵀ/PV block is bigger; head-parallel T1 has coarser granularity (8-way) but each unit is larger. |
| **Biggest FFN (9216) & vocab (256000)** | T2 and T4 leverage largest of all four. LM-head N=256000 → 2000 tiles, the most splittable. |

Ranking: **T2 FFN (biggest) ≈ T4 LM-head (biggest vocab) > T1 attention (soft-cap+window add SIMT)
> the 4-norm/soft-cap epilogue overlaps (best SIMT-fill of the four models).**

Note: sliding-window vs full-attention **alternating layers are still sequential** (layer k+1 needs
k's residual) — the alternation does not create cross-layer parallelism, only per-layer attention
cost variation.

---

## 4. SmolVLA — the multi-tower model (where the big cross-block independence lives)

Confirmed module structure (manifest): `vlm_with_expert.vlm` = SigLIP **vision_model** (192 encoder
weights ≈ 27 layers, dim 768 / FFN 3072, LayerNorm+GELU+bias) + **text_model** (18 layer weights,
dim 960 / FFN 2560, RMSNorm/SwiGLU, 15 Q / 5 KV heads) + `connector.modality_projection`; plus
`lm_expert` **action expert** (dim 720, flow-matching). Denoise step (loader): `embed_prefix`
(vision+lang+state) → `vlm_with_expert.forward` (prefix, builds KV cache) → `denoise_step` (expert).

**Honest read on "vision ‖ text" — it is real but not two-big-blocks-overlap:**

- The **vision tower is by far the largest compute block** (27 SigLIP layers on ~hundreds of image
  patches, dim 768/FFN 3072). It runs in `embed_prefix`.
- It is **independent of language-token embedding and state_proj** — but those are *trivial* (a
  gather + one small GEMV). So during the vision tower, the "text side" has almost nothing to do:
  it's vision-dominated, not a balanced vision‖text race. Data-flow-wise **vision feeds the LM**
  (image tokens are concatenated into the LM sequence) → vision **precedes** text_model, not
  concurrent with it.

**Where SmolVLA's genuine large-scale independence actually is (ranked):**

| # | Region | Independence argument | Mapping | Leverage |
|---|---|---|---|---|
| **S1** | **Multi-camera vision towers** | SmolVLA config supports up to 3 camera views; each image goes through the **same** SigLIP tower with **no cross-view dependency** until concatenation → embarrassingly parallel. (Capture uses 1 cam; check deployment cam count.) | Pipeline views across engines / batch on MX; near-linear. | **Huge *if* >1 camera.** Fully independent towers. Conditional on deployment. |
| **S2** | **Vision tower internal T1/T2** | Same intra-layer independence as any transformer, applied to the biggest block: SigLIP LayerNorm+GELU+bias (SIMT) ‖ its GEMMs (MX); patch/head parallelism in its attention. LayerNorm (mean-sub) + GELU + bias give **more SIMT work** than an RMSNorm block → good balance. | MX: SigLIP QKV/FFN GEMMs ‖ SIMT: LayerNorm+GELU+bias epilogues, patch-parallel. | **High** — this is where most SmolVLA FLOPs are. |
| **S3** | **Prefix computed once ‖ multi-step denoise reuse** | Flow-matching runs several denoise steps; the VLM **prefix KV cache is computed once** and is **read-only** across all denoise steps → every expert step is independent of prefix recompute. | MX/SIMT dedicated to expert steps while prefix stays resident; no recompute. | **High for latency** (amortises the big VLM over N steps), though steps are sequential in the ODE. |
| **S4** | **Action-expert intra-step (self-attn/FFN) + cross-attn to frozen KV** | Expert tokens (chunk ~50, dim 720) cross-attend to the **frozen** prefix KV (independent of prefix compute); expert's own Q/K/V, gate/up follow the T2/T3 pattern at small dim. | Expert GEMMs on MX ‖ expert LayerNorm/SiLU on SIMT; cross-attn reads resident KV. | **Medium** (expert is small, dim 720). |
| **S5** | **State/time-MLP ‖ vision** | `state_proj`, `action_time_mlp` (flow-matching timestep embed, sin/cos) are tiny and **independent of the vision tower** → free SIMT work to hide under vision MX. | SIMT: time-MLP sin/cos + state_proj ‖ MX: vision tower. | **Low** (tiny) but genuinely concurrent. |

**SmolVLA verdict:** the "huge" independence is **S1 multi-camera** (if present) and **S2 the vision
tower being the FLOP-dominant block with LayerNorm+GELU+bias giving unusually good SIMT balance** —
*not* a literal vision‖text-expert race (vision feeds text; expert runs after and is small). Prefill-
heavy → compute-bound → overlap-favourable.

---

## 5. TOP 3 highest-leverage independent regions (hand-off to co-scheduling agents)

Ranked for **both-engines-active%** on the compute-bound (prefill/vision) regime that dominates the
priority workloads, TinyLlama-first:

1. **Flash-attention inner loop — online-softmax (SIMT) ‖ QKᵀ/PV block matmuls (MX)** [T1].
   Heads are block-independent; within a head the KV-block pipeline lets SIMT do softmax(j)+rescale
   while MX does QKᵀ(j+1)/PV(j). **Uniquely, the SIMT half (softmax) scales as S²**, so it actually
   fills MX time at real sequence length — the only per-op region where both engines are inherently,
   sustainably busy. Applies to all four models (Gemma2 adds soft-cap+window SIMT work). *Prefill/
   long-context only; negligible in S=1 decode.*

2. **FFN gate/up independence → SiLU/GELU + SwiGLU + quantize pipeline (SIMT) ‖ gate→up→down (MX)**
   [T2]. The largest MX block per layer (TinyLlama 34.6M·S; Gemma2/DeepSeek even more FFN-heavy).
   gate ⟂ up (same normed input, disjoint outputs); SiLU(gate) hides behind up; SwiGLU-mul +
   down-input fp8-quantize hide behind down; down's residual + next input-RMSNorm hide behind the
   next Q-proj. This is the steady SIMT-feed that makes the layer-fused megakernel's warp-spec
   overlap real. *Wins in both prefill and decode.*

3. **LM-head vocab N-tiling (MX‖SIMT split) — and, in decode, weight-DMA ‖ GEMV** [T4/T7].
   The single biggest GEMM (TinyLlama 65.5M·S; Gemma2 N=256000; DeepSeek N=151936), with **fully
   independent vocab tiles** (no reduction across N) → split N across MX and SIMT directly, and
   overlap the final softmax/argmax (SIMT) with the last MX tiles. In the decode regime (esp.
   DeepSeek long-CoT) this GEMV plus every layer's GEMV are **weight-BW-bound**, so the same
   independence is used to overlap layer N+1's weight-DMA with layer N compute.

---

## 6. Surprising / large-scale independence worth flagging

- **SmolVLA multi-camera vision towers [S1]** are *embarrassingly parallel* (identical read-only
  tower, no cross-view dep until concat). If deployment uses 2–3 cameras this is the single largest
  clean parallel region in any priority model — near-linear across engines. Verify camera count.
- **SmolVLA vision tower LayerNorm+GELU+bias gives the best SIMT/MX balance of all four models**
  (RMSNorm models starve SIMT; SigLIP's mean-subtract + erf-GELU + bias roughly doubles the SIMT
  epilogue work), so vision-tower [S2] overlap yields higher both-active% per GEMM than any LLM
  block.
- **The FLOP asymmetry is the load-bearing caveat for the whole effort:** elementwise is <1% of
  model FLOPs, so `GEMM ‖ single-fused-activation` leaves SIMT ~99% idle. High both-engines-active%
  requires the **S²-scaling** (attention softmax) or **large-N** (LM-head) SIMT regions, or handing
  SIMT whole small GEMMs (K/V proj, an N-slice) — *not* per-op activation fusion alone. Co-schedulers
  should target T1/T4 for both-active%, and use T2/T3/T5 mainly to eliminate bubbles/DRAM round-trips.
- **No cross-layer or attn‖FFN parallelism exists** in any of the four (all serial transformers; no
  MoE, no PaLM-style parallel block). Cross-layer overlap is limited to **weight-prefetch/DMA** [T7],
  not concurrent compute. Gemma2's sliding-window *alternation* is likewise sequential.
