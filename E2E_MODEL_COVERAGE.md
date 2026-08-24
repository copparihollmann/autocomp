# E2E model coverage on Radiance — the 4 priority workloads

Goal: run **SmolVLA, Gemma-2-2b, DeepSeek-R1-Distill-Qwen-1.5B, TinyLlama-1.1B** end-to-end on Radiance —
which requires a **RTL-correct kernel for EVERY op in EVERY layer** (the e2e claim), then optimize (the
megakernel). Dims below are authoritative (parsed from the weight manifests / MLIR graphs, not guessed).

## 1. Real model dimensions (authoritative)

| Model | hidden d | FFN | layers | Q heads | KV heads | head_dim | vocab | FFN act | norm | distinctive ops |
|---|---|---|---|---|---|---|---|---|---|---|
| **TinyLlama-1.1B** | 2048 | 5632 | 22 | 32 | 4 | 64 | 32000 | SwiGLU (SiLU) | RMSNorm | — (baseline Llama-2) |
| **Gemma-2-2b** | 2304 | 9216 | 26 | 8 | 4 | 256 | 256000 | **GeGLU (GELU)** | RMSNorm **(1+w), pre+post** | **logit soft-cap (tanh)**, **sliding-window (alt layers)**, embed ×√d, Q pre-scale |
| **DeepSeek-R1-Qwen-1.5B** | 1536 | 8960 | 28 | 12 | 2 | 128 | 151936 | SwiGLU (SiLU) | RMSNorm | **QKV bias** (Qwen2); reasoning → long decode |
| **SmolVLA** (VLM text) | 960 | 2560 | ~32 | 15 | 5 | 64 | 49280 | SwiGLU | RMSNorm | + **SigLIP vision** (768/3072, **LayerNorm+GELU+bias**), **action expert** (flow-matching, cross-attn) |

Per-layer matmul shapes (M=seq S; fp8 1B/elem). Q/K/V widths = heads×head_dim. FFN gate/up/down dominate.
- TinyLlama: QKV = S×2560×2048 (fused), O = S×2048×2048, FFN = S×5632×2048 ×2 + S×2048×5632.
- Gemma2: QKV = S×4096×2304, O = S×2048×2304, FFN = S×9216×2304 ×2 + S×2304×9216 (biggest).
- DeepSeek: QKV = S×2048×1536, O = S×1536×1536, FFN = S×8960×1536 ×2 + S×1536×8960.
- SmolVLA text: QKV = S×1600×960, FFN = S×2560×960 ×2; vision: S×2304×768, S×3072×768.

**All K-depths are 960–9216 — 1–2 orders of magnitude beyond the 128–1024 the current kernels test.**

## 2. E2E op coverage matrix (the claim-backing deliverable)

Legend: ✅ have RTL-correct kernel · 🔸 have variant, needs small adaptation · ❌ GAP (build needed).

| Op | TinyLlama | DeepSeek | Gemma2 | SmolVLA | Kernel status |
|---|---|---|---|---|---|
| Embedding lookup | ✅ | ✅ | ✅ (×√d scale) | ✅ | gather; Gemma ×√d = **test41** ✅ cyc |
| RMSNorm | ✅ | ✅ | ✅ (1+w, 4-norm) | ✅ | test27; Gemma (1+w)=**test40**, 4-norm wiring=**test42** ✅ cyc |
| **LayerNorm** (mean-sub + bias) | — | — | — | ✅ | SmolVLA vision **CLOSED** — `autocomp_layernorm_vision` (D=768, cyclotron tohost=0, §8) |
| Q/K/V proj (GEMM) | ✅ | ✅ +bias | ✅ | ✅ +bias | MX GEMM; **QKV bias-add** ✅ (autocomp_qkv_bias_qwen); **vision proj/MLP bias-add** ✅ (autocomp_bias_add_vision, tohost=0, §8) |
| RoPE | ✅ | ✅ | ✅ | ✅ | test26 |
| Attention Q·Kᵀ / P·V | ✅ | ✅ | ✅ +softcap +window | 🔸 +bias, cross-attn | MX/flash; **attn soft-cap folded in online-softmax + sliding-window** ✅ (`gemma_attn_online` tohost=0; wired into `autocomp_fa_gemma`) |
| Softmax | ✅ | ✅ | ✅ (attn softcap-in-flash) | ✅ | test5; Gemma online-softmax softcap `gemma_attn_online` tohost=0 |
| O proj | ✅ | ✅ | ✅ | ✅ | MX GEMM |
| Residual add | ✅ | ✅ | ✅ | ✅ | test28 (fuse to clear l0d) |
| FFN gate/up/down (GEMM) | ✅ | ✅ | ✅ | ✅ | MX GEMM |
| **FFN activation** | ✅ SwiGLU | ✅ SwiGLU | ✅ **GeGLU (gelu_pytorch_tanh)** | ✅ SwiGLU + GELU (vision) | GeGLU ✅ (`autocomp_gelu`); vision plain-GELU ✅ (`autocomp_gelu_vision`, tanh, tohost=0, §8) |
| **Logit soft-cap** (tanh) | — | — | ✅ | — | ✅ attn cap=50 folded in flash online-softmax (`gemma_attn_online`) + final cap=30 epilogue (`gemma_final_softcap`), both cyclotron tohost=0 |
| LM head (GEMM) | ✅ | ✅ | ✅ | ✅ | MX GEMM (big N=vocab) |
| **Vision patch/conv embed** | — | — | — | ✅ | SmolVLA SigLIP **CLOSED** — `autocomp_patch_embed_vision` (patchify+GEMM+bias+pos, K=768, tohost=0, §8) |
| **Attention pooling / cross-attn** | — | — | — | ✅ | pooling **ABSENT** in SmolVLM SigLIP (not a gap); cross-attn = attention subset (external K/V), §8 |
| **Flow-matching time-MLP** | — | — | — | 🔸 | SmolVLA action expert (SiLU MLP + sin/cos ✅) |
| Requantize (activation→fp8) | 🔸 | 🔸 | 🔸 | 🔸 | HW requant BROKEN → **SIMT quantize** (see REQUANT_INVESTIGATION.md) |

### The coverage GAPS to close for e2e (ranked)
1. ~~**GELU/GeGLU** — Gemma2 GeGLU FFN + SmolVLA vision MLP.~~ ✅ **CLOSED** (`autocomp_gelu`, cyclotron tohost=0 @ Gemma FFN N=9216, M=4). Real Gemma-2 act = **gelu_pytorch_tanh** (the TANH approx, NOT erf — confirmed via llama.cpp GEGLU map + HF Gemma2Config default; the recapture golden.npy uses erf, ~1e-3 off — do NOT validate against it). Kernel does GeGLU `gelu_tanh(gate)*up` as a SIMT epilogue (activation fp32; gate/up/down GEMMs stay fp4/6/8), using the algebraic sigmoid form `x/(1+exp(-2·√(2/π)·(x+0.044715x³)))` to stay under the 256-reg Rename wall. Plain GELU (vision) = the B=1 subset.
2. ~~**LayerNorm** (mean-subtract + γ/β bias) — SmolVLA vision.~~ ✅ **CLOSED** (`autocomp_layernorm_vision`, D=768, cyclotron tohost=0, §8).
3. ~~**Bias-add epilogue** — DeepSeek QKV bias, SmolVLA vision.~~ ✅ **CLOSED** (DeepSeek `autocomp_qkv_bias_qwen`; SmolVLA vision `autocomp_bias_add_vision`, tohost=0, §8).
4. ~~**Logit soft-capping** (`c·tanh(x/c)`) — Gemma2 attention scores + final logits.~~ ✅ **CLOSED** (see §2c). attn cap=50 folded into the flash online-softmax; final cap=30 SIMT epilogue. Both cyclotron tohost=0.
5. ~~**Sliding-window attention mask** — Gemma2 alternating layers.~~ ✅ **CLOSED** (see §2c). Windowed-causal mask in the flash online-softmax, cyclotron tohost=0.
6. ~~**RMSNorm (1+w) + pre/post placement** — Gemma2.~~ ✅ **CLOSED** (test40/test42, see §2b).
7. ~~**GeGLU gating** — Gemma2 (SwiGLU with GELU).~~ ✅ **CLOSED** — same `autocomp_gelu` kernel is the GeGLU gate (`gelu_tanh(gate)*up`); see item 1.
8. ~~**SmolVLA vision front-end** — patch/conv embedding, attention pooling, cross-attention VLM↔expert.~~ ✅ **CLOSED for coverage** (§8): LayerNorm + patch/conv-embed + bias-add + GELU all cyclotron tohost=0; **attention pooling ABSENT** in SmolVLM SigLIP (not a gap); cross-attn = attention subset (external K/V, no causal).
9. **SIMT activation-quantize** — the HW-requant workaround, needed wherever fp8 chaining is used.

Everything else (all GEMMs, RMSNorm, RoPE, softmax, SwiGLU, residual) is already RTL-correct.

## 2b. Gemma-2-2b NORM/EMBED coverage — CLOSED (cyclotron tohost=0)
All variants confirmed against the real `config.json`
(`oscar-merlin/out/artifacts/cache/models/gemma2_2b/base/config.json`) + HF `modeling_gemma2.py`.
Coverage bar = cyclotron `verify_body` tohost=0 (PASS). Precision-light fp32 SIMT (RMSNorm-class).

| Variant | Confirmed fact (config + HF source) | Test | Kernel | Cyclotron |
|---|---|---|---|---|
| **RMSNorm (1+w)** | `Gemma2RMSNorm.forward`: `rms(x.float()) * (1.0 + weight.float())`, weight stored **zero-centered**, `eps=1e-6`. vs Llama (weight direct, eps 1e-5). | test40 | sol40_baseline | **PASS** (tohost=0) |
| **Embedding ×√d** | `Gemma2TextScaledWordEmbedding`: `embed_scale = hidden_size**0.5`; out = `table[ids]*embed_scale`. √2304 = **48.0 exact** (bf16-safe). Tied to LM head (`_tied_weights_keys`, no separate `lm_head.weight` in manifest). | test41 | sol41_baseline | **PASS** (tohost=0) |
| **4-norm wiring** | `Gemma2DecoderLayer.forward` "sandwich" — see ordering below. | test42 | sol42_baseline | **PASS** (tohost=0) |

**Confirmed 4-norm residual ordering** (`Gemma2DecoderLayer.forward`, verbatim topology):
```
residual = h
h = input_layernorm(h)            # NORM 1  pre-attention
h = self_attn(h)
h = post_attention_layernorm(h)   # NORM 2  post-attention
h = residual + h                  # RESIDUAL 1  (bypasses BOTH norm 1 & 2)
residual = h
h = pre_feedforward_layernorm(h)  # NORM 3  pre-FFN
h = mlp(h)
h = post_feedforward_layernorm(h) # NORM 4  post-FFN
h = residual + h                  # RESIDUAL 2  (bypasses BOTH norm 3 & 4)
```
i.e. a norm BEFORE and AFTER each sublayer (attn, FFN); each residual carries the *pre-norm* value
across the whole sublayer. Confirmed 4 weight tensors/layer in the manifest: `input_layernorm`,
`post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`. test42
composes 4× RMSNorm(1+w) + 2× residual in this exact order (attn/mlp stubbed as deterministic
per-channel scales — those ops are covered separately; the norm/residual TOPOLOGY is what is verified).
Discrimination check: a Llama-style pre-norm-only wiring diverges on 98% of elements (max |Δ|=1.5 ≫ tol),
so the PASS is non-trivial. Dims used: real Gemma hidden D=2304.
Note: embeddings are TIED to the LM head (affects nothing for op coverage; flagged per request).

## 2c. Gemma-2-2b ATTENTION coverage — CLOSED (cyclotron tohost=0 / in-band rel-err)

All four Gemma-2 attention deltas vs plain GQA flash-attention (TinyLlama), each CONFIRMED from the
real `google/gemma-2-2b-it` config.json and validated against HAND-BUILT numpy goldens (the oscar-merlin
recapture is seq=8 and exercises NEITHER attn-softcap saturation NOR the sliding window, so it proves
nothing for these — a dedicated wide-score / small-window golden is used instead):

| Gemma-2 attention op | config.json | status | proof |
|---|---|---|---|
| **attn_logit_softcapping** | `50.0` | ✅ | `gemma_attn_online` (cap=50, window=0): tohost=0. Cap folded into the flash online-softmax BEFORE the running-max/exp. |
| **sliding_window** (alt layers) | `4096` (even layers) | ✅ | `gemma_attn_online` (cap=50, window∈{8,32}): tohost=0. Windowed-causal mask. |
| **final_logit_softcapping** | `30.0` | ✅ | `gemma_final_softcap` SIMT epilogue `30·tanh(x/30)`, 64×512, 51% of inputs past the cap: tohost=0. |
| **query_pre_attn_scalar** | `256` = head_dim | — (no-op) | equals `1/√head_dim` already applied as softmax_scale → NO separate op (would differ only for 9b/27b). |
| head_dim / GQA | `256`, 8Q/4KV | ✅ | grp=2 GQA path exercised in `autocomp_fa_gemma`; softcap/window logic is head_dim-independent (verified at layout-safe d=64). |

THE WRINKLE (attn soft-cap in the flash online-softmax). Non-flash Gemma applies `s → 50·tanh(s/50)` to the
FULL score matrix before the softmax. In the streaming/flash form the cap must be applied to EACH partial
score tile Sj right after the softmax-scale and **BEFORE** the per-block max, the running max `m`, and every
`exp` — NOT as a post-softmax epilogue. Reason: tanh is a nonlinear monotonic map; the running max carried
across key blocks is a max of already-capped scores, and `corr = exp(m_old−m_new)` rescales the accumulator
on that capped scale. Capping only the exp argument while taking the running max on UN-capped scores mixes two
score scales in the cross-block `max(m_run, bmax)` compare, and in bf16 makes `exp(s_capped − m_uncapped)`
underflow every far-from-max term (m_uncapped can be ≫ 50 while s_capped ≤ 50) → the denominator `l` collapses
→ wrong probabilities. Folding the cap up front makes every max/exp/corr live on one consistent capped scale,
so the standard flash recurrence is reused verbatim. Verified: fp32 streaming (cap-before-max) == fp32 dense
softcap softmax to 1e-6; the poly-arithmetic dense golden matches TRUE fp64 softcap+window softmax to 0.000%.

SLIDING-WINDOW MASK. A sliding layer masks an entry when `key_pos > q_pos` (causal high side) OR
`key_pos ≤ q_pos − window` (too-old low side); the low bound is written `key_pos + window ≤ q_pos` to avoid
unsigned underflow. Full-causal (odd) layers pass window=0 (low side disabled). Confirmed active: window=32
changes the attention output by 109% vs full-causal; the cap changes it by 21.5% vs no-cap (neither is a no-op).

Kernels (in `radiance-kernels/kernels/`):
- `gemma_final_softcap/` — SIMT `cap·tanh(x/cap)` (fp32 exp-based tanh, no libm), the shared soft-cap
  primitive; self-checking tohost=0.
- `gemma_attn_online/` — mesh-free SIMT kernel running the EXACT streaming online-softmax recurrence of
  `flash_mx_impl.hpp::fused_softmax_requant` (softcap-before-max + windowed mask), self-checked vs a DENSE
  golden via Frobenius rel-err (≤2%, the fp32-attention bar; the fold LOGIC is certified exact off-device,
  the residual is only muon-fp32-vs-numpy-fp32 FMA rounding). tohost=0 at {cap50/win0, cap50/win8,
  cap50/win32, cap0/win32}.
- `autocomp_fa_gemma/` — the INTEGRATED MX flash-attention kernel (fork of `autocomp_fa_gqa_causal`): the
  bf16 `bf16_softcap` primitive + soft-cap fold + windowed mask wired into `fused_softmax_requant` (compile
  flags `FA_ATTN_SOFTCAP`/`FA_SLIDING`/`FA_WINDOW` from the data header) + a safe-`l` guard in `finalize_O`.
  Builds and runs to completion on cyclotron (8Q/4KV, softcap+window active, 863k cycles, no assert/panic).

PLATFORM NOTE: the full mesh-attention output O is NOT FP-verifiable on cyclotron — the MX-mesh PV co-model
emits NaN for these flash kernels, and the PRISTINE upstream `autocomp_fa_gqa_causal` shows the identical
all-NaN O on cyclotron-direct (dmem trace), so this is a platform limitation, not a kernel bug. Hence the
soft-cap-fold + window LOGIC (all SIMT, in `fused_softmax_requant`) is proven tohost=0 by the mesh-free
`gemma_attn_online`; O itself is FP-verified through the VCS/RTL soc.elf host flow (subject to the known
softmax write-drain race, memory `muon-rtl-writedrain-bug`).

## 3. Memory-hierarchy reality (drives the megakernel)
- **Weights do NOT fit on-chip** (fp8): smallest FFN weight (SmolVLA gate 2560×960) = 2.4 MB; largest
  (Gemma2 gate 9216×2304) = 21 MB; per-layer total 4–42 MB. 128 KB SMEM / 512 KB L2 → **weights stream
  from DRAM, re-read every layer** = the dominant DRAM traffic.
- **Activations DO fit**: residual [T tokens, d] fp8 = T×(960..2304) B. T=16 ≈ 15–37 KB (SMEM); T=64 ≈
  60–147 KB (L2). Attention scores S×S fit SMEM only for S≲256 → flash-attn tiling mandatory at real seq.
- Regimes: **prefill / VLA-vision = compute-bound** (weights reused across S/T tokens) → overlap+fusion
  win; **decode (esp. DeepSeek-R1 long CoT) = weight-BW-bound GEMV** → weight-DMA/compute-overlap win.

## 4. The megakernel (layer-fused, weight-streaming, activation-resident)
One kernel per transformer layer that:
1. keeps the **residual stream + intermediates SMEM/L2-resident** across all ops (kills the ~14 between-op
   DRAM round-trips/layer, and the l0d streaming assert with it — no sustained GMEM stream);
2. **streams weights** DRAM→SMEM in tiles, double-buffered, hiding DMA behind MX compute;
3. **warp-specializes** so MX computes matmul N while SIMT runs the epilogue (norm/RoPE/act/residual/
   quantize) of matmul N−1 — realizing the measured ~95%-idle-SIMT headroom → the 1.0×→~2× overlap;
4. **fuses every elementwise op into a matmul epilogue/prologue** (RMSNorm, RoPE, GELU/SwiGLU, residual,
   bias, soft-cap, bf16→fp8 quantize) — SMEM-resident, hidden behind MX;
5. **flash-attention** block for scores (SMEM-resident per KV-block).
Prefill vs decode take the same skeleton with different tiling (T large vs M=1 GEMV + KV-cache).

## 5. Plan: coverage-first (e2e claim), then megakernel (perf)
**Phase E (coverage — unblocks the e2e claim):** build + RTL-verify the gap kernels §2.1–2.9, all at REAL
dims. Order by reuse/ROI: GELU → LayerNorm → bias-add → soft-cap → (1+w)RMSNorm/GeGLU → sliding-window →
SIMT-quantize → SmolVLA vision front-end. Each is the easy-correct SIMT/GEMM class; RTL-gate each.
**Phase M (megakernel — perf):** enabling build = **multi-output-tile MX GEMM** (config once, stream
weights, loop tiles) at real K; then the fused-layer megakernel §4 with warp-spec overlap; RTL-measure
vs the serial per-op baseline. Precision (fp4/fp6, proven correct) rides on top for a further up-to-2×.

Ground truth files: dims in `oscar-merlin/out/artifacts/recaptures/{gemma2_2b,deepseek_qwen_1_5b}_gguf_q8/
weights.safetensors.manifest.json` + `model2MLIR/workloads/{tiny_llama,smolvla}/*.safetensors.manifest.json`;
op graphs in the corresponding `*.mlir`. Model explorer: `model2MLIR` (`m2m convert <loader.py>`).

## 6. Coverage status log — DeepSeek-R1-Distill-Qwen-1.5B: COMPLETE (2026-07-17)

**Delta vs TinyLlama, confirmed from the weight manifest** (`oscar-merlin/out/artifacts/recaptures/
deepseek_qwen_1_5b_gguf_q8/weights.safetensors.manifest.json`, layer-0 + globals):
- `self_attn.{q,k,v}_proj.bias` present → shapes q[1536], k[256], v[256]. **The one new op: QKV bias-add.**
- `self_attn.o_proj` → weight only, **NO o-proj bias**.
- **NO `q_norm`/`k_norm`** tensors → Qwen2 (not Qwen3-style QK-norm).
- `input_layernorm.weight` + `post_attention_layernorm.weight` → RMSNorm, weight-only, standard pre-norm
  (same class as TinyLlama; not a new op).
- RoPE: `rotary_emb.inv_freq` present, head_dim=128 → **full RoPE** (rotary_dim == head_dim), not partial.
- `embed_tokens.weight` **and** `lm_head.weight` both present (not tied) → LM-head GEMM already covered; not new.
- Dims: hidden 1536, FFN 8960, 28 layers, GQA 12 Q : 2 KV, head_dim 128, vocab 151936, SwiGLU.

**So the entire new-op set for DeepSeek-Qwen vs TinyLlama = {QKV bias-add}.** Everything else (RMSNorm,
RoPE, GQA flash-attn, SwiGLU, residual, MX-GEMM fp4/fp6/fp8 for all projections + LM-head, embedding
gather) is already RTL-correct from the TinyLlama set; head_dim=128 / GQA 12:2 are parameters of those
same kernels, not new ops.

**Kernel built + proven:** `radiance-kernels/kernels/autocomp_qkv_bias_qwen/` (kernel.cpp + gen_data.py).
- SIMT bias-add epilogue: `out[t,c] = proj[t,c] + bias[c]`, bias broadcast over tokens.
- Real dims: M=64 tokens × N=2048 QKV cols = q(12×128) | k(2×128) | v(2×128); head_dim=128; K=1536 hidden.
- proj = **genuine fp8 MX move-out** (mx_golden fp8 e4m3 A/B + block scales), bf16; bias in **fp32**
  (GEMM stays fp8, bias precision-light). Golden = `bf16_trunc(bf16_to_f(proj) + bias)`.
- **cyclotron verify_body tohost=0 (Error: 0)** — CYCLOTRON_MXGEMMINI=1, config_muon.toml, --timing.

**Verdict: DeepSeek-R1-Distill-Qwen-1.5B is op-COVERAGE-complete = YES.** (Coverage bar = cyclotron
functional tohost=0; RTL cycle numbers are later-phase optimization, not required for the coverage claim.)
Update to §2 matrix: DeepSeek "Q/K/V proj + bias" row 🔸→✅ (bias epilogue built, autocomp_qkv_bias_qwen).

## 7. fp6-e3m2 production coverage (OPT-fp6-prod, 2026-07-18)

The fp4-coherence GO recipe = FFN fp4, **attention q/k/v/o fp6-e3m2**, **lm_head fp6-e3m2**, embeddings fp4.
Landed the fp6 kernels that path requires and mapped buildable-now vs blocked.

**Built + verified (cyclotron tohost=0 bit-exact + Verilator RTL):**
- `radiance-kernels/kernels/autocomp_gemm_fp6_k2048/` — **fp6 lm_head tile** (128×128×2048; the LM-head N=32000
  primitive unit). RTL net_kernel_cycles(core) = **149,301** (43.9% util). Logit accuracy fp6 7.66% vs fp8
  4.07% / fp4 16.28% → fp6 2.1× tighter than fp4.
- `radiance-kernels/kernels/autocomp_fa_qk_fp6/` — **fp6 attention QKᵀ scores block** (Sq64×Bk64×d64, static
  fp6 LUT operands). RTL net_kernel_cycles(core) = **68,324**. Attention accuracy fp6 10.55% vs fp8 5.33% /
  fp4 24.0% → fp6 2.3× tighter than fp4. (rel-err = the correctness gate for attention; mesh FP verify is
  platform-blocked — `fp6_attn_relerr.py`.)

**MXFP6 = 4-bit index into a per-row DATA-FITTED 16-entry LUT (not plain e3m2), baked at data-gen. So:**
- fp6 STATIC-weight matmuls buildable+verified NOW: lm_head, QKᵀ scores, O-proj. + fp4 FFN/embeddings (existing).
- ⛔ fp6 PV + any runtime fp6 activation BLOCKED: no on-device runtime per-row fp6-LUT activation quantizer
  (only `quantize_fp4` exists). Blocks the PV half of *fused* fp6 attention and live fp6 activation chaining.

**Recipe status: buildable on-device for lm_head (fp6) + attention QKᵀ scores (fp6) + O-proj (fp6, static W)
+ FFN (fp4); the single remaining enabler is the SIMT runtime fp6-LUT activation quantizer (→ PV + fused attn).**

## 8. Coverage status log — SmolVLA VISION (SigLIP) front-end: COMPLETE (2026-07-18)

**Model = SmolVLA (lerobot) = SmolVLM2-500M-Video-Instruct VLM (SigLIP vision + SmolLM2 text) + action
expert.** Confirmed from the real weight manifest (`model2MLIR/workloads/smolvla/smolvla.safetensors.
manifest.json`), the captured op graph (`smolvla.mlir`), and the HF config (`hf_cache/.../smolvla_base/
config.json`: `vlm_model_name = HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `attention_mode = cross_attn`,
16 VLM layers, `self_attn_every_n_layers=2`, `expert_width_multiplier=0.75`).

**SigLIP vision tower structure (manifest-authoritative), per encoder layer + globals:**
- `embeddings.patch_embedding`: **Conv2d(3→768, k=16, s=16) + bias[768]** (patchify). image 3×512×512 →
  32×32 = **1024 patch tokens × 768**. `embeddings.position_embedding` [1024,768] **added**.
- `encoder.layers.N`: **layer_norm1** (LayerNorm w+b) → **self_attn q/k/v/out_proj [768×768] + bias[768]**
  (12 heads, d=64, **bidirectional**, no causal) → residual → **layer_norm2** (LayerNorm w+b) → **mlp
  fc1[3072,768]+b → GELU → fc2[768,3072]+b** → residual.
- `post_layernorm` (LayerNorm w+b). Then `connector.modality_projection.proj` [960,12288] (pixel-shuffle
  reshape ×16 → linear to text hidden 960).
- Op-graph histogram: **575 native_layer_norm** (the dominant new op), 12 gelu (erf in capture / tanh in
  the real config — see below), 4 convolution (patch embed), 176 softmax (attn).

**Confirmed NEW-op list for the vision tower and each kernel (coverage bar = cyclotron two-line pass:
`simulation finished after N cycles` AND `isa-test passed with tohost=0`, vs a numpy fp32 golden):**

| Vision op | Confirmed fact | Kernel | Cyclotron (two-line, tohost=0) |
|---|---|---|---|
| **LayerNorm** (mean-sub + γ + β) | `layer_norm1/2` + `post_layernorm` carry `.weight` AND `.bias`; `native_layer_norm`, eps=**1e-6**. NOT RMSNorm (has mean-subtraction + β). | `autocomp_layernorm_vision` (D=768, M=32; single-pass Sx/Sxx stats + affine) | **PASS** 2,736,227 cyc |
| **Patch/conv embedding** | `patch_embedding.weight [768,3,16,16]` + `bias[768]`, stride=kernel=16 ⇒ non-overlapping ⇒ patchify(im2col)+GEMM+bias; `position_embedding [1024,768]` added. | `autocomp_patch_embed_vision` (16 patches, K=3·16·16=**768**, OC-tile 64; strided im2col + dot + bias + pos) | **PASS** 3,815,307 cyc |
| **Bias-add** (projections + MLP) | every vision linear has a bias (`q/k/v/out_proj.bias`, `fc1/fc2.bias`) — the text tower has none. | `autocomp_bias_add_vision` (M=32, N=768; `out=proj+bias[c]` broadcast) | **PASS** 2,637,350 cyc |
| **GELU** (MLP activation) | `mlp.activation_fn`; SmolVLM2 SigLIP config = `gelu_pytorch_tanh` (real model) — capture graph traced erf (~1e-3 off, within bf16 tol; same erf-vs-tanh quirk as Gemma-2 §2, validate vs the real model ⇒ tanh). | `autocomp_gelu_vision` (plain GELU, tanh, M=32,N=3072) + canonical `autocomp_gelu` | **PASS** 9,067,986 / 3,440,349 cyc |

**Expected ops that turned out NOT to be gaps (confirmed against the manifest):**
- **Attention pooling head — ABSENT.** SmolVLM's SigLIP has **no** `MultiheadAttentionPoolingHead`/probe/pool
  tensors (only the text `lm_head` exists); the encoder token outputs feed the connector directly. So the
  "attention pooling" gap listed in §2 does **not** exist for SmolVLA.
- **Cross-attention — a SUBSET of existing attention.** The action expert (`lm_expert`, `attention_mode=
  cross_attn`) cross-attends to the VLM prefix: Q from expert tokens, **K/V from the VLM KV-cache**. That is
  the same QKᵀ/softmax/PV as our flash-attention kernels with the K/V operand pointers swapped and the causal
  mask off (bidirectional) — no new arithmetic. Expert `self_attn q/k/v/o` are **weight-only** (no bias);
  its `input_layernorm/post_attention_layernorm/norm` are **RMSNorm** and its MLP is **SwiGLU** (Llama-class,
  covered). (Attention O itself is not cyclotron-FP-verifiable — the MX-mesh emits NaN for flash kernels, a
  platform limit §2c — so a two-line tohost proof is for the SIMT elementwise ops; attention is covered via
  the existing GQA/flash kernels + rel-err.)
- **Vision self-attention — covered by composition.** bidirectional MHA = QKᵀ/softmax/PV (have) + projection
  bias (built above) + no causal mask (strictly-easier subset of the causal kernels).
- **Connector `modality_projection`** = pixel-shuffle reshape + linear (12288→960 GEMM) — GEMM covered,
  reshape is addressing. **Action head** (`state_proj`, `action_in/out_proj`, `action_time_mlp_in/out`) =
  GEMM+bias + SiLU time-MLP + sin/cos — covered (§2 flow-matching 🔸 + bias-add).

**SmolVLA op-coverage: COMPLETE.** Every op in the SigLIP vision tower now has a correct kernel with a
cyclotron two-line tohost=0 proof — LayerNorm (the primary genuinely-new op), patch/conv embedding (+pos),
projection/MLP bias-add, and MLP GELU. The remaining vision pieces (bidirectional self-attn, cross-attn,
connector, action head) are subsets/compositions of already-RTL-correct ops (attention, GEMM, RMSNorm,
SwiGLU, bias); the expected "attention pooling head" is absent from this architecture. What remains for a
full SmolVLA e2e is not coverage but the same perf/integration work as the other models (megakernel, real-
seq attention, mesh-FP verify platform limit). Update §2 matrix: **LayerNorm** ❌→✅
(`autocomp_layernorm_vision`); **Vision patch/conv embed** ❌→✅ (`autocomp_patch_embed_vision`); **Attention
pooling / cross-attn** ❌→ pooling ABSENT + cross-attn = attention subset; SmolVLA **Q/K/V proj +bias** 🔸→✅
(`autocomp_bias_add_vision`); SmolVLA **FFN activation** 🔸→✅ (`autocomp_gelu_vision`).

Kernels (in `radiance-kernels/kernels/`): `autocomp_layernorm_vision/`, `autocomp_patch_embed_vision/`,
`autocomp_bias_add_vision/`, `autocomp_gelu_vision/` — each self-contained (gen_data.py + kernel.cpp +
Makefile), self-verifying vs a numpy fp32 golden. BUILD/RUN gotcha recorded: define `NUM_WARPS` **before**
`#include <mu_intrinsics.h>` (it provides its own default 8 via `#ifndef`; a define placed after the include
is silently ignored → occupancy 8 → register-heavy kernels trip the 256-phys-reg rename wall). All fp32
(norms/activations are precision-light; the GEMMs they wrap use the existing fp4/6/8 MX path).

## LEVER-94 — fp6 attention op-coverage CLOSED (fp6 PV runtime quantizer)
The mixed-precision recipe's **fp6 attention** path is now fully op-covered on-device. Previously the
runtime fp6-LUT activation quantizer (#86) covered fp6 QKᵀ (static Q/K) but **fp6 PV was BLOCKED** — P=softmax
is a runtime activation needing live fp6 quantization. Built `autocomp_fa_pv_fp6_live/`: P → SIMT runtime
fp6 quantizer (fitted LUT) → fp6 mxgemm(P, static V) → O. cyclotron two-line tohost=0 BIT-EXACT (531,309
cyc). LUT fitted to P's [0,1] range (greedy error-min cover) = 9.9× tighter P activation-quant than a
generic palette (7.39% vs 72.96%), near plain-grid fp6 (~5.2%). Attention-O ladder (fp6_attn_relerr.py):
fp8 5.33% < fp6 10.55% < fp4 24.0% → fp6 2.28× tighter than fp4. Chain: QKᵀ(static fp6) → softmax(bf16 SIMT)
→ PV(runtime fp6-quant P @ static fp6 V) → all fp6 on-device; + fp4 FFN. **RTL finding:** the runtime fp6
quantizer $stopped on the Muon IPDOM warp-reconvergence stack (WarpScheduler.sv:793) — fixed by making the
quantizer BRANCHLESS (dev_bf16_to_fp6_code + finder via mask-select, bit-identical 0/65536 + 0/200000),
which is why #86 was cyclotron-only; both fa_pv/fa_qk fp6-live kernels now retire on RTL.
