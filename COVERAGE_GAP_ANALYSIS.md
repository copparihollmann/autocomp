# Coverage-gap analysis: Gemma-2-2B & DeepSeek-R1-Distill-Qwen-1.5B vs the TinyLlama-complete Radiance kernel set

**Question answered:** for every op in every layer (embed / attention / FFN / final / head), do we have a
*correct* kernel? This is **COVERAGE, not shape-tuning** — "correct kernel exists, yes/no", ignoring whether
the GEMM tiling is optimal.

**Method (ground truth, not memory):** parsed the real op graphs and weight manifests captured from the
actual trained models:
- `oscar-merlin/out/artifacts/recaptures/gemma2_2b_gguf_q8/{model.mlir, weights.safetensors.manifest.json, extra.npz}`
- `oscar-merlin/out/artifacts/recaptures/deepseek_qwen_1_5b_gguf_q8/{model.mlir, weights.safetensors.manifest.json, extra.npz}`
- (HF cache `/scratch/agustin/cache/huggingface/hub/models--{google--gemma-2-2b-it, deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B}`
  holds **refs only, no blobs** — configs not on disk; architecture constants below are cross-checked against
  the known HF `Gemma2ForCausalLM` / `Qwen2ForCausalLM` configs and confirmed against the captured graph.)

**Baseline we already have (RTL-correct, TinyLlama = plain Llama-2):** embedding gather, RMSNorm (`autocomp_rms_t27`),
RoPE (test26), GQA flash-attention causal (`autocomp_fa_gqa_*`), online-softmax (test5), SwiGLU/SiLU
(`autocomp_*swiglu*`), residual-add (test28), MX-GEMM fp4/fp6/fp8 for all projections + LM-head, SIMT
activation-quantize. New elementwise/mask ops are SIMT (bf16/fp32) → **precision-agnostic** (fp4/6/8 GEMM
paths already exist), so data-type variety adds no coverage ops.

Confirmed capture facts used below: **seq len = 8** for both goldens (`inputs.npz in0 = (1,8) int64`).

---

## Model A — DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch)

Confirmed from manifest + graph: hidden 1536, FFN 8960, 28 layers, **12 Q / 2 KV heads, head_dim 128**,
vocab 151936. **2 RMSNorms/layer** (`input_layernorm` + `post_attention_layernorm`; graph has 57 rsqrt =
28×2+1). Activation = **SwiGLU/SiLU** (graph: 28 `aten.sigmoid`, 0 gelu/tanh/erf). RoPE present (shared
sin/cos table, `neg` = rotate_half). Causal mask only (select/where/cumsum/arange). No softcap. **No QK-norm**
(no `q_norm`/`k_norm` tensors — that is Qwen3, not Qwen2-1.5B). Embeddings **tied** (`lm_head` materialized).

**The one distinctive op: QKV bias.** Manifest confirms `self_attn.{q,k,v}_proj.bias` present
(`q [1536], k [256], v [256]`); **`o_proj` has NO bias**; MLP has no bias. Graph shows this as `aten.addmm`
(fused matmul+bias) on the QKV projections.

| Op (per layer + global) | Covered? | Kernel / NEW | Effort | Wrinkle |
|---|---|---|---|---|
| Embedding gather | ✅ | gather | — | tied embeddings — just a GEMM at head |
| RMSNorm ×2/layer + final | ✅ | `autocomp_rms_t27` | — | plain RMSNorm (no 1+w) |
| Q/K/V proj GEMM | ✅ | MX-GEMM | — | — |
| **QKV bias-add** | ❌ **NEW** | bias-add epilogue | **trivial** | fold `+bias[n]` into GEMM move-out; **QKV only, not O** |
| RoPE | ✅ | test26 | — | head_dim 128 |
| O proj GEMM | ✅ | MX-GEMM | — | no bias |
| QKᵀ / softmax / P·V (GQA flash, causal) | ✅ | `autocomp_fa_gqa_*` + test5 | — | 12:2 GQA ratio; standard 1/√128 scale |
| Residual add ×2 | ✅ | test28 | — | — |
| FFN gate/up/down GEMM | ✅ | MX-GEMM | — | — |
| FFN activation SwiGLU/SiLU | ✅ | `autocomp_*swiglu*` | — | — |
| LM head GEMM | ✅ | MX-GEMM | — | N=151936 |
| Activation→fp8 requant | ✅ | SIMT quantize | — | HW requant broken → SIMT (known) |

**Bottom line — DeepSeek needs exactly 1 new kernel: the QKV bias-add epilogue (trivial).** Nothing else is
new. It is one trivial epilogue away from coverage-complete.

---

## Model B — Gemma-2-2B (Gemma2 arch)

Confirmed from manifest + graph: hidden 2304, FFN 9216, 26 layers, **8 Q / 4 KV heads, head_dim 256**,
vocab 256000. **NO biases anywhere** (no `.bias` tensors). Embeddings **tied**. `extra.npz` gives
**`embed_scale = 48.0` = √2304** (embed ×√d confirmed).

**Norms — 4 per layer confirmed** (manifest has `input_layernorm`, `post_attention_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`) + 1 final = **105 RMSNorm** (graph: 105 rsqrt/pow).
Norm is the **(1+w)** Gemma variant — the norm region has an extra `addf` (add_6) after the weight-mul,
i.e. `x̂·(1+w)`.

**FFN activation = GELU gate (GeGLU)** — 26 `aten.gelu`. **Graph uses EXACT erf GELU** (`math.erf`,
constant `0.707106769 = 1/√2` ⇒ `0.5·x·(1+erf(x/√2))`). See Surprise #1.

**Final logit softcap = present and confirmed:** graph has exactly **1 tanh** on the `1×8×256000` logits,
region `div_0 → tanh → mul(30.0)` ⇒ **`30·tanh(x/30)`**, cap = 30.

**Attn logit softcap (cap 50) — architecturally in the config but ABSENT from this capture.** Only 1 tanh in
the whole graph (the final one). See Surprise #2.

**Sliding-window attention — architecturally present (window 4096, alternating layers) but INACTIVE at the
captured seq=8** (8 ≪ 4096 ⇒ mask ≡ plain causal). See Surprise #3.

**Query pre-attn scalar — NO-OP for gemma-2-2b** (`query_pre_attn_scalar = 256 = head_dim` ⇒ scale = 1/√256,
identical to the standard attention scale). See Surprise #4.

| Op (per layer + global) | Covered? | Kernel / NEW | Effort | Wrinkle |
|---|---|---|---|---|
| Embedding gather | ✅ | gather | — | — |
| **Embed ×√d (×48)** | 🔸 NEW | scalar-mul epilogue on gather | **trivial** | constant 48.0 from `extra.npz` |
| **RMSNorm (1+w), ×4/layer** | 🔸 variant | `autocomp_rms_t27` + `(1+w)` | **trivial** | 1-line: multiply by `(1+w)` not `w`; schedule 4/layer |
| Q/K/V proj GEMM | ✅ | MX-GEMM | — | no bias |
| RoPE | ✅ | test26 | — | head_dim 256 |
| **Query pre-attn scale** | ✅ | (folded scale) | — | **no-op**: equals 1/√256 already |
| QKᵀ / softmax / P·V (GQA flash, causal) | ✅ | `autocomp_fa_gqa_*` + test5 | — | 8:4 GQA |
| **Attn logit softcap `50·tanh(·/50)`** | ❌ NEW (architectural) | softcap folded INTO flash online-softmax | **moderate** | **the real wrinkle** — see below; NOT in this golden |
| **Sliding-window causal mask (w=4096, alt layers)** | ❌ NEW (architectural) | windowed-causal mask variant | **moderate** | inactive ≤4096; alt-layer pattern; NOT in this golden |
| O proj GEMM | ✅ | MX-GEMM | — | O out-dim 2048 (8×256) |
| Residual add ×2 | ✅ | test28 | — | — |
| FFN gate/up/down GEMM | ✅ | MX-GEMM | — | FFN 9216 (largest) |
| **FFN activation GELU (GeGLU)** | 🔸 BUILT | `autocomp_gelu` (parked) | **done*** | *kernel is tanh-approx; graph is erf — Surprise #1 |
| **Final logit softcap `30·tanh(·/30)`** | ❌ NEW | SIMT epilogue on LM-head | **small** | reuse `tanh_mu` already in `autocomp_gelu`; cap 30 |
| LM head GEMM | ✅ | MX-GEMM | — | N=256000 |
| Activation→fp8 requant | ✅ | SIMT quantize | — | known |

### The attn-softcap wrinkle (why it is not a plain epilogue)
Gemma2 applies `S = 50·tanh(QKᵀ·scale / 50)` to the attention logits **before** softmax. In a flash /
online-softmax kernel there is no materialized full `S` row to post-process — you consume each KV block's
partial scores while tracking a running max `m` and running denom `l`. The softcap must be applied to each
**partial** score tile **before** it enters the running-max/exp recurrence: for each KV block compute
`s = 50·tanh((qkᵀ·scale)/50)`, then update `m' = max(m, rowmax(s))`, rescale the accumulator by
`exp(m−m')`, and add `exp(s−m')`. Applying it after the running-max (or once at the end) is numerically wrong
because the cap changes which element is the max. So it is a per-tile transform inside the flash loop, not a
detachable epilogue. (Sliding-window is the easier sibling: mask each KV block by `|i−j| < W` alongside the
causal mask inside the same loop.)

### Minimum-effort ranked list — Gemma-2
1. **RMSNorm (1+w)** — trivial 1-line variant of `autocomp_rms_t27`; schedule 4/layer. (variant)
2. **Embed ×√48** — trivial scalar-mul on gather output. (variant)
3. **GeGLU** — kernel **already built** (`autocomp_gelu`); pick erf vs tanh (Surprise #1), wire in. (built)
4. **Final logit softcap** `30·tanh(x/30)` — **small**, genuinely new SIMT LM-head epilogue; `tanh_mu`
   already exists in the gelu kernel. This is the ONE new kernel needed to pass **this** golden.
5. **Attn logit softcap** `50·tanh(·/50)` — **moderate**, folded into flash online-softmax (wrinkle above).
   Needed for architectural correctness at long seq / eager attention; **not exercised by the seq-8 golden**.
6. **Sliding-window mask** (w=4096, alternating layers) — **moderate**, windowed-causal variant in the flash
   loop. Needed only for seq > 4096; **not exercised by the seq-8 golden**.

**Bottom line — Gemma-2:**
- To pass **this recaptured golden (seq 8, sdpa, no attn-softcap):** **1 genuinely new SIMT kernel** (final
  logit softcap) + 2 trivial variants (RMSNorm 1+w, embed×48) + 1 already-built (GeGLU). ≈ **2 new kernels.**
- For **full architectural correctness** (long seq / eager attention, sliding window active): **+2 moderate**
  attention-integration kernels (attn softcap folded into flash-softmax; windowed-causal mask). ≈ **4 new
  kernels total.**

---

## Cross-model ranking (by total coverage effort)

| Rank | Model | New kernels for e2e | Total effort |
|---|---|---|---|
| 1 (easiest) | **DeepSeek-R1-Qwen-1.5B** | **1** — QKV bias-add epilogue (trivial) | **trivial** — essentially coverage-complete |
| 2 | **Gemma-2-2B** | **2** to pass the golden (final softcap NEW + RMSNorm(1+w)/embed×48 variants; GeGLU built); **4** for full long-seq correctness (+ attn-softcap-in-flash, + sliding-window mask) | **moderate** — 1 small + 2 moderate genuinely new, rest trivial |

**DeepSeek ≪ Gemma-2.** DeepSeek is one trivial bias-add away from done. Gemma-2's real work is the softcap
family and the two attention-integration pieces (attn softcap folded into online-softmax; windowed mask),
which are the only non-trivial builds across both models.

---

## Surprises (things neither the memory nor E2E_MODEL_COVERAGE.md anticipated)

1. **GELU form mismatch (erf vs tanh).** The parked kernel `autocomp_gelu` implements
   `gelu_pytorch_tanh` (the tanh approximation — which *does* match the real HF Gemma2
   `hidden_activation="gelu_pytorch_tanh"`), **but the recaptured golden graph uses EXACT erf GELU**
   (`math.erf`, `1/√2`). So the built kernel and the golden disagree by ~1e-3. Both are within fp8/bf16
   tolerance, so GELU is effectively covered — but **decide the reference**: validate against the real HF
   model → keep tanh; validate against THIS golden.npy → switch to erf. Neither doc flagged that the
   recapture pipeline emitted a different GELU than the model config specifies.

2. **Attn logit softcap is ABSENT from the captured Gemma2 graph.** Only the *final* softcap (1 tanh on the
   256000-wide logits) is present; the per-layer attn softcap (cap 50) never appears — the recapture used an
   attention impl (sdpa) that drops attn softcapping, and seq=8 makes it moot anyway. **Trap: passing this
   golden does NOT prove architectural completeness.** Both docs list "soft-cap into flash-attention" as a
   gap — correct in principle, but you can (misleadingly) pass the provided golden *without* it. The real
   model at long seq with eager attention needs it.

3. **Sliding-window attention is inactive at the captured seq.** Window is 4096, seq is 8, so the mask ≡
   plain causal in the golden. The kernel is architecturally required (alternating layers) but the golden
   cannot exercise or validate it.

4. **Query pre-attn scalar is a NO-OP for gemma-2-2b.** `query_pre_attn_scalar = 256 = head_dim`, so the Q
   pre-scale equals the standard `1/√256` attention scale — the E2E doc lists "Q pre-scale" as a distinctive
   op, but it costs nothing here (it only differs from `1/√head_dim` in larger Gemma2 variants like 27B).

5. **DeepSeek bias is QKV-only.** `o_proj` has no bias and MLP has no bias (manifest-confirmed) — the
   bias-add epilogue attaches to the three QKV projections only, not to O or FFN. Also both models **tie
   embeddings** (lm_head materialized but shared with embed) — no coverage impact, just a GEMM at the head.

---

## Per-model NEW-op lists (the return payload)

**DeepSeek-R1-Distill-Qwen-1.5B — NEW ops:**
- QKV bias-add epilogue (`+bias[n]` on q/k/v proj move-out; NOT o_proj). *trivial.*
- → **1 new kernel. Otherwise coverage-complete.**

**Gemma-2-2B — NEW ops:**
- Final logit softcap `30·tanh(x/30)` — SIMT LM-head epilogue (reuse `tanh_mu`). *small — new.*
- RMSNorm (1+w) — 1-line variant of `autocomp_rms_t27`, 4/layer. *trivial — variant.*
- Embed ×√48 — scalar-mul on gather. *trivial — variant.*
- GeGLU (GELU gate) — `autocomp_gelu` **already built** (resolve erf-vs-tanh). *done.*
- Attn logit softcap `50·tanh(·/50)` folded into flash online-softmax — *moderate, architectural, not in golden.*
- Sliding-window causal mask (w=4096, alt layers) — *moderate, architectural, not in golden.*
- → **2 new kernels to pass this golden; 4 for full long-seq correctness.**
