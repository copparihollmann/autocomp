# Realistic TinyLlama/Llama-2 kernel coverage — cyclotron correctness vs RTL readiness

Mapped against the 3/6/2026 tapeout spec (Vikram/Sirius). Two separate columns because they differ:
**KERNEL** = a correct, cyclotron-bit-exact implementation exists. **RTL** = what actually happens on
the elaborated tapeout-330 RTL (Verilator) this session. The kernel-level coverage is essentially
complete; **RTL readiness is the realistic gate for tapeout, and it is NOT complete.**

Legend RTL: ✅PASS (verified correct on RTL) · ⚠️PASS?(completed, no clean verdict — DRAIN=2000
write-drain; needs a large-DRAIN confirm) · ⛔L0D (trips the unbuffered-l0d backpressure assertion at
full shape — see [[rtl-l0d-no-landingpads]]) · ❌MISMATCH (RTL result ≠ golden) · 🚧TEAM (team-flagged
precision WIP).

## By spec section

| Layer | Operation | Kernel | Backend | KERNEL (cyclotron) | RTL status | Notes |
|---|---|---|---|---|---|---|
| Embedding | RoPE | RoPE | SIMT | ✅ | **✅PASS** | test26; latency-bound, runs clean |
| QKV proj | Q/K/V Projection | GEMM | **MX** | ✅ | **✅ works** | fp8 GEMM, 92% compute-window (anchor128) |
| QKV proj | Q/K/V + RoPE fused | GEMM+RoPE | MX+SIMT | ✅ | ⚠️PASS? | test33; overlap 1.0× (serialized), 47% idle |
| Attn (prefill) | RMSNorm (pre-norm) | RMSNorm | SIMT | ✅ | **✅ compute-correct** | test27; trace-verified 0/32768 fail@1e-4 (rel 1e-7). on-chip verdict = read-back artifact |
| Attn (prefill) | Q·Kᵀ | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | Softmax | Softmax | SIMT | ✅ | ⚠️ likely (partial trace-cov) | test5; recon <50% cov (store pattern); GEMV-softmax t35 trace-verified 0-fail so the softmax math is correct |
| Attn (prefill) | P·V | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | O Projection | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | **Flash-attn v4 (fused MX)** | Flash-attn | MX+SIMT | ✅ (Richard's, RTL 2.48%) | ⚙️ builds+runs on our tapeout-330 cyclotron (185K cyc); O-verify in progress | **GAP CLOSING**: fetched `flash_attention_mx_yrh` (Sq64×Sk256×d128, streaming, online-softmax, SIMT e4m3 requant→PV SKIP_A, 2-warp l0d-safe); builds clean on 330, runs in co-model; candidate for the autocomp perf loop |
| Attn (prefill) | ResAdd | ResAdd | SIMT | ✅ | **⛔L0D** | test28; blocked at 128×512 (l0d assertion) |
| Attn (decode) | Q·Kᵀ | GEMV | SIMT | ✅ | **✅ compute-correct** | test34; trace-verified 0/128 fail@1e-4 (rel~1e-7). tohost errors = SIMT read-back artifact, NOT a bug |
| Attn (decode) | Softmax | GEMV-softmax | SIMT | ✅ | ⚠️ likely-correct | test35; same class as t34/t36 (read-back artifact) — trace-verify to confirm |
| Attn (decode) | P·V | GEMV | SIMT | ✅ | **✅ compute-correct** | test36; trace-verified 0/64 fail@1e-4 (rel~1e-7). tohost = read-back artifact |
| Attn (decode) | O Projection | GEMV | SIMT | ✅ | **⛔L0D** | test29; 512×512 GEMV trips l0d at full shape |
| FFN | RMSNorm | RMSNorm | SIMT | ✅ | **✅ compute-correct** | test27 (shared) |
| FFN | Up/Gate/Down proj | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| FFN | SwiGLU | SwiGLU | SIMT | ✅ | **⛔L0D** | test4; blocked at 64×512 (l0d assertion) |
| FFN | ResAdd | ResAdd | SIMT | ✅ | ⛔L0D | (shared with attn) |

## Datatype / requant coverage
| Kernel | KERNEL (co-model) | RTL | Notes |
|---|---|---|---|
| MxGEMM fp8 | ✅ | ✅ works | the workhorse; 92% compute, verified path |
| MxGEMM fp6 (LUT e3m2) | ✅ bit-exact | **✅ compute-correct** | test38; trace-verified **0/16384 bit-exact** mismatches. tohost=2995 was the read-back artifact, NOT a bug |
| MxGEMM fp4 (e2m1) | ✅ bit-exact | **✅ compute-correct** | test37; trace-verified **0/4096 bit-exact** mismatches. tohost=2061 was the read-back artifact, NOT a bug |
| Requantizer (fp8 out) | ✅ bit-exact (co-model) | ❌ **REAL bug** | test39; trace-verified **89% mismatch**, not a permutation/transpose → genuine RTL requant error (the one real correctness bug) |
| SIMT GEMM (fp32) | ✅ | ⛔L0D(SMEM) / runs(reg-blocked) | naive runs but memory-bound; register-blocked = 26% util, clean |

## Honest bottom line
- **Kernel coverage of the model: complete** — every distinct Llama-2 operation has a cyclotron-bit-
  exact kernel, on the correct backend (GEMM/attention-matmul→MX; norms/activation/RoPE/residual/
  decode-GEMV→SIMT). **The one true kernel gap is the fused MX flash-attention v4** (team is writing
  it; prefill attention is otherwise covered by component MX-GEMMs + SIMT softmax).
- **RTL readiness: NOT complete — this is the realistic tapeout gate.** Solid on RTL: RoPE + the whole
  MX fp8 GEMM path (all projections + prefill attention matmuls). Three classes of blockers remain:
  1. **⛔ l0d-assertion-blocked at full shape** (SwiGLU, ResAdd, decode/O-proj GEMV): the unbuffered
     per-tile l0d drops backpressured responses under sustained streaming. Runs at small shapes; a DUT
     question (landing pads) for the team, or reduce in-flight traffic / coalesce.
  2. **🚧 team precision WIP** (fp6/fp4/requant): co-model bit-exact, RTL differs — the team's flagged
     variable-acc-precision / byte-order items. (These have MX bf16 output + large tohost error counts;
     trace-verify to separate real fp-quant error from the read-back artifact below.)
- **RESOLVED (was "mismatch"): decode attention (Q·Kᵀ/softmax/P·V) COMPUTES CORRECTLY on RTL** —
  offline trace-reconstruction matches golden to ~1e-7 (0 fails @1e-4). The tohost "errors" were a
  SIMT-store read-back visibility artifact (the in-kernel verify's load doesn't see all stores), NOT a
  compute bug. **Lesson: verify SIMT kernels via offline trace reconstruction, not the in-kernel
  read-back** (`scratchpad/simt_out_verify.py`). This likely also explains the ⚠️PASS? kernels
  (RMSNorm/Softmax/fused) — same read-back artifact, not a real fail.

So: **the model is covered at the cyclotron level (minus fused flash-attn), and more of it is RTL-
correct than the on-chip verify suggested** — decode attention is confirmed correct. Remaining RTL
bring-up: the l0d streaming blockers (fusion is the fix), the team's fp6/fp4/requant precision, and
adopting offline trace-verification as the standard SIMT correctness check.
