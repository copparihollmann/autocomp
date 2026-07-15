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
| Attn (prefill) | RMSNorm (pre-norm) | RMSNorm | SIMT | ✅ | ⚠️PASS? | test27; completed, verify verdict unconfirmed |
| Attn (prefill) | Q·Kᵀ | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | Softmax | Softmax | SIMT | ✅ | ⚠️PASS? | test5; completed, verdict unconfirmed |
| Attn (prefill) | P·V | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | O Projection | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| Attn (prefill) | **Flash-attn v4 (fused MX)** | Flash-attn | MX | ❌ **GAP** | — | **team WIP ("Writing")**; covered piecewise by MX-GEMM+softmax + SIMT flash (test7) |
| Attn (prefill) | ResAdd | ResAdd | SIMT | ✅ | **⛔L0D** | test28; blocked at 128×512 (l0d assertion) |
| Attn (decode) | Q·Kᵀ | GEMV | SIMT | ✅ | **❌MISMATCH** | test34; drain-invariant (29/128 wrong) — real cyclotron-vs-RTL |
| Attn (decode) | Softmax | GEMV-softmax | SIMT | ✅ | ❌MISMATCH | test35 (tohost=219) |
| Attn (decode) | P·V | GEMV | SIMT | ✅ | ❌MISMATCH | test36 (tohost=65) |
| Attn (decode) | O Projection | GEMV | SIMT | ✅ | **⛔L0D** | test29; 512×512 GEMV trips l0d at full shape |
| FFN | RMSNorm | RMSNorm | SIMT | ✅ | ⚠️PASS? | test27 (shared) |
| FFN | Up/Gate/Down proj | GEMM | MX | ✅ | ✅ works | MX fp8 GEMM path |
| FFN | SwiGLU | SwiGLU | SIMT | ✅ | **⛔L0D** | test4; blocked at 64×512 (l0d assertion) |
| FFN | ResAdd | ResAdd | SIMT | ✅ | ⛔L0D | (shared with attn) |

## Datatype / requant coverage
| Kernel | KERNEL (co-model) | RTL | Notes |
|---|---|---|---|
| MxGEMM fp8 | ✅ | ✅ works | the workhorse; 92% compute, verified path |
| MxGEMM fp6 (LUT e3m2) | ✅ bit-exact | ❌MISMATCH / 🚧TEAM | test38 (tohost=2995) — team fp6 variable-acc-precision WIP |
| MxGEMM fp4 (e2m1) | ✅ bit-exact | ❌MISMATCH / 🚧TEAM | test37 (tohost=2061) — team fp4 precision WIP |
| Requantizer (fp8 out) | ✅ bit-exact | ❌MISMATCH / 🚧TEAM | test39 (tohost=4593) — team byte-order WIP |
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
  2. **❌ RTL correctness mismatch** (decode attention Q·Kᵀ/softmax/P·V): drain-invariant, so a real
     cyclotron-vs-RTL divergence to root-cause (FP ordering / SFU), not write-drain.
  3. **🚧 team precision WIP** (fp6/fp4/requant): co-model bit-exact, RTL differs — the team's flagged
     variable-acc-precision / byte-order items.
- **⚠️ Confirm-needed:** RMSNorm, Softmax, fused-QKV+RoPE completed on RTL without a clean PASS verdict
  under DRAIN=2000 — re-run at large DRAIN to confirm they PASS (expected; write-drain, not a mismatch).

So: **yes, we have a realistic kernel list covering the whole model at the cyclotron level (minus fused
flash-attn); the remaining work before tapeout is RTL bring-up — clearing the l0d blockers, root-
causing the decode-attention mismatch, and the team's fp6/fp4/requant precision.**
