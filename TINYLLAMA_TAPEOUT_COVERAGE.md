# TinyLlama tapeout kernel coverage — vs the 3/6/2026 spec (Vikram/Sirius workload)

Every distinct kernel in the TinyLlama (Llama-2 arch) tapeout spec, mapped to an implementation
+ verification status. **VERIFIED** = bit-exact (or FP-tol) vs an HF-matching golden in cyclotron
this session. **EXISTS** = standalone kernel present (radiance-kernels/gemm_mxgemmini or team's),
not yet wrapped as a verified autocomp problem. **GAP** = to build.

Backends: GEMM/proj/attention-matmul → MX-Gemmini; norms/activation/RoPE/residual/decode-GEMV → Muon SIMT.

## By spec section

### Embedding
| Op | Kernel | Backend | Status | Where |
|---|---|---|---|---|
| RoPE | RoPE | SIMT | ✅ VERIFIED | test26 (sol26); head_dim≤512 (test31) |

### QKV Projection
| Op | Kernel | Backend | Status | Where |
|---|---|---|---|---|
| Q/K/V Projection | GEMM | MX | ✅ VERIFIED (fp8) | test20–24, 128×128 anchors |
| **Q/K/V proj + RoPE FUSED** | GEMM+RoPE | MX+SIMT | ✅ VERIFIED (new) | test33 — both engines, SMEM-shared |

### Multi-head Self-Attention (Prefill)
| Op | Kernel | Backend | Status | Where |
|---|---|---|---|---|
| RMSNorm (pre-norm) | RMSNorm | SIMT | ✅ VERIFIED | test27; hidden≤2048 (test30) |
| Q·Kᵀ (prefill) | GEMM | MX | ✅ VERIFIED | MX matmul (test20-class) |
| Softmax | Softmax | SIMT | ✅ VERIFIED | test5 |
| P·V | GEMM | MX | ✅ VERIFIED | MX matmul |
| O Projection | GEMM | MX | ✅ VERIFIED | MX matmul |
| Flash-attention v4 (fused) | Flash-attn | MX | ⚠️ EXISTS (team, unverified here) | gemm_mxgemmini/mxgemm.flash_contention.cpp; SIMT flash test7 ✅ |
| ResAdd | ResAdd | SIMT | ✅ VERIFIED | test28 |

### Multi-head Self-Attention (Decode + KV$)
| Op | Kernel | Backend | Status | Where |
|---|---|---|---|---|
| Q·Kᵀ (decode) | GEMV | SIMT | ✅ VERIFIED | test34 |
| Softmax (decode) | GEMV Softmax | SIMT | ✅ VERIFIED | test35 |
| P·V (decode) | GEMV | SIMT | ✅ VERIFIED | test36 |
| O Projection (decode) | GEMV | SIMT | ✅ VERIFIED (generic GEMV) | test29 |

### FFN
| Op | Kernel | Backend | Status | Where |
|---|---|---|---|---|
| RMSNorm | RMSNorm | SIMT | ✅ VERIFIED | test27 |
| Up / Gate / Down proj | GEMM | MX | ✅ VERIFIED (fp8) | test20-class |
| SwiGLU | SwiGLU | SIMT | ✅ VERIFIED | test4 (silu(gate)·up) |
| ResAdd | ResAdd | SIMT | ✅ VERIFIED | test28 |

## Datatype / requant coverage (spec test-status table)
| Kernel | Status | Where |
|---|---|---|
| MxGEMM fp8 64³, 128²×256, 128²×512 | ✅ VERIFIED | test20-24, gemm_mxgemmini fp8 |
| MxGEMM fp6 128²×256/512/1024 | ⚠️ EXISTS (kernels), need data headers + verify | gemm_mxgemmini/mxgemm.fp6.* |
| MxGEMM fp4 64³, 128²×256/1024 | ⚠️ EXISTS (kernels + fp4 data headers) | gemm_mxgemmini/mxgemm.fp4.* |
| Requantizer (fp8 output) | ⚠️ EXISTS (byte-order bug per team) | gemm_mxgemmini/requant.cpp |
| GEMM-SIMT BF16 | ⚠️ EXISTS (CI, no datacheck) | radiance-kernels/kernels/gemm_simt |

## Build queue (the true gaps), priority order
2. **GEMV-softmax** — softmax over one decode score row.
3. **fp6 / fp4 MxGEMM** as verified autocomp problems (data headers + goldens exist for some).
4. **Requantizer** — verify the fp8-output path, chase the byte-order bug.
5. **BF16 SIMT GEMM** with a datacheck (currently CI-only, no verify).
6. **MX flash-attention v4** — verify the team's fused prefill attention on the co-model.

Once each passes the cyclotron ladder it is RTL-gateable via search_then_rtl_gate.sh / rtl_gate_mx.sh.
