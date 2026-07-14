# TinyLlama tapeout kernel coverage (Radiance: Muon SIMT + MX-Gemmini)

Maps each op in a TinyLlama (Llama-2-arch) decoder layer to the kernel that implements it, its
autocomp harness, and verification status. "PASS" = bit-exact (within FP tolerance) vs an
HF-matching golden in cyclotron. GEMMs run on MX-Gemmini; the rest are Muon SIMT.

Config targets from the tapeout spec: hidden ∈ {192, 512, 2048}, FFN hidden ∈ {192, 512, 2048},
seq ≤ 1024, 4 heads.

| # | Layer op | Kernel / harness | Backend | Status | Shape tested |
|---|---|---|---|---|---|
| 1 | **RMSNorm** (input + post-attn) | `sol27_baseline` / test27 | SIMT | ✅ PASS | 64×512 (hidden=512) |
| 2 | **RoPE** (Q, K) | `sol26_baseline` / test26 | SIMT | ✅ PASS | seq=128, 4 heads, d=128 |
| 3 | **QKV / O / FFN projections** (GEMM) | `mxgemm.fp8.*` (64×64 test20–25; 128×128 K=128/256/512) | MX | ✅ PASS (co-model + RTL-measured) | up to 128×128, K≤1024 |
| 4 | **Q·Kᵀ, softmax, P·V** (flash-attention) | `sol7/11/12` / test7,11,12 | SIMT | ⏳ verifying (prev. verified) | seq 96–128 |
| 5 | **Softmax** (standalone) | `sol5_baseline` / test5 | SIMT | ✅ PASS | 64×67 (non-pow2) |
| 6 | **SwiGLU** (silu(gate)·up) | `sol4_baseline` / test4 | SIMT | ✅ PASS | 64×128 |
| 7 | **Residual add** (×2) | `sol28_baseline` / test28 | SIMT | ✅ PASS | 128×512 |
| 8 | **Decode GEMV / KV-cache** | `sol29_baseline` / test29 | SIMT | ✅ PASS | N=512, K=512 (hidden=512) |
| 9 | **GELU** (if used) | `sol9_baseline` / test9 | SIMT | ✅ available | — |

## New this session (were genuine gaps)
- **RoPE** (test26) — HF Llama rotate_half; cos/sin precomputed so the kernel is pure FMA. Needs
  `unroll(disable)` (full unroll of the 128-wide loop blows the 256-phys-reg wall).
- **RMSNorm** (test27) — distinct from the existing LayerNorm (test8): no mean-subtraction, no beta.
- **ResAdd** (test28) — element-wise; two per block.

## Remaining to be a complete, RTL-gated TinyLlama set
- **End-to-end shape sweep** at hidden/FFN 192 and 2048 (only 512-class shapes tested so far).
- **RTL gate** each SIMT kernel (currently cyclotron-verified; RTL PASS needs the VCS full-drain
  simv — pending licence, see VERSIONS.lock).
- **Fused** RoPE-in-attention and RMSNorm-into-projection are future optimizations, not blockers.

## How to verify a kernel
```
scripts/muon/run_problem.sh <N> sols/muon/sol<N>_baseline.cpp <tag>   # build + cyclotron verdict
# RTL cycles/utilization (once the VCS or trace-fixed Verilator simv is used):
scripts/muon/kernel_utilization.py <elf> <trace.sqlite> M N K [--fmt fp8]
```
