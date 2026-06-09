fp32 matmul C[64,64] = A[64,768] @ B[768,64] on the radiance Muon SIMT GPU.
smolvla-representative tile: M=64, N=64, K=768 — the 768 contraction depth mirrors
smolvla's dominant 1024x768 @ 768x768 GEMM (12x deeper than a 64^3 tile, so each
output reuses A/B rows 768 times → on-chip staging of the K-strip pays off far more).
One threadblock: 2 cores x NUM_WARPS(4) warps x 16 lanes = 128 threads.
Baseline: one output element per thread, K-loop in global memory. Optimize with
shared-memory tiling (128 KiB SMEM), per-thread register tiles, loop unroll/ILP.
Verification: relative 1e-4 against fixed golden; reassociation is allowed.
