fp32 matmul C[64,64] = A[64,64] @ B[64,64] on the radiance Muon SIMT GPU.
One threadblock: 2 cores x NUM_WARPS(4) warps x 16 lanes = 128 threads.
Baseline: one output element per thread, K-loop in global memory. Optimize with
shared-memory tiling (128 KiB SMEM), per-thread register tiles, loop unroll/ILP.
Verification: relative 1e-4 against fixed golden; reassociation is allowed.
