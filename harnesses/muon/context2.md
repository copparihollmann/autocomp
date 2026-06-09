Single-head attention O = softmax(Q K^T / sqrt(64)) V, fp32, SEQ=64, HEAD_DIM=64.
One threadblock (128 threads); scratch[SEQ*SEQ] in global memory.
Baseline: 3 phases (scores, softmax, V product) with barriers. Use mu_exp() only.
Optimize: smem tiles, fused phases, warp reductions, fewer barriers.
Verification: relative 2e-4 vs fixed golden.
