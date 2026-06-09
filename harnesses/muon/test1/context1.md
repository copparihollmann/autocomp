fp32 conv2d patch-embedding tile (vision transformer): OUT[16,4,4] = IN[3,64,64] * W[16,3,16,16],
stride 16 = non-overlapping patches. One threadblock (128 threads).
Baseline computes each output element directly (768 MACs each).
Optimize: shared-memory weight reuse, per-warp patch assignment, reuse the input patch.
Verification: relative 1e-4 against fixed golden.
