Single-head attention O = softmax(Q K^T / sqrt(64)) V, fp32, SEQ=128, HEAD_DIM=64.
One threadblock (128 threads). At seq=128 the SEQ*SEQ score matrix is 64KB — it does
NOT fit cache, so the naive 3-phase baseline (write scores to global, barrier, softmax,
barrier, read for PV) is DRAM-bound on the score round-trips: softmax is the bottleneck.
Accelerator-centric approach (sol7_baseline): FLASH ATTENTION — online softmax. Stage
K and V in SMEM; for each query row stream keys, keep running max m and denom l, and a
per-row O accumulator (in SMEM); rescale on max-updates. The SxS score matrix is NEVER
materialized in global memory. Optimize further: warp-cooperative dot/reduction, KV
tiling, register-blocked O, fewer SMEM round-trips, double-buffering.
Use mu_exp() only. Verification: relative 2e-4 vs fixed golden.
