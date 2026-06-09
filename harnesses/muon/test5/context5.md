Row softmax: OUT[64, 67] = softmax(IN[r, :]), fp32. One threadblock (128 threads).
Baseline: one row per thread, sequential. Optimize: a warp per row, warp-level
reduction (smem); use mu_exp() only. Tolerance 2e-4.
