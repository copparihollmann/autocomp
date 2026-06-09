Single-head attention O = softmax(Q K^T / sqrt(64)) V, fp32, SEQ=96, HEAD_DIM=64.
Same as test2 but bigger sequence: scratch (96x96) no longer fits 128 KiB SMEM,
so a flash-attention style streaming softmax (tile K/V, online max/sum rescale) is
the intended optimization. Use mu_exp() only. Tolerance 2e-4.
