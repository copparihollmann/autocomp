Attention O=softmax(QK^T/sqrt(72))V, fp32, SEQ=96, HEAD_DIM=72 (non-power-of-2).
Real op: pi05 16x256x72. Stresses strided SMEM access. One threadblock.
