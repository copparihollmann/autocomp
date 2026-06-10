Attention O=softmax(QK^T/sqrt(64))V, fp32, SEQ=256, HEAD_DIM=64. scratch 256x256x4=256KB > SMEM
=> online-softmax/streaming (flash) regime. Real op: pi05/smolvla long seq. One threadblock.
