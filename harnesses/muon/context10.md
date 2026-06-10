Large-K matmul C[64,64]=A[64,768]@B[768,64], fp32. K=768 > SMEM => K-streaming kernel.
Real op: smolvla 1024x768@768x768. Contrast with test6 (K=128 fits). One threadblock (128 threads).
