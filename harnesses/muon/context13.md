Tall-skinny matmul C[8,768]=A[8,256]@B[256,768], fp32. Only M=8 rows => underutilization study.
Real op: tiny_llama 8x2048, rdt 1x256@256x2048. One threadblock (128 threads).
