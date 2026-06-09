SwiGLU: C[64,128] = silu(A) * B = A*sigmoid(A)*B (fp32, elementwise).
One threadblock (128 threads). Use mu_exp() for sigmoid (must match golden).
Memory-bound: optimize coalescing and ILP. Tolerance 2e-4.
