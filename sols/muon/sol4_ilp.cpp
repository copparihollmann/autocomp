// SwiGLU element-wise, 4-wide ILP with COALESCED access: each thread handles
// indices i, i+T, i+2T, i+3T (T = threads_per_threadblock), so consecutive
// lanes still touch consecutive addresses on every load. Issue the 4 A and 4 B
// loads up front to hide global-load latency. Memory-bound kernel.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t total = a->M * a->N;
  const uint32_t T = threads_per_threadblock;
  const uint32_t step = T * 4;

  uint32_t i = tid_in_threadblock;
  for (; i + 3 * T < total; i += step) {
    const uint32_t i0 = i, i1 = i + T, i2 = i + 2 * T, i3 = i + 3 * T;
    const float a0 = a->A[i0], a1 = a->A[i1], a2 = a->A[i2], a3 = a->A[i3];
    const float b0 = a->B[i0], b1 = a->B[i1], b2 = a->B[i2], b3 = a->B[i3];
    const float s0 = 1.0f / (1.0f + mu_exp(-a0));
    const float s1 = 1.0f / (1.0f + mu_exp(-a1));
    const float s2 = 1.0f / (1.0f + mu_exp(-a2));
    const float s3 = 1.0f / (1.0f + mu_exp(-a3));
    a->C[i0] = a0 * s0 * b0;
    a->C[i1] = a1 * s1 * b1;
    a->C[i2] = a2 * s2 * b2;
    a->C[i3] = a3 * s3 * b3;
  }
  // tail
  for (; i < total; i += T) {
    const float x = a->A[i];
    const float sig = 1.0f / (1.0f + mu_exp(-x));
    a->C[i] = x * sig * a->B[i];
  }
}
