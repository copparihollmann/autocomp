// Baseline RMSNorm: one row per thread. Sum of squares -> rsqrt(ms+eps) -> scale by gamma.
// (No mean subtraction, no beta -- Llama RMSNorm.) unroll(disable) keeps register pressure
// under the 256-phys-reg wall.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  const float eps = 1.0e-5f;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    float ss = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++) { const float v = a->in[base + j]; ss += v * v; }
    const float inv = mu_rsqrt(ss / (float)cols + eps);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++) a->out[base + j] = a->in[base + j] * inv * a->gamma[j];
  }
}
