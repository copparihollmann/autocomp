// Baseline layer_norm: one row per thread, two-pass (mean, then variance), then normalize+affine.
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
    float sum = 0.0f;
    for (uint32_t j = 0; j < cols; j++) sum += a->in[base + j];
    const float mean = sum / (float)cols;
    float vsum = 0.0f;
    for (uint32_t j = 0; j < cols; j++) {
      const float d = a->in[base + j] - mean;
      vsum += d * d;
    }
    const float var = vsum / (float)cols;
    const float inv_std = mu_rsqrt(var + eps);
    for (uint32_t j = 0; j < cols; j++) {
      a->out[base + j] = (a->in[base + j] - mean) * inv_std * a->gamma[j] + a->beta[j];
    }
  }
}
