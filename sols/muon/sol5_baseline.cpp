// Baseline: row softmax. One row per thread, strided.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);

  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * a->cols;
    float m = a->in[base];
    for (uint32_t j = 1; j < a->cols; j++) {
      const float v = a->in[base + j];
      if (v > m) m = v;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < a->cols; j++) {
      const float e = mu_exp(a->in[base + j] - m);
      a->out[base + j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < a->cols; j++) {
      a->out[base + j] *= inv;
    }
  }
}
