// Row softmax, register-cached: read each row once into locals, compute
// max + exp + sum + normalize without re-reading global, write once.
// Baseline did 3 global read passes; this does 1 read + 1 write.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;

  float r[COLS];
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    float m = a->in[base];
    #pragma unroll 4
    for (uint32_t j = 0; j < cols; j++) {
      const float v = a->in[base + j];
      r[j] = v;
      if (v > m) m = v;
    }
    float sum = 0.0f;
    #pragma unroll 4
    for (uint32_t j = 0; j < cols; j++) {
      const float e = mu_exp(r[j] - m);
      r[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    #pragma unroll 4
    for (uint32_t j = 0; j < cols; j++) {
      a->out[base + j] = r[j] * inv;
    }
  }
}
