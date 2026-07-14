// Baseline decode GEMV: one output n per thread; dot product over K. Grid-stride.
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t K = a->K;
  for (uint32_t n = tid_in_threadblock; n < a->N; n += threads_per_threadblock) {
    const uint32_t base = n * K;
    float acc = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < K; k++) acc += a->W[base + k] * a->x[k];
    a->out[n] = acc;
  }
}
