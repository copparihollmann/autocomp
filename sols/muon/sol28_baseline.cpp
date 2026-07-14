// Baseline ResAdd: element-wise out = in + res, one row per thread, grid-stride.
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++) a->out[base + j] = a->in[base + j] + a->res[base + j];
  }
}
