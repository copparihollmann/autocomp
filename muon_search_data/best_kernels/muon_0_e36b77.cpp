// Baseline: naive fp32 matmul. One output element per thread, strided over C.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t total = args->M * args->N;

  for (uint32_t idx = tid_in_threadblock; idx < total; idx += threads_per_threadblock) {
    const uint32_t row = idx / args->N;
    const uint32_t col = idx % args->N;
    float acc = 0.0f;
    for (uint32_t k = 0; k < args->K; k++) {
      acc += args->A[row * args->K + k] * args->B[k * args->N + col];
    }
    args->C[idx] = acc;
  }
}
