static inline void kernel_body(void* raw_arg, uint32_t tid, uint32_t tpt, uint32_t tbid) {
  (void)tbid;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t nw = tpt / MU_NUM_THREADS;
  const uint32_t total = a->rows * a->cols;
  for (uint32_t i = tid; i < total; i += tpt) a->out[i] = a->in[i] + 1.0f;
  mu_barrier(1, nw);  // matmul-style end barrier to drain writes before completion
}
