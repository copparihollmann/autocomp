// Gemma-2 scaled embedding gather: OUT[t,:] = TABLE[ids[t],:] * sqrt(D).
// One token-row per thread, grid-stride; gather the row TABLE[id] and scale by 48.0 (=sqrt(2304)).
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t D = a->D;
  const float scale = 48.0f; // sqrt(2304), exact
  for (uint32_t t = tid_in_threadblock; t < a->T; t += threads_per_threadblock) {
    const uint32_t src = a->ids[t] * D;
    const uint32_t dst = t * D;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++) a->out[dst + j] = a->table[src + j] * scale;
  }
}
