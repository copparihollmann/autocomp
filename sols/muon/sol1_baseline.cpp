// Baseline: naive direct conv. One output element per thread, strided.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t total = a->OC * a->OH * a->OW;

  for (uint32_t idx = tid_in_threadblock; idx < total; idx += threads_per_threadblock) {
    const uint32_t oc = idx / (a->OH * a->OW);
    const uint32_t oh = (idx / a->OW) % a->OH;
    const uint32_t ow = idx % a->OW;
    float acc = 0.0f;
    for (uint32_t c = 0; c < a->C; c++) {
      for (uint32_t kh = 0; kh < a->KH; kh++) {
        for (uint32_t kw = 0; kw < a->KW; kw++) {
          const float x = a->in[(c * a->H + oh * a->stride + kh) * a->W + ow * a->stride + kw];
          const float w = a->w[((oc * a->C + c) * a->KH + kh) * a->KW + kw];
          acc += x * w;
        }
      }
    }
    a->out[idx] = acc;
  }
}
