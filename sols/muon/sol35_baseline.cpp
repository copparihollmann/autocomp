// Baseline GEMV softmax: single row of length S; one thread does max/sum/normalize.
static inline void kernel_body(void* raw_arg, uint32_t tid, uint32_t tpb, uint32_t) {
  (void)tpb;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  if (tid != 0) return;           // one decode row -> single reducer
  const uint32_t S = a->S;
  float m = a->in[0];
  #pragma clang loop unroll(disable)
  for (uint32_t j = 1; j < S; j++) { float v = a->in[j]; if (v > m) m = v; }
  float sum = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint32_t j = 0; j < S; j++) { float e = mu_exp(a->in[j] - m); a->out[j] = e; sum += e; }
  const float inv = 1.0f / sum;
  #pragma clang loop unroll(disable)
  for (uint32_t j = 0; j < S; j++) a->out[j] *= inv;
}
