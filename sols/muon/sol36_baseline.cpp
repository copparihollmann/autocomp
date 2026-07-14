// Baseline decode P.V GEMV: one output d per thread; sum over S of probs[s]*V[s][d].
static inline void kernel_body(void* raw_arg, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t S = a->S, D = a->D;
  for (uint32_t d = tid; d < D; d += tpb) {
    float acc = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t s = 0; s < S; s++) acc += a->p[s] * a->V[s * D + d];
    a->out[d] = acc;
  }
}
