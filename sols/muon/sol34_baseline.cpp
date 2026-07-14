// Baseline decode Q.K^T GEMV: one score s per thread; dot over D; scale.
static inline void kernel_body(void* raw_arg, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t D = a->D;
  for (uint32_t s = tid; s < a->S; s += tpb) {
    const uint32_t base = s * D;
    float acc = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t d = 0; d < D; d++) acc += a->K[base + d] * a->q[d];
    a->out[s] = acc * a->scale;
  }
}
