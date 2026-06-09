// SMEM attention, register-LEAN v2: stage only K (row-major), V (transposed) and
// scores in SMEM; stream Q from global (each Q element read once). One score/thread
// and one output/thread, no unroll/tiling. Fewer live SMEM bases -> fits <= 31 regs.
static inline float bits_to_float(uint32_t b) { union { uint32_t u; float f; } c; c.u = b; return c.f; }
static inline uint32_t float_to_bits(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }

static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;
  const uint32_t d = a->d;
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;

  const uint32_t PAD = 16;
  const uint32_t k_stride = (d + PAD) * 4;
  const uint32_t s_stride = (seq + PAD) * 4;
  const uint32_t smem_k = 0;
  const uint32_t smem_v = smem_k + seq * k_stride;
  const uint32_t smem_s = smem_v + d * s_stride;

  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    const uint32_t row = i / d, col = i % d;
    store_shared(smem_k + row * k_stride + col * 4, 0, float_to_bits(a->K[i]));
    store_shared(smem_v + col * s_stride + row * 4, 0, float_to_bits(a->V[i]));
  }
  mu_barrier(1, num_warps);

  // Phase 1: S = scale * Q K^T, one score per thread (Q streamed from global).
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    const uint32_t i = idx / seq;
    const uint32_t j = idx % seq;
    const __global float* qrow = a->Q + i * d;
    uint32_t ka = smem_k + j * k_stride;
    float acc = 0.f;
    for (uint32_t k = 0; k < d; k++) {
      acc += qrow[k] * bits_to_float(load32_shared(ka));
      ka += 4;
    }
    store_shared(smem_s + i * s_stride + j * 4, 0, float_to_bits(acc * scale));
  }
  mu_barrier(1, num_warps);

  // Phase 2: softmax per row in SMEM.
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    const uint32_t rb = smem_s + row * s_stride;
    float m = bits_to_float(load32_shared(rb));
    for (uint32_t j = 1; j < seq; j++) {
      const float v = bits_to_float(load32_shared(rb + j * 4));
      if (v > m) m = v;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      const float e = mu_exp(bits_to_float(load32_shared(rb + j * 4)) - m);
      store_shared(rb + j * 4, 0, float_to_bits(e));
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      store_shared(rb + j * 4, 0, float_to_bits(bits_to_float(load32_shared(rb + j * 4)) * inv));
    }
  }
  mu_barrier(1, num_warps);

  // Phase 3: O = P V, one output per thread.
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    const uint32_t i = idx / d;
    const uint32_t k = idx % d;
    uint32_t pa = smem_s + i * s_stride;
    uint32_t va = smem_v + k * s_stride;
    float acc = 0.f;
    for (uint32_t j = 0; j < seq; j++) {
      acc += bits_to_float(load32_shared(pa)) * bits_to_float(load32_shared(va));
      pa += 4; va += 4;
    }
    a->O[idx] = acc;
  }
}
