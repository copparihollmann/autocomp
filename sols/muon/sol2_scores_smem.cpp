// Minimal-register attention win: identical 3-phase structure to the naive baseline
// (one score/thread, one output/thread — NO register tiling, so it stays inside the
// 32-reg/warp RTL budget), but the SEQ*SEQ score matrix lives in SMEM instead of the
// global scratch buffer. That removes the global write+read round-trip of the scores
// (the only memory traffic the naive baseline adds beyond the unavoidable Q/K/V reads),
// while keeping Q/K/V reads in global (cached). Register footprint ~ naive + 1 base addr.
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
  const float scale = 0.125f; // 1/sqrt(64)
  const uint32_t S = 0; // SMEM base for scores[seq*seq]

  // Phase 1: scores in SMEM.
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    const uint32_t i = idx / seq;
    const uint32_t j = idx % seq;
    float acc = 0.0f;
    for (uint32_t k = 0; k < d; k++) {
      acc += a->Q[i * d + k] * a->K[j * d + k];
    }
    store_shared(S + idx * 4, 0, __builtin_bit_cast(uint32_t, acc * scale));
  }
  mu_barrier(1, num_warps);

  // Phase 2: softmax per row, in SMEM.
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    const uint32_t rb = S + row * seq * 4;
    float m = __builtin_bit_cast(float, load32_shared(rb));
    for (uint32_t j = 1; j < seq; j++) {
      const float v = __builtin_bit_cast(float, load32_shared(rb + j * 4));
      if (v > m) m = v;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      const float e = mu_exp(__builtin_bit_cast(float, load32_shared(rb + j * 4)) - m);
      store_shared(rb + j * 4, 0, __builtin_bit_cast(uint32_t, e));
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      store_shared(rb + j * 4, 0, __builtin_bit_cast(uint32_t,
        __builtin_bit_cast(float, load32_shared(rb + j * 4)) * inv));
    }
  }
  mu_barrier(1, num_warps);

  // Phase 3: O = P V; scores from SMEM, V from global.
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    const uint32_t i = idx / d;
    const uint32_t k = idx % d;
    const uint32_t pb = S + i * seq * 4;
    float acc = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      acc += __builtin_bit_cast(float, load32_shared(pb + j * 4)) * a->V[j * d + k];
    }
    a->O[idx] = acc;
  }
}
