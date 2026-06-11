static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;
  const uint32_t d   = a->d;
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;

  // Stage the seq*seq score matrix in SMEM at base 0.
  // 64*64*4 = 16 KB, well within 128 KB.
  const uint32_t smem_base = 0x0;

  // Phase 1: compute QK^T scores, write directly to SMEM.
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    const uint32_t i = idx / seq;
    const uint32_t j = idx % seq;
    const __global float* Q = a->Q;
    const __global float* K = a->K;
    float acc = 0.0f;
    for (uint32_t k = 0; k < d; k++) {
      acc += Q[i * d + k] * K[j * d + k];
    }
    acc *= scale;
    uint32_t addr = smem_base + idx * 4;
    store_shared(addr, 0, __builtin_bit_cast(uint32_t, acc));
  }
  mu_barrier(1, num_warps);

  // Phase 2: softmax per row, reading and writing SMEM.
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    uint32_t row_base = smem_base + row * seq * 4;

    // Find row max.
    float m = __builtin_bit_cast(float, load32_shared(row_base));
    for (uint32_t j = 1; j < seq; j++) {
      float v = __builtin_bit_cast(float, load32_shared(row_base + j * 4));
      if (v > m) m = v;
    }

    // Compute exp(x - m) and accumulate sum.
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      float v = __builtin_bit_cast(float, load32_shared(row_base + j * 4));
      float e = mu_exp(v - m);
      store_shared(row_base + j * 4, 0, __builtin_bit_cast(uint32_t, e));
      sum += e;
    }

    // Normalize.
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      uint32_t addr = row_base + j * 4;
      float e = __builtin_bit_cast(float, load32_shared(addr));
      store_shared(addr, 0, __builtin_bit_cast(uint32_t, e * inv));
    }
  }
  mu_barrier(1, num_warps);

  // Phase 3: O = P * V, reading P from SMEM, V from global.
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    const uint32_t i = idx / d;
    const uint32_t k = idx % d;
    const __global float* V = a->V;
    float acc = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      float p = __builtin_bit_cast(float, load32_shared(smem_base + (i * seq + j) * 4));
      acc += p * V[j * d + k];
    }
    a->O[idx] = acc;
  }
}
