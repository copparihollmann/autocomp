// Flash attention, 2-QUERY-ROWS-PER-THREAD. Same online-softmax math as sol7_baseline, but
// each thread handles two query rows (i0,i1) at once so every K[j]/V[j] SMEM load is REUSED
// for both rows -> ~2x fewer SMEM loads (the dominant cost; key-blocking alone didn't help
// because load VOLUME, not the O round-trip, is the bottleneck). Self-contained.
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

  const uint32_t smem_k = 0;
  const uint32_t smem_v = smem_k + seq * d * 4;
  const uint32_t smem_o = smem_v + seq * d * 4;   // 2 O slots per thread

  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    store_shared(smem_k + i * 4, 0, __builtin_bit_cast(uint32_t, a->K[i]));
    store_shared(smem_v + i * 4, 0, __builtin_bit_cast(uint32_t, a->V[i]));
  }
  mu_barrier(1, num_warps);

  for (uint32_t i0 = tid_in_threadblock; i0 < seq; i0 += 2 * threads_per_threadblock) {
    const uint32_t i1 = i0 + threads_per_threadblock;
    const bool has1 = (i1 < seq);
    const __global float* q0 = a->Q + i0 * d;
    const __global float* q1 = a->Q + (has1 ? i1 : i0) * d;
    const uint32_t o0 = smem_o + tid_in_threadblock * 2 * d * 4;
    const uint32_t o1 = o0 + d * 4;
    for (uint32_t k = 0; k < d; k++) {
      store_shared(o0 + k * 4, 0, __builtin_bit_cast(uint32_t, 0.0f));
      store_shared(o1 + k * 4, 0, __builtin_bit_cast(uint32_t, 0.0f));
    }

    float m0 = -3.0e38f, l0 = 0.0f, m1 = -3.0e38f, l1 = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      const uint32_t kb = smem_k + j * d * 4;
      float s0 = 0.0f, s1 = 0.0f;
      for (uint32_t k = 0; k < d; k++) {
        const float kk = __builtin_bit_cast(float, load32_shared(kb + k * 4)); // one K load, reused
        s0 += q0[k] * kk;
        s1 += q1[k] * kk;
      }
      s0 *= scale; s1 *= scale;

      const float mn0 = s0 > m0 ? s0 : m0;
      const float c0 = (m0 < -1.0e37f) ? 0.0f : mu_exp(m0 - mn0);
      const float p0 = mu_exp(s0 - mn0);
      l0 = l0 * c0 + p0;
      const float mn1 = s1 > m1 ? s1 : m1;
      const float c1 = (m1 < -1.0e37f) ? 0.0f : mu_exp(m1 - mn1);
      const float p1 = mu_exp(s1 - mn1);
      l1 = l1 * c1 + p1;

      const uint32_t vb = smem_v + j * d * 4;
      for (uint32_t k = 0; k < d; k++) {
        const float vv = __builtin_bit_cast(float, load32_shared(vb + k * 4)); // one V load, reused
        const float a0 = __builtin_bit_cast(float, load32_shared(o0 + k * 4)) * c0 + p0 * vv;
        store_shared(o0 + k * 4, 0, __builtin_bit_cast(uint32_t, a0));
        const float a1 = __builtin_bit_cast(float, load32_shared(o1 + k * 4)) * c1 + p1 * vv;
        store_shared(o1 + k * 4, 0, __builtin_bit_cast(uint32_t, a1));
      }
      m0 = mn0; m1 = mn1;
    }

    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    for (uint32_t k = 0; k < d; k++) {
      a->O[i0 * d + k] = __builtin_bit_cast(float, load32_shared(o0 + k * 4)) * inv0;
      if (has1) a->O[i1 * d + k] = __builtin_bit_cast(float, load32_shared(o1 + k * 4)) * inv1;
    }
  }
}
