// Flash attention, KEY-BLOCKED. Same online-softmax math as sol7_baseline, but processes
// BC keys per outer step so the per-row O accumulator (in SMEM) is read-modify-written ONCE
// per block instead of once per key -> ~BC x fewer O SMEM round-trips (the dominant cost).
// Self-contained (inline __builtin_bit_cast).
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
  const uint32_t BC = 4;

  const uint32_t smem_k = 0;
  const uint32_t smem_v = smem_k + seq * d * 4;
  const uint32_t smem_o = smem_v + seq * d * 4;

  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    store_shared(smem_k + i * 4, 0, __builtin_bit_cast(uint32_t, a->K[i]));
    store_shared(smem_v + i * 4, 0, __builtin_bit_cast(uint32_t, a->V[i]));
  }
  mu_barrier(1, num_warps);

  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {
    const __global float* q = a->Q + i * d;
    const uint32_t obase = smem_o + tid_in_threadblock * d * 4;
    for (uint32_t k = 0; k < d; k++) store_shared(obase + k * 4, 0, __builtin_bit_cast(uint32_t, 0.0f));

    float m = -3.0e38f, l = 0.0f;
    for (uint32_t j0 = 0; j0 < seq; j0 += BC) {
      const uint32_t bn = (j0 + BC <= seq) ? BC : (seq - j0);
      // Block scores.
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      for (uint32_t k = 0; k < d; k++) {
        const float qk = q[k];
        if (bn > 0) s0 += qk * __builtin_bit_cast(float, load32_shared(smem_k + (j0 + 0) * d * 4 + k * 4));
        if (bn > 1) s1 += qk * __builtin_bit_cast(float, load32_shared(smem_k + (j0 + 1) * d * 4 + k * 4));
        if (bn > 2) s2 += qk * __builtin_bit_cast(float, load32_shared(smem_k + (j0 + 2) * d * 4 + k * 4));
        if (bn > 3) s3 += qk * __builtin_bit_cast(float, load32_shared(smem_k + (j0 + 3) * d * 4 + k * 4));
      }
      s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
      // Block max.
      float bm = -3.0e38f;
      if (bn > 0 && s0 > bm) bm = s0;
      if (bn > 1 && s1 > bm) bm = s1;
      if (bn > 2 && s2 > bm) bm = s2;
      if (bn > 3 && s3 > bm) bm = s3;
      const float mnew = bm > m ? bm : m;
      const float corr = (m < -1.0e37f) ? 0.0f : mu_exp(m - mnew);
      const float p0 = (bn > 0) ? mu_exp(s0 - mnew) : 0.0f;
      const float p1 = (bn > 1) ? mu_exp(s1 - mnew) : 0.0f;
      const float p2 = (bn > 2) ? mu_exp(s2 - mnew) : 0.0f;
      const float p3 = (bn > 3) ? mu_exp(s3 - mnew) : 0.0f;
      l = l * corr + (p0 + p1) + (p2 + p3);
      // Update O once for the whole block.
      for (uint32_t k = 0; k < d; k++) {
        float acc = 0.0f;
        if (bn > 0) acc += p0 * __builtin_bit_cast(float, load32_shared(smem_v + (j0 + 0) * d * 4 + k * 4));
        if (bn > 1) acc += p1 * __builtin_bit_cast(float, load32_shared(smem_v + (j0 + 1) * d * 4 + k * 4));
        if (bn > 2) acc += p2 * __builtin_bit_cast(float, load32_shared(smem_v + (j0 + 2) * d * 4 + k * 4));
        if (bn > 3) acc += p3 * __builtin_bit_cast(float, load32_shared(smem_v + (j0 + 3) * d * 4 + k * 4));
        const float ok = __builtin_bit_cast(float, load32_shared(obase + k * 4)) * corr + acc;
        store_shared(obase + k * 4, 0, __builtin_bit_cast(uint32_t, ok));
      }
      m = mnew;
    }

    const float inv = 1.0f / l;
    for (uint32_t k = 0; k < d; k++) a->O[i * d + k] = __builtin_bit_cast(float, load32_shared(obase + k * 4)) * inv;
  }
}
