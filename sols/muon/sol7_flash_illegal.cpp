// Flash attention (online softmax) — accelerator-centric: the SEQ x SEQ score matrix
// is NEVER materialized in global memory (that round-trip is the softmax bottleneck at
// seq=128). K and V are staged once into SMEM; each query row streams the keys, keeping
// a running max (m) and denominator (l) and a per-row O accumulator in SMEM, rescaling
// the accumulator whenever the max grows. One query row per thread (looped). Self-contained
// (inline __builtin_bit_cast — no top-level helpers, survives the search's code extraction).
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

  // SMEM layout: K_s[seq*d] | V_s[seq*d] | O_s[threads*d]  (all row-major, 4B/elem)
  const uint32_t smem_k = 0;
  const uint32_t smem_v = smem_k + seq * d * 4;
  const uint32_t smem_o = smem_v + seq * d * 4;

  // Stage K, V into SMEM once.
  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    store_shared(smem_k + i * 4, 0, __builtin_bit_cast(uint32_t, a->K[i]));
    store_shared(smem_v + i * 4, 0, __builtin_bit_cast(uint32_t, a->V[i]));
  }
  mu_barrier(1, num_warps);

  // One query row per thread (strided). O accumulator lives in this thread's SMEM slot.
  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {
    const __global float* q = a->Q + i * d;
    const uint32_t obase = smem_o + tid_in_threadblock * d * 4;
    for (uint32_t k = 0; k < d; k++) store_shared(obase + k * 4, 0, __builtin_bit_cast(uint32_t, 0.0f));

    float m = -3.0e38f, l = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      const uint32_t kbase = smem_k + j * d * 4;
      float s = 0.0f;
      for (uint32_t k = 0; k < d; k++) s += q[k] * __builtin_bit_cast(float, load32_shared(kbase + k * 4));
      s *= scale;

      const float mnew = s > m ? s : m;
      // corr rescales the running accumulator. On the first key m is -inf-sentinel and
      // m-mnew underflows the polynomial mu_exp's int cast -> force 0 (empty accumulator).
      const float corr = (m < -1.0e37f) ? 0.0f : mu_exp(m - mnew);   // 1.0 when m unchanged
      const float p = mu_exp(s - mnew);
      l = l * corr + p;

      const uint32_t vbase = smem_v + j * d * 4;
      for (uint32_t k = 0; k < d; k++) {
        float ok = __builtin_bit_cast(float, load32_shared(obase + k * 4));
        ok = ok * corr + p * __builtin_bit_cast(float, load32_shared(vbase + k * 4));
        store_shared(obase + k * 4, 0, __builtin_bit_cast(uint32_t, ok));
      }
      m = mnew;
    }

    const float inv = 1.0f / l;
    for (uint32_t k = 0; k < d; k++) a->O[i * d + k] = __builtin_bit_cast(float, load32_shared(obase + k * 4)) * inv;
  }
}
