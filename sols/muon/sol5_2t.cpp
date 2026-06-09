// Row softmax using ALL threads: 2 threads per row (baseline leaves ~half the
// threads idle since rows < total threads). exp work is split across the 2 threads
// and the per-row max/sum are reduced via shared memory. A wave loop keeps every
// thread executing the same number of barriers regardless of threads_per_threadblock,
// so the reductions never deadlock.
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
  const uint32_t rows = a->rows;
  const uint32_t cols = a->cols;
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;

  const uint32_t my_pair = tid_in_threadblock / 2;   // row within a wave
  const uint32_t half = tid_in_threadblock & 1u;     // column half
  const uint32_t pairs = threads_per_threadblock / 2; // rows processed per wave
  const uint32_t waves = (rows + pairs - 1) / pairs;

  const uint32_t hcols = (cols + 1) / 2;
  const uint32_t c0 = half ? hcols : 0;
  const uint32_t c1 = half ? cols : hcols;

  // Per-(threadblock) SMEM: exp cache [pairs][cols(pad)] + max/sum reductions.
  const uint32_t row_pitch = (cols + 1) * 4;
  const uint32_t smem_exp = 0;
  const uint32_t smem_max = smem_exp + pairs * row_pitch;     // [my_pair][half]
  const uint32_t smem_sum = smem_max + pairs * 2 * 4;

  for (uint32_t w = 0; w < waves; w++) {
    const uint32_t row = w * pairs + my_pair;
    const bool active = row < rows;
    const __global float* x = a->in + row * cols;

    // Phase 1: partial max over my columns.
    float pm = -3.0e38f;
    if (active) {
      for (uint32_t j = c0; j < c1; j++) {
        const float v = x[j];
        if (v > pm) pm = v;
      }
      store_shared(smem_max + (my_pair * 2 + half) * 4, 0, float_to_bits(pm));
    }
    mu_barrier(1, num_warps);

    float m = -3.0e38f, inv = 0.0f;
    if (active) {
      m = bits_to_float(load32_shared(smem_max + (my_pair * 2 + 0) * 4));
      const float m1 = bits_to_float(load32_shared(smem_max + (my_pair * 2 + 1) * 4));
      if (m1 > m) m = m1;
      // Phase 2: exp my columns into SMEM, partial sum.
      float ps = 0.0f;
      for (uint32_t j = c0; j < c1; j++) {
        const float e = mu_exp(x[j] - m);
        store_shared(smem_exp + my_pair * row_pitch + j * 4, 0, float_to_bits(e));
        ps += e;
      }
      store_shared(smem_sum + (my_pair * 2 + half) * 4, 0, float_to_bits(ps));
    }
    mu_barrier(1, num_warps);

    if (active) {
      const float sum = bits_to_float(load32_shared(smem_sum + (my_pair * 2 + 0) * 4))
                      + bits_to_float(load32_shared(smem_sum + (my_pair * 2 + 1) * 4));
      inv = 1.0f / sum;
      // Phase 3: normalize my columns from the SMEM cache.
      for (uint32_t j = c0; j < c1; j++) {
        const float e = bits_to_float(load32_shared(smem_exp + my_pair * row_pitch + j * 4));
        a->out[row * cols + j] = e * inv;
      }
    }
    mu_barrier(1, num_warps);  // protect SMEM reuse before next wave
  }
}
