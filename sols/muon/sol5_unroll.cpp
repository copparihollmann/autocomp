// Row softmax, baseline streaming structure but ILP-friendly: unroll each pass so
// independent loads/compares/exps issue back-to-back instead of a dependent chain,
// and use branchless max. No SMEM, no cross-thread sync (no reuse to amortize those).
static inline float fmaxf_(float a, float b) { return a > b ? a : b; }

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

  for (uint32_t row = tid_in_threadblock; row < rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    const __global float* x = a->in + base;
    __global float* y = a->out + base;

    // Pass 1: row max, 4-way unrolled with 4 independent accumulators.
    float m0 = -3.0e38f, m1 = -3.0e38f, m2 = -3.0e38f, m3 = -3.0e38f;
    uint32_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      m0 = fmaxf_(m0, x[j + 0]);
      m1 = fmaxf_(m1, x[j + 1]);
      m2 = fmaxf_(m2, x[j + 2]);
      m3 = fmaxf_(m3, x[j + 3]);
    }
    float m = fmaxf_(fmaxf_(m0, m1), fmaxf_(m2, m3));
    for (; j < cols; j++) m = fmaxf_(m, x[j]);

    // Pass 2: exp + sum, 4-way unrolled with 4 independent sum accumulators.
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (j = 0; j + 4 <= cols; j += 4) {
      const float e0 = mu_exp(x[j + 0] - m);
      const float e1 = mu_exp(x[j + 1] - m);
      const float e2 = mu_exp(x[j + 2] - m);
      const float e3 = mu_exp(x[j + 3] - m);
      y[j + 0] = e0; y[j + 1] = e1; y[j + 2] = e2; y[j + 3] = e3;
      s0 += e0; s1 += e1; s2 += e2; s3 += e3;
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; j < cols; j++) { const float e = mu_exp(x[j] - m); y[j] = e; sum += e; }

    // Pass 3: normalize, 4-way unrolled.
    const float inv = 1.0f / sum;
    for (j = 0; j + 4 <= cols; j += 4) {
      y[j + 0] *= inv; y[j + 1] *= inv; y[j + 2] *= inv; y[j + 3] *= inv;
    }
    for (; j < cols; j++) y[j] *= inv;
  }
}
