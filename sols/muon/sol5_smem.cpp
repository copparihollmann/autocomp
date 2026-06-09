// Row softmax, SMEM-staged: the baseline streams each row from global memory 3x
// (max pass, exp pass, normalize pass). Global latency dominates. Here each thread
// stages its row into shared memory once (1 global read), runs max/exp/normalize out
// of SMEM, and writes the result once (1 global write). Loads are issued in an
// unrolled batch first to hide latency (ILP).
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

  // Per-thread SMEM row buffer (one per lane in flight). Lay out by tid so each
  // thread owns a contiguous, bank-padded row region.
  const uint32_t row_pitch = (cols + 1) * 4;
  const uint32_t my_base = tid_in_threadblock * row_pitch;

  for (uint32_t row = tid_in_threadblock; row < rows; row += threads_per_threadblock) {
    const __global float* x = a->in + row * cols;

    // Stage row into SMEM (single global read pass) + track max on the way in.
    float m = -3.0e38f;
    uint32_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      const float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], v3 = x[j + 3];
      store_shared(my_base + (j + 0) * 4, 0, float_to_bits(v0));
      store_shared(my_base + (j + 1) * 4, 0, float_to_bits(v1));
      store_shared(my_base + (j + 2) * 4, 0, float_to_bits(v2));
      store_shared(my_base + (j + 3) * 4, 0, float_to_bits(v3));
      if (v0 > m) m = v0;
      if (v1 > m) m = v1;
      if (v2 > m) m = v2;
      if (v3 > m) m = v3;
    }
    for (; j < cols; j++) {
      const float v = x[j];
      store_shared(my_base + j * 4, 0, float_to_bits(v));
      if (v > m) m = v;
    }

    // exp out of SMEM, overwrite in place, accumulate sum.
    float sum = 0.0f;
    for (j = 0; j < cols; j++) {
      const float e = mu_exp(bits_to_float(load32_shared(my_base + j * 4)) - m);
      store_shared(my_base + j * 4, 0, float_to_bits(e));
      sum += e;
    }

    // normalize from SMEM, single global write pass.
    const float inv = 1.0f / sum;
    for (j = 0; j < cols; j++) {
      a->out[row * cols + j] = bits_to_float(load32_shared(my_base + j * 4)) * inv;
    }
  }
}
