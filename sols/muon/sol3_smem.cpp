// SMEM attention for seq=96: K (row-major), V (transposed), and scores S in
// padded SMEM (~92 KiB). Q rows are read from global (each element read once).
// Register-tiled QK^T and PV (4 outputs/thread).
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
  const uint32_t seq = a->seq;          // 96
  const uint32_t d = a->d;              // 64
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;           // 1/sqrt(64)

  const uint32_t PAD = 16;
  const uint32_t k_stride = (d + PAD) * 4;    // K row pitch
  const uint32_t s_stride = (seq + PAD) * 4;  // V column / S row pitch
  const uint32_t smem_k = 0;                  // 96 x 80 x4 = 30 KiB
  const uint32_t smem_v = smem_k + seq * k_stride;   // 64 x 112 x4 = 28 KiB
  const uint32_t smem_s = smem_v + d * s_stride;     // 96 x 112 x4 = 42 KiB

  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    const uint32_t row = i / d, col = i % d;
    store_shared(smem_k + row * k_stride + col * 4, 0, float_to_bits(a->K[i]));
    store_shared(smem_v + col * s_stride + row * 4, 0, float_to_bits(a->V[i]));
  }
  mu_barrier(1, num_warps);

  // Phase 1: S = scale * Q K^T; 4 scores per thread (one Q row, 4 K rows).
  const uint32_t squads = (seq * seq) / 4;
  for (uint32_t q = tid_in_threadblock; q < squads; q += threads_per_threadblock) {
    const uint32_t i = q / (seq / 4);
    const uint32_t j0 = (q % (seq / 4)) * 4;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const __global float* qrow = a->Q + i * d;
    const uint32_t kb = smem_k + j0 * k_stride;
    #pragma unroll 4
    for (uint32_t k = 0; k < d; k++) {
      const uint32_t off = k * 4;
      const float qv = qrow[k];
      s0 += qv * bits_to_float(load32_shared(kb + off));
      s1 += qv * bits_to_float(load32_shared(kb + k_stride + off));
      s2 += qv * bits_to_float(load32_shared(kb + 2 * k_stride + off));
      s3 += qv * bits_to_float(load32_shared(kb + 3 * k_stride + off));
    }
    const uint32_t sb = smem_s + i * s_stride + j0 * 4;
    store_shared(sb, 0, float_to_bits(s0 * scale));
    store_shared(sb, 4, float_to_bits(s1 * scale));
    store_shared(sb, 8, float_to_bits(s2 * scale));
    store_shared(sb, 12, float_to_bits(s3 * scale));
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

  // Phase 3: O = P V; 4 outputs per thread (one P row, 4 V columns).
  const uint32_t oquads = (seq * d) / 4;
  for (uint32_t q = tid_in_threadblock; q < oquads; q += threads_per_threadblock) {
    const uint32_t i = q / (d / 4);
    const uint32_t k0 = (q % (d / 4)) * 4;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    const uint32_t pb = smem_s + i * s_stride;
    const uint32_t vb = smem_v + k0 * s_stride;
    #pragma unroll 4
    for (uint32_t j = 0; j < seq; j++) {
      const uint32_t off = j * 4;
      const float p = bits_to_float(load32_shared(pb + off));
      o0 += p * bits_to_float(load32_shared(vb + off));
      o1 += p * bits_to_float(load32_shared(vb + s_stride + off));
      o2 += p * bits_to_float(load32_shared(vb + 2 * s_stride + off));
      o3 += p * bits_to_float(load32_shared(vb + 3 * s_stride + off));
    }
    const uint32_t ob = i * d + k0;
    a->O[ob + 0] = o0;
    a->O[ob + 1] = o1;
    a->O[ob + 2] = o2;
    a->O[ob + 3] = o3;
  }
}
