// SMEM matmul, 2x8 register tile: each thread computes 2 rows x 8 cols.
// A reads amortized over 8 columns; B reads amortized over 2 rows.
static inline float bits_to_float(uint32_t b) { union { uint32_t u; float f; } c; c.u = b; return c.f; }
static inline uint32_t float_to_bits(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }

static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t M = args->M, N = args->N, K = args->K;
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const uint32_t PAD = 16;
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;

  for (uint32_t i = tid_in_threadblock; i < M * K; i += threads_per_threadblock) {
    store_shared(smem_a + i * 4, 0, float_to_bits(args->A[i]));
  }
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    const uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * (K + PAD) + k) * 4, 0, float_to_bits(args->B[i]));
  }
  mu_barrier(1, num_warps);

  const uint32_t tiles = (M * N) / 16;  // 2x8 outputs per tile
  const uint32_t tiles_per_row = N / 8;
  for (uint32_t t = tid_in_threadblock; t < tiles; t += threads_per_threadblock) {
    const uint32_t row0 = (t / tiles_per_row) * 2;
    const uint32_t col0 = (t % tiles_per_row) * 8;

    float a0c0 = 0.f, a0c1 = 0.f, a0c2 = 0.f, a0c3 = 0.f;
    float a0c4 = 0.f, a0c5 = 0.f, a0c6 = 0.f, a0c7 = 0.f;
    float a1c0 = 0.f, a1c1 = 0.f, a1c2 = 0.f, a1c3 = 0.f;
    float a1c4 = 0.f, a1c5 = 0.f, a1c6 = 0.f, a1c7 = 0.f;

    uint32_t a0 = smem_a + row0 * K * 4;
    uint32_t a1 = a0 + K * 4;
    const uint32_t stride = (K + PAD) * 4;
    uint32_t b = smem_b + col0 * stride;

    for (uint32_t k = 0; k < K; k++) {
      const float ra0 = bits_to_float(load32_shared(a0));
      const float ra1 = bits_to_float(load32_shared(a1));
      uint32_t baddr = b;
      const float rb0 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb1 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb2 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb3 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb4 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb5 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb6 = bits_to_float(load32_shared(baddr)); baddr += stride;
      const float rb7 = bits_to_float(load32_shared(baddr));
      a0c0 += ra0 * rb0; a0c1 += ra0 * rb1; a0c2 += ra0 * rb2; a0c3 += ra0 * rb3;
      a0c4 += ra0 * rb4; a0c5 += ra0 * rb5; a0c6 += ra0 * rb6; a0c7 += ra0 * rb7;
      a1c0 += ra1 * rb0; a1c1 += ra1 * rb1; a1c2 += ra1 * rb2; a1c3 += ra1 * rb3;
      a1c4 += ra1 * rb4; a1c5 += ra1 * rb5; a1c6 += ra1 * rb6; a1c7 += ra1 * rb7;
      a0 += 4; a1 += 4; b += 4;
    }

    const uint32_t c0 = row0 * N + col0;
    args->C[c0 + 0] = a0c0; args->C[c0 + 1] = a0c1;
    args->C[c0 + 2] = a0c2; args->C[c0 + 3] = a0c3;
    args->C[c0 + 4] = a0c4; args->C[c0 + 5] = a0c5;
    args->C[c0 + 6] = a0c6; args->C[c0 + 7] = a0c7;
    const uint32_t c1 = c0 + N;
    args->C[c1 + 0] = a1c0; args->C[c1 + 1] = a1c1;
    args->C[c1 + 2] = a1c2; args->C[c1 + 3] = a1c3;
    args->C[c1 + 4] = a1c4; args->C[c1 + 5] = a1c5;
    args->C[c1 + 6] = a1c6; args->C[c1 + 7] = a1c7;
  }
}
