// SMEM-tiled matmul with 4-output register tiling: each thread computes 4 adjacent
// outputs C[row, 4c..4c+3], reusing one A read across 4 columns.
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

  const uint32_t quarter = (M * N) / 4;
  for (uint32_t q = tid_in_threadblock; q < quarter; q += threads_per_threadblock) {
    const uint32_t row = q / (N / 4);
    const uint32_t col0 = (q % (N / 4)) * 4;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const uint32_t a_base = smem_a + row * K * 4;
    const uint32_t b0 = smem_b + (col0 + 0) * (K + PAD) * 4;
    const uint32_t b1 = smem_b + (col0 + 1) * (K + PAD) * 4;
    const uint32_t b2 = smem_b + (col0 + 2) * (K + PAD) * 4;
    const uint32_t b3 = smem_b + (col0 + 3) * (K + PAD) * 4;

    #pragma unroll 4
    for (uint32_t k = 0; k < K; k++) {
      const float a = bits_to_float(load32_shared(a_base + k * 4));
      acc0 += a * bits_to_float(load32_shared(b0 + k * 4));
      acc1 += a * bits_to_float(load32_shared(b1 + k * 4));
      acc2 += a * bits_to_float(load32_shared(b2 + k * 4));
      acc3 += a * bits_to_float(load32_shared(b3 + k * 4));
    }

    const uint32_t c = row * N + col0;
    args->C[c + 0] = acc0;
    args->C[c + 1] = acc1;
    args->C[c + 2] = acc2;
    args->C[c + 3] = acc3;
  }
}
