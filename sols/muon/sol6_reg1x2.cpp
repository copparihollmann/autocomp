// Register-FRUGAL matmul tile: each thread computes 1 row x 2 cols (2 accumulators).
// Smallest useful tile — reuses each A[k] SMEM load across 2 output columns. Minimal
// register growth over the lean baseline, to stay well inside the 32-reg/warp RTL budget.
// Self-contained (inline __builtin_bit_cast).
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
    store_shared(smem_a + i * 4, 0, __builtin_bit_cast(uint32_t, args->A[i]));
  }
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    const uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * (K + PAD) + k) * 4, 0, __builtin_bit_cast(uint32_t, args->B[i]));
  }
  mu_barrier(1, num_warps);

  const uint32_t tiles = (M * N) / 2;
  const uint32_t tiles_per_row = N / 2;
  const uint32_t stride = (K + PAD) * 4;
  for (uint32_t t = tid_in_threadblock; t < tiles; t += threads_per_threadblock) {
    const uint32_t row = t / tiles_per_row;
    const uint32_t col0 = (t % tiles_per_row) * 2;
    float c0 = 0.f, c1 = 0.f;
    uint32_t a = smem_a + row * K * 4;
    uint32_t b = smem_b + col0 * stride;
    for (uint32_t k = 0; k < K; k++) {
      const float ra = __builtin_bit_cast(float, load32_shared(a));
      c0 += ra * __builtin_bit_cast(float, load32_shared(b));
      c1 += ra * __builtin_bit_cast(float, load32_shared(b + stride));
      a += 4; b += 4;
    }
    const uint32_t c = row * N + col0;
    args->C[c + 0] = c0; args->C[c + 1] = c1;
  }
}
