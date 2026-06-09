// SMEM matmul, register-LEAN: stage full A,B into SMEM (fits at K<=192), 1 output/thread,
// no unroll. Self-contained (inline __builtin_bit_cast — no top-level helpers, so it
// survives the search's code extraction). The naive global-memory baseline is intractable
// at this K (DRAM-bound, >80M cycles); this is the competent starting point to optimize.
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

  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    const uint32_t row = idx / N;
    const uint32_t col = idx % N;
    uint32_t a_addr = smem_a + row * K * 4;
    uint32_t b_addr = smem_b + col * (K + PAD) * 4;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
      acc += __builtin_bit_cast(float, load32_shared(a_addr)) *
             __builtin_bit_cast(float, load32_shared(b_addr));
      a_addr += 4;
      b_addr += 4;
    }
    args->C[idx] = acc;
  }
}
