// Shared-memory matmul: stage all of A and B (16 KB each) in SMEM, then each
// thread computes its outputs out of SMEM. One global pass; one barrier.
static inline float bits_to_float(uint32_t b) {
  union { uint32_t u; float f; } c;
  c.u = b;
  return c.f;
}
static inline uint32_t float_to_bits(float f) {
  union { uint32_t u; float f; } c;
  c.f = f;
  return c.u;
}

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

  // SMEM layout: A[64*64] at 0, B[64*64] after - 32 KiB total.
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;

  // Cooperative load (offset operand of store_shared must be an immediate).
  for (uint32_t i = tid_in_threadblock; i < M * K; i += threads_per_threadblock) {
    store_shared(smem_a + i * 4, 0, float_to_bits(args->A[i]));
  }
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    const uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * (K + 16) + k) * 4, 0, float_to_bits(args->B[i]));
  }
  mu_barrier(1, num_warps);

  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    const uint32_t row = idx / N;
    const uint32_t col = idx % N;
    float acc = 0.0f;
    const uint32_t a_base = smem_a + row * K * 4;
    uint32_t b_addr = smem_b + col * (K + 16) * 4;
    #pragma unroll 8
    for (uint32_t k = 0; k < K; k++) {
      acc += bits_to_float(load32_shared(a_base + k * 4)) *
             bits_to_float(load32_shared(b_addr + k * 4));
    }
    args->C[idx] = acc;
  }
}
