// SMEM matmul, register-LEAN: 1 output/thread, no unrolling, minimal live temps.
// Target <= 31 distinct registers so 8 warps * regs <= 256 physregs (RTL-legal).
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
  mu_fence_smem(); // cross-core SMEM visibility before the barrier (RTL: barrier alone doesn't flush)
  mu_barrier(0, num_warps); // ID 0 = cross-core/cluster barrier (ID 1 doesn't sync across cores)

  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    const uint32_t row = idx / N;
    const uint32_t col = idx % N;
    uint32_t a_addr = smem_a + row * K * 4;
    uint32_t b_addr = smem_b + col * (K + PAD) * 4;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
      acc += bits_to_float(load32_shared(a_addr)) * bits_to_float(load32_shared(b_addr));
      a_addr += 4;
      b_addr += 4;
    }
    args->C[idx] = acc;
  }
}
