// Lean SMEM matmul + ILP: the K-reduction acc += a*b is a sequential dependency
// chain (each acc depends on the previous). Split into 4 independent accumulators
// over interleaved k so the FMAs issue back-to-back (same trick that gave softmax
// its big win). Still 1 output/thread -> low register count, RTL-legal.
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

  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    const uint32_t row = idx / N;
    const uint32_t col = idx % N;
    uint32_t aa = smem_a + row * K * 4;
    uint32_t bb = smem_b + col * (K + PAD) * 4;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t k = 0;
    for (; k + 4 <= K; k += 4) {
      a0 += bits_to_float(load32_shared(aa + 0))  * bits_to_float(load32_shared(bb + 0));
      a1 += bits_to_float(load32_shared(aa + 4))  * bits_to_float(load32_shared(bb + 4));
      a2 += bits_to_float(load32_shared(aa + 8))  * bits_to_float(load32_shared(bb + 8));
      a3 += bits_to_float(load32_shared(aa + 12)) * bits_to_float(load32_shared(bb + 12));
      aa += 16; bb += 16;
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k++) {
      acc += bits_to_float(load32_shared(aa)) * bits_to_float(load32_shared(bb));
      aa += 4; bb += 4;
    }
    args->C[idx] = acc;
  }
}
