// SMEM-tiled matmul + 1x2 register tile: each thread computes TWO adjacent output columns,
// loading each A value once and reusing it across both columns (halves A-loads + row address
// arithmetic). Issue-bound model rewards the lower instruction count — if it stays <=256 regs.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t M = args->M;
  const uint32_t N = args->N;
  const uint32_t K = args->K;
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;

  const uint32_t PAD = 16;
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;
  const uint32_t kpad = K + PAD;

  const __global float* ga = args->A;
  for (uint32_t i = tid_in_threadblock; i < M * K; i += threads_per_threadblock) {
    store_shared(smem_a + i * 4, 0, __builtin_bit_cast(uint32_t, ga[i]));
  }
  const __global float* gb = args->B;
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * kpad + k) * 4, 0, __builtin_bit_cast(uint32_t, gb[i]));
  }

  mu_barrier(0, nw);

  const uint32_t stride_b = kpad * 4;
  __global float* gc = args->C;
  const uint32_t halfN = N / 2;            // each thread owns 2 columns: col0 and col0+halfN
  const uint32_t tiles = M * halfN;        // M rows x (N/2) column-pairs
  for (uint32_t idx = tid_in_threadblock; idx < tiles; idx += threads_per_threadblock) {
    uint32_t row = idx / halfN;
    uint32_t c0  = idx % halfN;
    uint32_t c1  = c0 + halfN;
    uint32_t a_base = smem_a + row * K * 4;
    uint32_t b0 = smem_b + c0 * stride_b;
    uint32_t b1 = smem_b + c1 * stride_b;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
      float a_val = __builtin_bit_cast(float, load32_shared(a_base + k * 4));
      acc0 += a_val * __builtin_bit_cast(float, load32_shared(b0 + k * 4));
      acc1 += a_val * __builtin_bit_cast(float, load32_shared(b1 + k * 4));
    }
    gc[row * N + c0] = acc0;
    gc[row * N + c1] = acc1;
  }
}
