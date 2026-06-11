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

  // PAD=16 floats per row to avoid SMEM bank conflicts
  const uint32_t PAD = 16;

  // smem_a: A stored row-major at offset 0
  // smem_b: B stored transposed (col-major) with padding at offset M*K*4
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;

  // Cooperatively load A (row-major) into SMEM
  // A[i] maps directly: element i -> smem_a + i*4
  const __global float* ga = args->A;
  uint32_t mk = M * K;
  for (uint32_t i = tid_in_threadblock; i < mk; i += threads_per_threadblock) {
    uint32_t addr = smem_a + i * 4;
    store_shared(addr, 0, __builtin_bit_cast(uint32_t, ga[i]));
  }

  // Cooperatively load B transposed+padded into SMEM
  // B[k][col] -> smem_b + (col*(K+PAD) + k)*4
  const __global float* gb = args->B;
  uint32_t kn = K * N;
  uint32_t kpad = K + PAD;
  for (uint32_t i = tid_in_threadblock; i < kn; i += threads_per_threadblock) {
    uint32_t k   = i / N;
    uint32_t col = i % N;
    uint32_t addr = smem_b + (col * kpad + k) * 4;
    store_shared(addr, 0, __builtin_bit_cast(uint32_t, gb[i]));
  }

  // Synchronize all 8 warps (2 cores x 4 warps) before reading SMEM
  mu_barrier(1, 8);

  const uint32_t stride_b = kpad * 4;
  uint32_t mn = M * N;
  __global float* gc = args->C;

  for (uint32_t idx = tid_in_threadblock; idx < mn; idx += threads_per_threadblock) {
    uint32_t row = idx / N;
    uint32_t col = idx % N;
    float acc = 0.0f;
    uint32_t a_base = smem_a + row * K * 4;
    uint32_t b_base = smem_b + col * stride_b;
    for (uint32_t k = 0; k < K; k++) {
      float a_val = __builtin_bit_cast(float, load32_shared(a_base + k * 4));
      float b_val = __builtin_bit_cast(float, load32_shared(b_base + k * 4));
      acc += a_val * b_val;
    }
    gc[idx] = acc;
  }
}
