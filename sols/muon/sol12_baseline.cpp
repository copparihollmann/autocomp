// Row-parallel attention (seq>128 flash regime), nounroll to keep the cyclotron
// distinct-register count under the 256/8-warp wall (long loops unroll -> more distinct rd's).
// Original note (seq=128): Dense flash/naive at seq=128 oversubscribe the
// 256-physreg pool (8 warps share one core: flash ~42/warp*... > 256). Fix: drive each phase by an
// outer loop over the seq=128 ROWS (one thread per row). With tpb=256 threads but only 128 rows, the
// loop bound idles warps 4-7 (their tid >= seq -> 0 iterations -> ~harness-floor regs), so only 4
// warps are register-active -> 4*~work + 4*~16 < 256 pool. No predication tricks; the loop bound does
// it (same mechanism that idles softmax warps). Scores live in global scratch (not registers).
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;
  const uint32_t d = a->d;
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;

  // Phase 1: S[i][j] = scale * Q[i]·K[j]  (one thread per row i)
  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {
    const __global float* qi = a->Q + i * d;
    for (uint32_t j = 0; j < seq; j++) {
      const __global float* kj = a->K + j * d;
      float acc = 0.0f;
      #pragma GCC unroll 1
      for (uint32_t k = 0; k < d; k++) acc += qi[k] * kj[k];
      a->scratch[i * seq + j] = acc * 0.125f;
    }
  }
  mu_barrier(1, nw);

  // Phase 2: row softmax in place
  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {
    __global float* sp = a->scratch + i * seq;
    float m = sp[0];
    #pragma GCC unroll 1
    for (uint32_t j = 1; j < seq; j++) { if (sp[j] > m) m = sp[j]; }
    float sum = 0.0f;
    #pragma GCC unroll 1
    for (uint32_t j = 0; j < seq; j++) { const float e = mu_exp(sp[j] - m); sp[j] = e; sum += e; }
    const float inv = 1.0f / sum;
    #pragma GCC unroll 1
    for (uint32_t j = 0; j < seq; j++) sp[j] *= inv;
  }
  mu_barrier(1, nw);

  // Phase 3: O[i][k] = sum_j P[i][j] * V[j][k]  (one thread per row i)
  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {
    const __global float* sp = a->scratch + i * seq;
    for (uint32_t k = 0; k < d; k++) {
      const __global float* vk = a->V + k;
      float acc = 0.0f;
      #pragma GCC unroll 1
      for (uint32_t j = 0; j < seq; j++) acc += sp[j] * vk[j * d];
      a->O[i * d + k] = acc;
    }
  }
}
