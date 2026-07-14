// fp32 <-> u32 bit-cast for the SMEM word API (store_shared/load32_shared move raw u32).
static inline uint32_t f2b(float f){ union{float f; uint32_t u;} v; v.f=f; return v.u; }
static inline float    b2f(uint32_t u){ union{float f; uint32_t u;} v; v.u=u; return v.f; }

// Hand-optimized decode GEMV: stage x[K] into SMEM once (it is reused across ALL N outputs),
// then each thread's K-dot reads x from SMEM instead of global. The naive baseline reads x[k]
// from global memory N times (once per output); this reads it once. W is read once either way
// (unavoidable, each element used once). Cyclotron (issue-bound) won't score this win — measure
// on RTL. unroll(disable) keeps register pressure under the 256-phys-reg wall.
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t K = a->K;
  const uint32_t SMEM_X = 0;  // x[K] fp32 at SMEM base 0
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;  // warps in the threadblock

  // Cooperatively stage x[K] from global into SMEM.
  #pragma clang loop unroll(disable)
  for (uint32_t k = tid_in_threadblock; k < K; k += threads_per_threadblock) {
    store_shared(SMEM_X + k * 4, 0, f2b(a->x[k]));
  }
  mu_barrier(1, nw);  // all warps: x fully staged before any dot reads it

  // Each thread computes one output; dot product reads x from SMEM.
  for (uint32_t n = tid_in_threadblock; n < a->N; n += threads_per_threadblock) {
    const uint32_t base = n * K;
    float acc = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < K; k++) {
      acc += a->W[base + k] * b2f(load32_shared(SMEM_X + k * 4));
    }
    a->out[n] = acc;
  }
}
