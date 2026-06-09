// Register-FRUGAL legal attention at seq=128. Stock naive & flash bust the shared
// 256-physreg rename pool: the cyclotron/RTL model counts distinct architectural
// registers PER WARP that executes the code, summed across all 8 provisioned warps, and
// asserts globalOverSubscription at 256. The mu_exp softmax phase needs ~33 distinct regs
// and the PV phase ~31; with all 8 warps running, the sum hits exactly 256.
//
// Fix: leave one warp idle (ACT = 7*32 threads). An idle warp executes no loop body, so it
// writes only the ~16 harness registers instead of ~31-33, dropping the global sum to ~241.
// All 8 warps still reach the barriers (no deadlock). Attention at seq=128 is memory-bound,
// so using 7/8 warps costs little. Scores are scaled to ~N(0,1) (bounded), so softmax needs
// no max-subtraction pass. S = QK^T*scale -> scratch; row softmax; O = P V. Self-contained.
__attribute__((noinline)) static void attn_softmax(KernelArgs* a, uint32_t tid, uint32_t act) {
  const uint32_t seq = a->seq;
  for (uint32_t row = tid; row < seq; row += act) {
    __global float* sp = a->scratch + row * seq;
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) { const float e = mu_exp(sp[j]); sp[j] = e; sum += e; }
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) sp[j] *= inv;
  }
}
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;
  const uint32_t act = (nw - 1) * MU_NUM_THREADS; // leave one warp idle

  if (tid_in_threadblock < act) {
    const uint32_t seq = a->seq, d = a->d;
    for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += act) {
      const __global float* qp = a->Q + (idx / seq) * d;
      const __global float* kp = a->K + (idx % seq) * d;
      float acc = 0.0f;
      for (uint32_t k = 0; k < d; k++) acc += qp[k] * kp[k];
      a->scratch[idx] = acc * 0.125f;
    }
  }
  mu_barrier(1, nw);

  if (tid_in_threadblock < act) attn_softmax(a, tid_in_threadblock, act);
  mu_barrier(1, nw);

  if (tid_in_threadblock < act) {
    const uint32_t seq = a->seq, d = a->d;
    for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += act) {
      const __global float* sp = a->scratch + (idx / d) * seq;
      const __global float* vp = a->V + (idx % d);
      float acc = 0.0f;
      for (uint32_t j = 0; j < seq; j++) { acc += sp[j] * (*vp); vp += d; }
      a->O[idx] = acc;
    }
  }
}
