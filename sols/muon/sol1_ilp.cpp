// Conv direct, ILP: 1 output/thread (baseline structure) but unroll the contiguous
// kw inner loop with independent accumulators + pointer walk, to break the reduction
// chain and reduce index arithmetic. No SMEM (conv has little reuse).
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t C = a->C, H = a->H, W = a->W;
  const uint32_t OC = a->OC, KH = a->KH, KW = a->KW, S = a->stride;
  const uint32_t OH = a->OH, OW = a->OW;
  const uint32_t total = OC * OH * OW;

  for (uint32_t idx = tid_in_threadblock; idx < total; idx += threads_per_threadblock) {
    const uint32_t oc = idx / (OH * OW);
    const uint32_t oh = (idx / OW) % OH;
    const uint32_t ow = idx % OW;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (uint32_t c = 0; c < C; c++) {
      for (uint32_t kh = 0; kh < KH; kh++) {
        const __global float* xr = a->in + (c * H + oh * S + kh) * W + ow * S;
        const __global float* wr = a->w + ((oc * C + c) * KH + kh) * KW;
        uint32_t kw = 0;
        for (; kw + 4 <= KW; kw += 4) {
          acc0 += xr[kw + 0] * wr[kw + 0];
          acc1 += xr[kw + 1] * wr[kw + 1];
          acc2 += xr[kw + 2] * wr[kw + 2];
          acc3 += xr[kw + 3] * wr[kw + 3];
        }
        for (; kw < KW; kw++) acc0 += xr[kw] * wr[kw];
      }
    }
    a->out[idx] = (acc0 + acc1) + (acc2 + acc3);
  }
}
