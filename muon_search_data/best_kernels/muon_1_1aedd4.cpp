static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t CKK = a->C * a->KH * a->KW;
  const uint32_t smem_w = a->OH * a->OW * CKK * 4;

  // (1) stage W flat into SMEM[smem_w..].
  for (uint32_t i = tid_in_threadblock; i < a->OC * CKK; i += threads_per_threadblock) {
    store_shared(smem_w + i * 4, 0, __builtin_bit_cast(uint32_t, a->w[i]));
  }

  // (2) gather im2col patches into SMEM[0..].
  {
    const uint32_t total = a->OH * a->OW * CKK;
    for (uint32_t i = tid_in_threadblock; i < total; i += threads_per_threadblock) {
      const uint32_t p   = i / CKK;
      const uint32_t ckk = i % CKK;
      const uint32_t c   = ckk >> 8;
      const uint32_t kh  = (ckk >> 4) & 15;
      const uint32_t kw  = ckk & 15;
      const uint32_t oh  = p >> 2;
      const uint32_t ow  = p & 3;
      const uint32_t row = (oh << 4) + kh;
      const uint32_t in_idx = (c << 12) + (row << 6) + (ow << 4) + kw;
      store_shared(i * 4, 0, __builtin_bit_cast(uint32_t, a->in[in_idx]));
    }
  }

  mu_fence_smem();
  mu_barrier(0, threads_per_threadblock / MU_NUM_THREADS);

  // (3) 2-OC x 2-patch tiling: each thread computes 4 outputs.
  // OC=16, P=OH*OW=16, P/2=8, OC/2=8 => 64 total units for 128 threads,
  // each thread handles one idx in [0,64): idx = oc_pair*8 + pair_p
  // oc_pair = idx >> 3, pair_p = idx & 7
  // oc0 = oc_pair*2, oc1 = oc_pair*2+1
  // base_p = pair_p*2
  {
    const uint32_t P      = a->OH * a->OW;
    const uint32_t half_P = P >> 1;                    // 8
    const uint32_t total_units = (a->OC >> 1) * half_P; // 64

    for (uint32_t idx = tid_in_threadblock; idx < total_units; idx += threads_per_threadblock) {
      const uint32_t oc_pair = idx / half_P;
      const uint32_t pair_p  = idx % half_P;
      const uint32_t oc0     = oc_pair * 2;
      const uint32_t oc1     = oc0 + 1;
      const uint32_t base_p  = pair_p * 2;

      uint32_t wa0 = smem_w + oc0 * CKK * 4;
      uint32_t wa1 = smem_w + oc1 * CKK * 4;
      uint32_t pa0 = base_p * CKK * 4;
      uint32_t pa1 = (base_p + 1) * CKK * 4;

      float acc00 = 0.0f;
      float acc01 = 0.0f;
      float acc10 = 0.0f;
      float acc11 = 0.0f;

      for (uint32_t k = 0; k < CKK; k++) {
        float w0 = __builtin_bit_cast(float, load32_shared(wa0));
        float w1 = __builtin_bit_cast(float, load32_shared(wa1));
        float p0 = __builtin_bit_cast(float, load32_shared(pa0));
        float p1 = __builtin_bit_cast(float, load32_shared(pa1));
        acc00 += w0 * p0;
        acc01 += w0 * p1;
        acc10 += w1 * p0;
        acc11 += w1 * p1;
        wa0 += 4;
        wa1 += 4;
        pa0 += 4;
        pa1 += 4;
      }

      a->out[oc0 * P + base_p]     = acc00;
      a->out[oc0 * P + base_p + 1] = acc01;
      a->out[oc1 * P + base_p]     = acc10;
      a->out[oc1 * P + base_p + 1] = acc11;
    }
  }
}
