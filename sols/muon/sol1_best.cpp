// Register-FRUGAL legal conv = im2col + SMEM matmul. The naive 6D direct conv keeps
// ~11 struct dims + 6 loop indices live at once -> busts the 32-reg/warp RTL budget
// (its baseline asserts globalOverSubscription). Here the work is split into phases, and
// each phase is tightly scoped ({...}) so its live register set holds only what it uses
// (the gather phase is the register peak; keeping matmul-only values out of its scope
// drops the peak below the budget):
//   (1) stage W[OC][CKK] into SMEM (flat copy),
//   (2) gather the im2col patch matrix patch[P][CKK] into SMEM,
//   (3) frugal dot-product matmul OUT[oc][p] = sum_ckk W[oc][ckk]*patch[p][ckk].
// CKK = C*KH*KW, P = OH*OW. patch[p][ckk] = in[c][(p/OW)*stride+kh][(p%OW)*stride+kw].
// Self-contained (inline __builtin_bit_cast).
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t CKK = a->C * a->KH * a->KW;
  const uint32_t smem_w = a->OH * a->OW * CKK * 4; // W staged AFTER patches

  // (1) stage W flat into SMEM[smem_w..].
  for (uint32_t i = tid_in_threadblock; i < a->OC * CKK; i += threads_per_threadblock) {
    store_shared(smem_w + i * 4, 0, __builtin_bit_cast(uint32_t, a->w[i]));
  }

  // (2) gather im2col patches into SMEM[0..] (store target is the immediate 0, so no
  // smem-base register is live in this register-peak loop). i = p*CKK + ckk.
  // in_idx = c*(H*Wd) + ((p/OW)*stride+kh)*Wd + (p%OW)*stride + kw.
  {
    const uint32_t KH = a->KH, KW = a->KW, OW = a->OW;
    const uint32_t Wd = a->W, HWd = a->H * Wd, stride = a->stride;
    const uint32_t total = a->OH * OW * CKK;
    for (uint32_t i = tid_in_threadblock; i < total; i += threads_per_threadblock) {
      const uint32_t p = i / CKK;
      const uint32_t ckk = i % CKK;
      const uint32_t in_idx = (ckk / (KH * KW)) * HWd
                            + ((p / OW) * stride + (ckk / KW) % KH) * Wd
                            + (p % OW) * stride + ckk % KW;
      store_shared(i * 4, 0, __builtin_bit_cast(uint32_t, a->in[in_idx]));
    }
  }
  // Cross-core SMEM staging: the threadblock spans MU_NUM_CORES cores, so warps on one core
  // read patches/W staged by warps on another. mu_barrier syncs execution but does NOT flush
  // SMEM write buffers across cores; fence.s makes the staged writes visible before any core
  // proceeds past the barrier to read them. Without it, RTL reads stale SMEM (cyclotron, which
  // models SMEM as instantly coherent, does not catch this).
  mu_fence_smem();
  mu_barrier(0, threads_per_threadblock / MU_NUM_THREADS); // ID 0 = cross-core/cluster barrier

  // (3) frugal matmul: OUT[oc*P + p] = sum_ckk W[oc][ckk] * patch[p][ckk].
  {
    const uint32_t P = a->OH * a->OW;
    for (uint32_t idx = tid_in_threadblock; idx < a->OC * P; idx += threads_per_threadblock) {
      uint32_t wa = smem_w + (idx / P) * CKK * 4; // W base (after patches)
      uint32_t pa = (idx % P) * CKK * 4;          // patch base (SMEM offset 0)
      float acc = 0.0f;
      for (uint32_t k = 0; k < CKK; k++) {
        acc += __builtin_bit_cast(float, load32_shared(wa)) *
               __builtin_bit_cast(float, load32_shared(pa));
        wa += 4; pa += 4;
      }
      a->out[idx] = acc;
    }
  }
}
