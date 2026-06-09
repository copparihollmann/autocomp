// Conv patch-embed: split-K. All 128 threads work the same (oc-pair, patch),
// each accumulating a quarter of K = 768; partials reduced via SMEM.
// SMEM: input 62 KiB (81-float pitch) + weights 48 KiB + partials 4 KiB.
static inline float bits_to_float(uint32_t b) { union { uint32_t u; float f; } c; c.u = b; return c.f; }
static inline uint32_t float_to_bits(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }

static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const uint32_t C = a->C, H = a->H, W = a->W;
  const uint32_t OC = a->OC, KH = a->KH, KW = a->KW, S = a->stride;
  const uint32_t OH = a->OH, OW = a->OW;
  const uint32_t K = C * KH * KW;     // 768
  const uint32_t P = OH * OW;         // 16

  const uint32_t IN_PITCH = W + 17;
  const uint32_t smem_in = 0;
  const uint32_t smem_w = C * H * IN_PITCH * 4;
  const uint32_t smem_p = smem_w + OC * K * 4;   // partials [2 oc][16 p][4 slice][pad]

  for (uint32_t i = tid_in_threadblock; i < C * H * W; i += threads_per_threadblock) {
    const uint32_t row = i / W, col = i % W;
    store_shared(smem_in + (row * IN_PITCH + col) * 4, 0, float_to_bits(a->in[i]));
  }
  for (uint32_t i = tid_in_threadblock; i < OC * K; i += threads_per_threadblock) {
    store_shared(smem_w + i * 4, 0, float_to_bits(a->w[i]));
  }
  mu_barrier(1, num_warps);

  const uint32_t lane = tid_in_threadblock % P;       // patch
  const uint32_t slice = (tid_in_threadblock / P) % 4; // K quarter
  const uint32_t pair = tid_in_threadblock / (P * 4);  // 0..1 -> 8 oc each
  const uint32_t oh = lane / OW, ow = lane % OW;

  const uint32_t kq = K / 4;                  // 192 = 1 channel x 4 kh
  const uint32_t c = slice * kq / (KH * KW);  // one channel per slice quarter? K/4=192 < 256
  // K = C*256; quarter = 192 -> spans channels. Walk linear k = slice*192 .. +192.
  const uint32_t k0 = slice * kq;

  // 8 outputs per thread: oc = pair*8 + i.
  float acc[8];
  for (uint32_t i = 0; i < 8; i++) acc[i] = 0.f;

  for (uint32_t k = k0; k < k0 + kq; k++) {
    const uint32_t ch = k / (KH * KW);
    const uint32_t kh = (k / KW) % KH;
    const uint32_t kw = k % KW;
    const float x = bits_to_float(load32_shared(
        smem_in + ((ch * H + oh * S + kh) * IN_PITCH + ow * S + kw) * 4));
    const uint32_t wb = smem_w + k * 4;
    #pragma unroll 8
    for (uint32_t i = 0; i < 8; i++) {
      acc[i] += x * bits_to_float(load32_shared(wb + ((pair * 8 + i) * K) * 4));
    }
  }

  // partials: [oc][patch][slice] with 1-line pad per (oc,patch)
  const uint32_t pp = 4;
  for (uint32_t i = 0; i < 8; i++) {
    const uint32_t oc = pair * 8 + i;
    store_shared(smem_p + ((oc * P + lane) * (pp + 1) + slice) * 4, 0, float_to_bits(acc[i]));
  }
  mu_barrier(1, num_warps);

  for (uint32_t idx = tid_in_threadblock; idx < OC * P; idx += threads_per_threadblock) {
    const uint32_t base = smem_p + idx * (pp + 1) * 4;
    float sum = bits_to_float(load32_shared(base));
    sum += bits_to_float(load32_shared(base + 4));
    sum += bits_to_float(load32_shared(base + 8));
    sum += bits_to_float(load32_shared(base + 12));
    a->out[idx] = sum;
  }
}
