// Conv patch-embed: input + weights in SMEM, input rows padded to 81 floats
// so lanes hit bank (oh+ow)%4 (4-way spread, conflict-free). 2 OCs per thread.
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

  const uint32_t IN_PITCH = W + 17;   // odd pitch: bank = (oh + ow) % 4
  const uint32_t smem_in = 0;                     // C*H*81*4 = 62 KiB
  const uint32_t smem_w = C * H * IN_PITCH * 4;   // 48 KiB

  for (uint32_t i = tid_in_threadblock; i < C * H * W; i += threads_per_threadblock) {
    const uint32_t row = i / W, col = i % W;
    store_shared(smem_in + (row * IN_PITCH + col) * 4, 0, float_to_bits(a->in[i]));
  }
  for (uint32_t i = tid_in_threadblock; i < OC * K; i += threads_per_threadblock) {
    store_shared(smem_w + i * 4, 0, float_to_bits(a->w[i]));
  }
  mu_barrier(1, num_warps);

  const uint32_t lane = tid_in_threadblock % P;
  const uint32_t group = tid_in_threadblock / P;
  const uint32_t oh = lane / OW, ow = lane % OW;

  uint32_t w0 = smem_w + group * K * 4;
  uint32_t w1 = smem_w + (group + 8) * K * 4;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint32_t c = 0; c < C; c++) {
    uint32_t in_row = smem_in + ((c * H + oh * S) * IN_PITCH + ow * S) * 4;
    for (uint32_t kh = 0; kh < KH; kh++) {
      #pragma unroll 16
      for (uint32_t kw = 0; kw < KW; kw++) {
        const float x = bits_to_float(load32_shared(in_row + kw * 4));
        acc0 += x * bits_to_float(load32_shared(w0));
        acc1 += x * bits_to_float(load32_shared(w1));
        w0 += 4; w1 += 4;
      }
      in_row += IN_PITCH * 4;
    }
  }
  a->out[group * P + lane] = acc0;
  a->out[(group + 8) * P + lane] = acc1;
}
