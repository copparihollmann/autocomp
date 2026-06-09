// Conv patch-embed as matmul: im2col into padded SMEM.
// OUT[oc, p] = W[oc, :] . PATCH[:, p]; K = C*KH*KW = 768; 16 patches; 16 ocs.
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
  const uint32_t K = C * KH * KW;        // 768
  const uint32_t P = OH * OW;            // 16 patches
  const uint32_t PAD = 16;
  const uint32_t patch_stride = (K + PAD) * 4;
  const uint32_t smem_patch = 0;                    // 16 x 784 floats = 49 KiB
  const uint32_t smem_w = P * patch_stride;         // 16 x 768 floats = 48 KiB

  // im2col: patch p = (oh, ow); element k = (c, kh, kw).
  for (uint32_t i = tid_in_threadblock; i < P * K; i += threads_per_threadblock) {
    const uint32_t p = i / K, k = i % K;
    const uint32_t oh = p / OW, ow = p % OW;
    const uint32_t c = k / (KH * KW), kh = (k / KW) % KH, kw = k % KW;
    const float v = a->in[(c * H + oh * S + kh) * W + ow * S + kw];
    store_shared(smem_patch + p * patch_stride + k * 4, 0, float_to_bits(v));
  }
  // Weights are already [OC][K] contiguous.
  for (uint32_t i = tid_in_threadblock; i < OC * K; i += threads_per_threadblock) {
    store_shared(smem_w + i * 4, 0, float_to_bits(a->w[i]));
  }
  mu_barrier(1, num_warps);

  // 256 outputs, 2 per thread: thread handles (oc, oc+8) x patch p.
  const uint32_t lane = tid_in_threadblock % P;     // patch
  const uint32_t group = tid_in_threadblock / P;    // 0..7 -> oc, oc+8
  const uint32_t pb = smem_patch + lane * patch_stride;
  const uint32_t w0 = smem_w + group * K * 4;
  const uint32_t w1 = smem_w + (group + 8) * K * 4;

  float acc0 = 0.f, acc1 = 0.f;
  #pragma unroll 4
  for (uint32_t k = 0; k < K; k++) {
    const uint32_t off = k * 4;
    const float x = bits_to_float(load32_shared(pb + off));
    acc0 += x * bits_to_float(load32_shared(w0 + off));
    acc1 += x * bits_to_float(load32_shared(w1 + off));
  }
  a->out[group * P + lane] = acc0;
  a->out[(group + 8) * P + lane] = acc1;
}
