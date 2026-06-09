// Conv patch-embed: weights staged in SMEM (read 16x each), input streamed
// from global once per (patch, channel-row). 2 OC outputs per thread to halve
// weight reads. No div/mod in stage loop.
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

  // Stage weights [OC][K] contiguous; PAD each OC row by 16 floats.
  const uint32_t PAD = 16;
  const uint32_t w_stride = (K + PAD) * 4;
  for (uint32_t oc = 0; oc < OC; oc++) {
    const __global float* src = a->w + oc * K;
    const uint32_t dst = oc * w_stride;
    for (uint32_t k = tid_in_threadblock; k < K; k += threads_per_threadblock) {
      store_shared(dst + k * 4, 0, float_to_bits(src[k]));
    }
  }
  mu_barrier(1, num_warps);

  // Thread = patch (lane) x oc pair. acc over c, kh, kw with direct walk.
  const uint32_t lane = tid_in_threadblock % P;
  const uint32_t group = tid_in_threadblock / P;   // 0..7
  const uint32_t oh = lane / OW, ow = lane % OW;

  uint32_t w0 = group * w_stride;
  uint32_t w1 = (group + 8) * w_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint32_t c = 0; c < C; c++) {
    const __global float* col = a->in + (c * H + oh * S) * W + ow * S;
    for (uint32_t kh = 0; kh < KH; kh++) {
      const __global float* row = col + kh * W;
      #pragma unroll 16
      for (uint32_t kw = 0; kw < KW; kw++) {
        const float x = row[kw];
        acc0 += x * bits_to_float(load32_shared(w0));
        acc1 += x * bits_to_float(load32_shared(w1));
        w0 += 4; w1 += 4;
      }
    }
  }
  a->out[group * P + lane] = acc0;
  a->out[(group + 8) * P + lane] = acc1;
}
