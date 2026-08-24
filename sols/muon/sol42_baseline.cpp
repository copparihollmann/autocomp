// Gemma-2 decoder-layer norm/residual wiring: one token-row per thread, grid-stride.
// Executes the exact 4-norm "sandwich" topology from Gemma2DecoderLayer.forward, using a global
// scratch row (tmp) so each RMSNorm can reduce over the full row before scaling. Sublayers are
// per-channel scales (wa/wm). eps=1e-6, norm applies (1.0 + weight).
static inline float row_inv_rms(__global const float* r, uint32_t D) {
  float ss = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint32_t j = 0; j < D; j++) { const float v = r[j]; ss += v * v; }
  return mu_rsqrt(ss / (float)D + 1.0e-6f);
}
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t D = a->D;
  for (uint32_t r = tid_in_threadblock; r < a->R; r += threads_per_threadblock) {
    const uint32_t o = r * D;
    __global float* x   = a->x   + o;   // input residual stream (preserved)
    __global float* tmp = a->tmp + o;   // scratch
    __global float* out = a->out + o;

    // NORM 1 (input_layernorm) + self_attn stub -> tmp
    float inv = row_inv_rms(x, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      tmp[j] = (x[j] * inv * (1.0f + a->w_in[j])) * a->wa[j];

    // NORM 2 (post_attention_layernorm) on tmp, then RESIDUAL 1 (+ x) -> out (= x1)
    inv = row_inv_rms(tmp, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      out[j] = x[j] + (tmp[j] * inv * (1.0f + a->w_pa[j]));

    // NORM 3 (pre_feedforward_layernorm) on x1 (=out) + mlp stub -> tmp
    inv = row_inv_rms(out, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      tmp[j] = (out[j] * inv * (1.0f + a->w_pf[j])) * a->wm[j];

    // NORM 4 (post_feedforward_layernorm) on tmp, then RESIDUAL 2 (+ x1=out) -> out
    inv = row_inv_rms(tmp, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      out[j] = out[j] + (tmp[j] * inv * (1.0f + a->w_pff[j]));
  }
}
