// Baseline RoPE: one (position,head) row per thread; grid-stride over rows.
// Paired form: for i in [0,HALF), process the (i, i+HALF) pair sharing one cos/sin
//   (cos/sin are duplicated across halves in the cache):
//     out[i]      = x[i]*c - x[i+HALF]*s
//     out[i+HALF] = x[i+HALF]*c + x[i]*s
// unroll(disable): the 128-wide row loop must NOT be unrolled or it exceeds the 256 physical
// registers (8 warps x 32) -- RTL Rename.scala:123 asserts. Tight loop reuses registers.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols, half = a->half;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < half; i++) {
      const float x1 = a->x[base + i];
      const float x2 = a->x[base + i + half];
      const float c  = a->cosc[base + i];
      const float s  = a->sinc[base + i];
      a->out[base + i]        = x1 * c - x2 * s;
      a->out[base + i + half] = x2 * c + x1 * s;
    }
  }
}
