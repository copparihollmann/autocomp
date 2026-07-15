// Coalesced ResAdd: flat grid-stride so the 16 lanes of a warp read CONSECUTIVE elements
// (one wide/coalesced l0d request per instr instead of 16 strided ones). Cuts l0d in-flight
// request pressure (clears the unbuffered-l0d backpressure assertion) AND is faster.
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t n = a->rows * a->cols;            // flatten
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid_in_threadblock; i < n; i += threads_per_threadblock)
    a->out[i] = a->in[i] + a->res[i];              // lane L -> element base+L => coalesced
}
