// Stream the large BSS region big[] (DRAM-bound). Reduce to prevent dead-code elimination.
static inline void kernel_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  (void)raw;
  uint32_t acc = 0;
  for (uint32_t i = tid; i < BW_WORDS; i += tpb) acc += big[i];
  big[tid] = acc;
}
