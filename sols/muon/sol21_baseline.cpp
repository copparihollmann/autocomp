void kernel_body(void *raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  constexpr GemmConfig GEMM_CFG{.TILE_M=128,.TILE_N=128,.TILE_K=256,
      .DATATYPE=GemmDatatype::FP8,.QUANT_OUTPUT=false};
  auto *arg = reinterpret_cast<KernelArgs *>(raw_arg);
  mxgemm<GEMM_CFG>(arg->M, arg->N, arg->K, arg->C, tid_in_threadblock,
                   threads_per_threadblock, threadblock_id);
}
