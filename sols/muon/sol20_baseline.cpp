// Baseline: single-output-tile MX-Gemmini FP8 matmul, full (bf16) output.
// The warps orchestrate the accelerator; mxgemm_lib.hpp issues the MMIO command stream:
//   configure -> stage e8m0 scales into SMEM -> loop_ws (accelerator DMAs A/B from DRAM)
//   -> loop_ws (compute) -> SIMT move-out of C from SMEM to GMEM.
//
// Keep everything inside kernel_body: the harness's unfenced-response fallback keeps only
// the text from `void kernel_body` onward, so declarations above it would be dropped.
void kernel_body(void *raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  constexpr GemmConfig GEMM_CFG{
      .TILE_M = MATMUL_M,
      .TILE_N = MATMUL_N,
      .TILE_K = MATMUL_K,
      .DATATYPE = GemmDatatype::FP8,
      .QUANT_OUTPUT = false,
  };
  auto *arg = reinterpret_cast<KernelArgs *>(raw_arg);
  mxgemm<GEMM_CFG>(arg->M, arg->N, arg->K, arg->C, tid_in_threadblock,
                   threads_per_threadblock, threadblock_id);
}
