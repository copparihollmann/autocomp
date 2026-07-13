// Baseline: single 64x64 output tile, K=512 accumulated over 8 K-tiles of TILE_K=64.
// Same orchestration as sol20 but TILE_K is pinned to 64 (NOT MATMUL_K), so the driver
// runs the software-pipelined K-loop: 8 loop_ws compute calls, each ex_accumulate-ing a
// 64-K slab into C in SMEM, with a gemmini_fence() per tile. This makes the accelerator's
// busy time (8x prob20's) reach total cycles, so KCMP/KDMA are identifiable.
//
// The SMEM layout is identical to prob20's proven, RTL-passing 64x64 tile: A occupies
// scratchpad rows 0..256, C lives at SPAD_DEST=256 (byte 4096) just past it -- no overlap.
//
// Keep everything inside kernel_body: the harness's unfenced-response fallback keeps only
// the text from `void kernel_body` onward, so declarations above it would be dropped.
void kernel_body(void *raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  constexpr GemmConfig GEMM_CFG{
      .TILE_M = MATMUL_M,
      .TILE_N = MATMUL_N,
      .TILE_K = 64, // fixed: 8 K-tiles for K=512
      .DATATYPE = GemmDatatype::FP8,
      .QUANT_OUTPUT = false,
  };
  auto *arg = reinterpret_cast<KernelArgs *>(raw_arg);
  mxgemm<GEMM_CFG>(arg->M, arg->N, arg->K, arg->C, tid_in_threadblock,
                   threads_per_threadblock, threadblock_id);
}
