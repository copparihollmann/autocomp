// Baseline MX-Gemmini fp8 64x64 WS matmul kernel (spike functional path).
// The body (between the braces) is what autocomp substitutes into the harness
// and optimizes. It references harness-scope vars (C_hw, scale_factors, tiles_*,
// a_base, b_base, SPAD_DEST) and the MxGen header globals (A_in, B_in,
// A_scales_row, B_scales_col). Output is read back into C_hw as packed bf16.
void solution(void) {
  // Configure execution: weight-stationary, MX requantize to bf16 output.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  // Load per-group scale factors into the scale-factor memory (A=0, B=1).
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  // MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  // Configure store + MX requantizer for the mvout, then run the WS compute loop.
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  // Read the bf16-packed result out of shared memory into C_hw.
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
