void solution(void) {
  const int tiles_I = MATMUL_M / DIM;
  const int tiles_J = MATMUL_N / DIM;
  const int tiles_K = MATMUL_K / DIM;

  const uint32_t a_base   = 0;
  const uint32_t b_base   = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  const int      SPAD_DEST = 128;

  // Step 1: Configure execution (fp8 in, BF16 out, weight-stationary)
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      1, 1, 0, 0, false, 0, 0, 3, 0);

  // Step 2: Load A and B scale factors
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // Step 3: Configure a single load port (port 0) for all mvin calls.
  // Both A and B have stride MATMUL_K * sizeof(elem_t), so one config suffices.
  // Eliminates the redundant port-1 gemmini_extended3_config_ld instruction
  // (Spike executes all RoCC instructions sequentially; mvin2 parallelism is
  //  not realized, but the port-1 config instruction does cost dispatch cycles).
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);

  // Step 4: MVIN all A and B tiles through port 0.
  // Interleaved k-slice order: for each k, load all i-rows of A then all
  // j-columns of B, improving DRAM access locality within each k-slice.
  for (int k = 0; k < tiles_K; k++) {
    for (int i = 0; i < tiles_I; i++)
      gemmini_extended_mvin(
          (void*)(((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM),
          a_base + (i * tiles_K + k) * DIM,
          DIM, DIM);
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin(
          (void*)(((elem_t*)B_in) + j * DIM * MATMUL_K + k * DIM),
          b_base + (j * tiles_K + k) * DIM,
          DIM, DIM);
  }

  // Step 5: Configure output stride and requantizer bounds
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors,
      tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Step 6: Fused weight-stationary tiled matmul over scratchpad
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
      a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

  // Step 7: Drain BF16 results from mx_smem to DRAM, then fence
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
