void solution(void) {
  // Step 1: Configure execution (fp8 e4m3, BF16 output, weight-stationary)
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      1, 1, 0, 0, false, 0, 0, 3, 0);

  // Step 2: Load scale factors for A (sel=0) and B (sel=1)
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // Step 3: Configure both mvin ports with the same DRAM row stride
  // Port 0 (mvin)  → A tiles
  // Port 1 (mvin2) → B tiles
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

  // Step 4: Interleaved A (mvin/port0) and B (mvin2/port1) tile loads.
  // Within each k-slice, issue all A-row tiles on port 0 then all B-col tiles
  // on port 1 back-to-back.  The two independent DMA FIFOs allow the hardware
  // to overlap their memory transactions.
  for (int k = 0; k < tiles_K; k++) {
    for (int i = 0; i < tiles_I; i++) {
      elem_t *a_ptr = ((elem_t *)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t a_sp = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *)a_ptr, a_sp, DIM, DIM);
    }
    for (int j = 0; j < tiles_J; j++) {
      elem_t *b_ptr = ((elem_t *)B_in) + j * DIM * MATMUL_K + k * DIM;
      uint32_t b_sp = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin2((void *)b_ptr, b_sp, DIM, DIM);
    }
  }

  // Step 5: Configure output stride and requantizer
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors,
      tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Step 6: Fused tiled weight-stationary matmul (mx_loop_spad_marker is
  // one-shot and is set by the LOOP_WS_CONFIG_SPAD_AB instruction that
  // gemmini_loop_ws_spad emits internally before LOOP_WS).
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
      a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

  // Step 7: Drain BF16 results from mx_smem to DRAM, then fence.
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
