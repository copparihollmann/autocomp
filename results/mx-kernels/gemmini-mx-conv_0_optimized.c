void solution(void) {
  // im2col: patch (pi,pj) -> row pi*PATCHES_X+pj; column c*k*k + dy*k + dx
  for (int pi = 0; pi < CONV_H / CONV_KSZ; pi++)
    for (int pj = 0; pj < PATCHES_X; pj++)
      for (int c = 0; c < CONV_C; c++)
        for (int dy = 0; dy < CONV_KSZ; dy++)
          for (int dx = 0; dx < CONV_KSZ; dx++)
            A_buf[pi * PATCHES_X + pj][c * CONV_KSZ * CONV_KSZ + dy * CONV_KSZ + dx] =
                X_in[c][pi * CONV_KSZ + dy][pj * CONV_KSZ + dx];

  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // Configure port 0 (mvin, id=0) for A tiles
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  // Configure port 1 (mvin2, id=1) for B tiles
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

  // Interleaved A and B mvin loop: for each k-slice, issue A tiles on port 1
  // and B tiles on port 2 so both DMA channels proceed in parallel
  for (int k = 0; k < tiles_K; k++) {
    // A tiles for all i-rows at this k-slice → port 1 (k_MVIN)
    for (int i = 0; i < tiles_I; i++)
      gemmini_extended_mvin((void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    // B tiles for all j-columns at this k-slice → port 2 (k_MVIN2), overlaps with A above
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin2((void*)(((elem_t*)B_in) + j * DIM * MATMUL_K + k * DIM),
                             b_base + (j * tiles_K + k) * DIM, DIM, DIM);
  }

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                       SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
