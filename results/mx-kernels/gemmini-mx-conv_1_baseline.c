// AUTO-GENERATED baseline MX patch-embed conv kernel (C=2 H=32 k=4 OC=32).
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

  gemmini_config_ld(MATMUL_K * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + j * DIM * MATMUL_K + k * DIM),
                            b_base + (j * tiles_K + k) * DIM, DIM, DIM);

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                       SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
