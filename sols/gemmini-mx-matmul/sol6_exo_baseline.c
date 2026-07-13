// AUTO-GENERATED baseline MX matmul kernel (64x64x64, fp6:e3m2).
void solution(void) {
  {
    int ti = MATMUL_M / DIM / 2, tj = MATMUL_N / DIM / 2, tk = MATMUL_K / DIM;
    uint32_t ab = 0;
    uint32_t bb = BANK_NUM * BANK_ROWS - tk * tj * DIM;
    int SDST = 128;
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);
    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
    gemmini_config_st(OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, ti, tj, tk, 0, 0, 7);
    gemmini_mx_load_lut((uint64_t)B_lut_p, 1, 0);
    gemmini_mx_load_lut((uint64_t)A_lut_p, 1, 1);
    gemmini_config_ld(MATMUL_K * sizeof(elem_t));
    for (int i = 0; i < ti; i++)
      for (int k = 0; k < tk; k++)
        gemmini_extended_mvin((void*)(((elem_t*)A_in_hw) + i*DIM*MATMUL_K + k*DIM), ab + (i*tk + k)*DIM, DIM, DIM);
    gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);
    for (int k = 0; k < tk; k++)
      for (int j = 0; j < tj; j++)
        gemmini_extended_mvin((void*)(((elem_t*)B_in) + k*DIM*(MATMUL_N/2) + j*DIM), bb + (k*tj + j)*DIM, DIM, DIM);
    gemmini_loop_ws_spad(ti, tj, tk, 0, 0, 0, ab, BANK_NUM * BANK_ROWS, 0, SDST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&C_hw[0][0], SDST * 16, MATMUL_M * MATMUL_N);
    gemmini_fence();
  }
}
