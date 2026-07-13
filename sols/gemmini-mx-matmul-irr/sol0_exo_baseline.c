// AUTO-GENERATED baseline MX matmul kernel (64x48x64, fp8:e4m3).
void solution(void) {
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // mvin only the REAL rows/cols (partial last tile); loop_ws pad_I/pad_J/pad_K skip the
  // padded remainder in compute so the untouched (garbage) pad rows/cols are never read.
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));      // A rows: stride K
  for (int i = 0; i < tiles_I; i++) {
    int mrows = REAL_M - i * DIM; if (mrows > DIM) mrows = DIM; if (mrows <= 0) continue;
    for (int k = 0; k < tiles_K; k++) {
      int kcols = REAL_K - k * DIM; if (kcols > DIM) kcols = DIM; if (kcols <= 0) continue;
      gemmini_extended_mvin((void*)(((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, kcols, mrows);
    }
  }
  // B tile (k,j) = B_in[k][j] (stride N) into scratchpad slot (k*tiles_J + j) —
  // the layout gemmini's MX loop reads (B_t = B_sp + (k*TJ + j)*DIM). The old
  // (j*tiles_K + k) slot was a transpose: correct only when N==K, garbage otherwise.
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));      // B rows: stride N
  for (int j = 0; j < tiles_J; j++) {
    int ncols = REAL_N - j * DIM; if (ncols > DIM) ncols = DIM; if (ncols <= 0) continue;
    for (int k = 0; k < tiles_K; k++) {
      int krows = REAL_K - k * DIM; if (krows > DIM) krows = DIM; if (krows <= 0) continue;
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, ncols, krows);
    }
  }

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, PAD_I, PAD_J, PAD_K, a_base, BANK_NUM * BANK_ROWS, 0,
                       SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
