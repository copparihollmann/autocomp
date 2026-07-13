// AUTO-GENERATED outer-tiled MX matmul baseline (1024x768x768).
void solution(void) {
  for (int i0 = 0; i0 < BLOCKS_M; i0++)
   for (int j0 = 0; j0 < BLOCKS_N; j0++) {
      // stage A/B blocks (BLK x K) + per-32-group scales
      for (int r = 0; r < BLK; r++)
        memcpy(&A_blk[r][0], &A_in[i0 * BLK + r][0], MATMUL_K);
      for (int r = 0; r < BLK; r++)
        memcpy(&B_blk[r][0], &B_in[j0 * BLK + r][0], MATMUL_K);
      for (int g = 0; g < MATMUL_K / 32; g++) {
        memcpy(&A_s[g][0], &A_scales_row[g][i0 * BLK], BLK);
        memcpy(&B_s[g][0], &B_scales_col[g][j0 * BLK], BLK);
      }
      int tiles = BLK / DIM, tiles_K = MATMUL_K / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
      int SPAD_DEST = 128;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)A_s, A_S_BYTES, 0);
      gemmini_mx_load_scales((uint64_t)B_s, B_S_BYTES, 1);
      gemmini_config_ld(MATMUL_K * sizeof(elem_t));
      for (int i = 0; i < tiles; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)A_blk) + i * DIM * MATMUL_K + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      for (int j = 0; j < tiles; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)B_blk) + j * DIM * MATMUL_K + k * DIM),
                                b_base + (j * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles, tiles, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles, tiles, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&C_blk[0][0], SPAD_DEST * 16, BLK * BLK);
      gemmini_fence();
      for (int r = 0; r < BLK; r++)
        for (int c = 0; c < BLK; c++) {
          uint16_t b = (uint16_t)((C_blk[r][c / 4] >> ((c % 4) * 16)) & 0xFFFF);
          C_acc[i0 * BLK + r][j0 * BLK + c] = bf16f(b);
        }
    }
}
