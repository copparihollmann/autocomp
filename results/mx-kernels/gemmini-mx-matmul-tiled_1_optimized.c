void solution(void) {
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      1, 1, 0, 0, false, 0, 0, 3, 0);

  int tiles   = BLK / DIM;
  int tiles_K = MATMUL_K / DIM;
  uint32_t a_base   = 0;
  uint32_t b_base   = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
  int      SPAD_DEST = 128;

  for (int i0 = 0; i0 < BLOCKS_M; i0++) {
    for (int j0 = 0; j0 < BLOCKS_N; j0++) {

      /* --- stage A block (BLK x K) ---------------------------------------- */
      for (int r = 0; r < BLK; r++)
        memcpy(&A_blk[r][0], &A_in[i0 * BLK + r][0], MATMUL_K);

      /* --- stage B block (BLK x K) ---------------------------------------- */
      for (int r = 0; r < BLK; r++)
        memcpy(&B_blk[r][0], &B_in[j0 * BLK + r][0], MATMUL_K);

      /* --- gather per-32-group scales -------------------------------------- */
      for (int g = 0; g < MATMUL_K / 32; g++) {
        memcpy(&A_s[g][0], &A_scales_row[g][i0 * BLK], BLK);
        memcpy(&B_s[g][0], &B_scales_col[g][j0 * BLK], BLK);
      }

      /* 1. load A and B scale factors --------------------------------------- */
      gemmini_mx_load_scales((uint64_t)A_s, A_S_BYTES, 0);  /* sel=0 → A */
      gemmini_mx_load_scales((uint64_t)B_s, B_S_BYTES, 1);  /* sel=1 → B */

      /* 2. mvin A tiles (port 0) ------------------------------------------- */
      gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
      for (int i = 0; i < tiles; i++) {
        for (int k = 0; k < tiles_K; k++) {
          const elem_t *src_a = (const elem_t *)A_blk + i * DIM * MATMUL_K + k * DIM;
          uint32_t dst_a = a_base + (i * tiles_K + k) * DIM;
          gemmini_extended_mvin(src_a, dst_a, DIM, DIM);
        }
      }

      /* 3. mvin B tiles (port 1) ------------------------------------------- */
      gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
      for (int j = 0; j < tiles; j++) {
        for (int k = 0; k < tiles_K; k++) {
          const elem_t *src_b = (const elem_t *)B_blk + j * DIM * MATMUL_K + k * DIM;
          uint32_t dst_b = b_base + (j * tiles_K + k) * DIM;
          gemmini_extended_mvin2(src_b, dst_b, DIM, DIM);
        }
      }

      /* 4. configure output / requantizer ----------------------------------- */
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors,
          tiles, tiles, tiles_K,
          0, 0, 1);

      /* 5. compute ---------------------------------------------------------- */
      gemmini_loop_ws_spad(tiles, tiles, tiles_K,
          0, 0, 0,
          a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
          false, false,
          false, false, false,
          NO_ACTIVATION, 0, 0, false, 0x38);

      /* 6. drain BF16 results from mx_smem → DRAM -------------------------- */
      gemmini_mx_read_smem(&C_blk[0][0], SPAD_DEST * 16, BLK * BLK);
      gemmini_fence();

      /* 7. unpack BF16 results into C_acc ----------------------------------- */
      for (int r = 0; r < BLK; r++) {
        for (int c = 0; c < BLK; c++) {
          uint16_t b = (uint16_t)((C_blk[r][c / 4] >> ((c % 4) * 16)) & 0xFFFF);
          C_acc[i0 * BLK + r][j0 * BLK + c] = bf16f(b);
        }
      }
    }
  }
}
