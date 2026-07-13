void solution(void) {
  // ---- flash attention: tile K/V along sequence, online softmax ----
  for (int i = 0; i < ATT_S; i++) { M_run[i] = -1e30f; L_run[i] = 0.0f; }
  memset(O_acc, 0, ATT_S * ATT_D * sizeof(float));

  // Hoist flush and invariant config OUTSIDE the tile loop
  gemmini_flush(0);
  // Both GEMMs share identical format/dataflow/stride config — issue once
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                               1, 1, 0, 0, false, 0, 0, 3, 0);

  for (int t = 0; t < N_TILES; t++) {
    // 1) scores tile S1 = Q @ K_t^T   [S x BT] on Gemmini
    {
      int tiles_I = ATT_S / DIM, tiles_J = BT / DIM, tiles_K = ATT_D / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      // NO gemmini_flush, NO gemmini_extended3_config_ex here
      gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
      gemmini_mx_load_scales((uint64_t)&KT_scales_col[0][t * BT], (ATT_D / 32) * BT, 1);
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (t * BT + j * DIM) * ATT_D + k * DIM),
                                b_base + (j * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * BT);
      gemmini_fence();
    }

    // 2) online softmax update + fp8 requant of P_t (CPU)
    for (int i = 0; i < ATT_S; i++) {
      float m_new = M_run[i];
      float row[BT];
      for (int j = 0; j < BT; j++) {
        uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
        row[j] = bf16_to_f(b);
        if (row[j] > m_new) m_new = row[j];
      }
      float corr = exp_f(M_run[i] - m_new);   // rescale of previous output/denominator
      float l_new = L_run[i] * corr;
      for (int j = 0; j < BT; j++) { row[j] = exp_f(row[j] - m_new); l_new += row[j]; }
      for (int d = 0; d < ATT_D; d++) O_acc[i][d] *= corr;
      M_run[i] = m_new; L_run[i] = l_new;
      // per-32 group scale + fp8 RTZ of P (unnormalized)
      for (int g = 0; g < GROUPS_BT; g++) {
        float gmax = 0.0f;
        for (int j = g * 32; j < (g + 1) * 32; j++)
          if (f_abs(row[j]) > gmax) gmax = f_abs(row[j]);
        int e = 0;
        if (gmax > 0.0f) e = f_exp(gmax / 448.0f) + 1;
        float inv_scale = pow2i(-e);
        P_s[g][i] = (uint8_t)(e + 127);
        for (int j = g * 32; j < (g + 1) * 32; j++)
          P_q[i][j] = fp8_e4m3_rtz(row[j] * inv_scale);
      }
    }

    // 3) O_tile = P_t @ V_t  [S x D] on Gemmini; accumulate in fp32
    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = BT / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      // NO gemmini_flush, NO gemmini_extended3_config_ex here
      gemmini_mx_load_scales((uint64_t)P_s, GROUPS_BT * ATT_S, 0);
      gemmini_mx_load_scales((uint64_t)&V_scales_col[t * GROUPS_BT][0], GROUPS_BT * ATT_D, 1);
      gemmini_config_ld(BT * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * BT + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)V_in) + j * DIM * ATT_S + t * BT + k * DIM),
                                b_base + (j * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_st(D_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
      gemmini_fence();
      for (int i = 0; i < ATT_S; i++)
        for (int d = 0; d < ATT_D; d++) {
          uint16_t b = (uint16_t)((O_hw[i][d / 4] >> ((d % 4) * 16)) & 0xFFFF);
          O_acc[i][d] += bf16_to_f(b);
        }
    }
  }

  // final normalization by the online denominator
  for (int i = 0; i < ATT_S; i++) {
    float inv_l = 1.0f / L_run[i];
    for (int d = 0; d < ATT_D; d++) O_acc[i][d] *= inv_l;
  }
}
