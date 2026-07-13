void solution(void) {
  // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);
    gemmini_config_ld(ATT_D * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)KT_in) + j * DIM * ATT_D + k * DIM),
                              b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    gemmini_config_st(S_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  // ---- 2) P = softmax(S1) rows on CPU; quantize to fp8 + per-32-group scales ----
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];
    float maxv = -1e30f;

    // Vectorized BF16 unpack: one load per uint64 word, extract 4 lanes, find max
    for (int jw = 0; jw < ATT_S / 4; jw++) {
      uint64_t word = S1_hw[i][jw];
      row[jw*4+0] = bf16_to_f((uint16_t)( word        & 0xFFFF));
      row[jw*4+1] = bf16_to_f((uint16_t)((word >> 16) & 0xFFFF));
      row[jw*4+2] = bf16_to_f((uint16_t)((word >> 32) & 0xFFFF));
      row[jw*4+3] = bf16_to_f((uint16_t)((word >> 48) & 0xFFFF));
      if (row[jw*4+0] > maxv) maxv = row[jw*4+0];
      if (row[jw*4+1] > maxv) maxv = row[jw*4+1];
      if (row[jw*4+2] > maxv) maxv = row[jw*4+2];
      if (row[jw*4+3] > maxv) maxv = row[jw*4+3];
    }

    // Exp pass: compute exp(row[j] - maxv) and accumulate sum
    // row[j] now holds raw (unnormalized) exp values; all values are >= 0
    float sum = 0.0f;
    for (int j = 0; j < ATT_S; j++) {
      row[j] = exp_f(row[j] - maxv);
      sum += row[j];
    }
    float inv_sum = 1.0f / sum;

    // Fused: for each group, find max of normalized values (row[j]*inv_sum),
    // compute scale, precompute combined = inv_sum * pow2i(-e) once per group,
    // then FP8-encode with a single multiply per element.
    for (int g = 0; g < GROUPS_S; g++) {
      // Find group max of normalized values without writing back to row[]
      // exp values are non-negative so abs(row[j]*inv_sum) == row[j]*inv_sum
      float gmax = 0.0f;
      for (int j = g * 32; j < (g + 1) * 32; j++) {
        float v = row[j] * inv_sum;
        if (v > gmax) gmax = v;
      }
      int e = 0;
      if (gmax > 0.0f) e = f_exp(gmax / 448.0f) + 1;
      // Precompute the single fused constant: inv_sum * pow2i(-e)
      // This replaces two per-element multiplies (row[j]*inv_sum, then *inv_scale)
      // with one per-element multiply (row[j]*combined)
      float combined = inv_sum * pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      // Single multiply per element: raw exp value * combined fused scale
      for (int j = g * 32; j < (g + 1) * 32; j++)
        P_q[i][j] = fp8_e4m3_rtz(row[j] * combined);
    }
  }

  // ---- 3) O = P @ V on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = ATT_S / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col, sizeof(V_scales_col), 1);
    gemmini_config_ld(ATT_S * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)V_in) + j * DIM * ATT_S + k * DIM),
                              b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
}
