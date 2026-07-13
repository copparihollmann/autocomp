// AUTO-GENERATED baseline MX attention kernel (S=64, D=64).
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
    gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S -> slot (k*tiles_J+j)
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, DIM, DIM);
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
    for (int j = 0; j < ATT_S; j++) {
      uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
      row[j] = bf16_to_f(b);
      if (row[j] > maxv) maxv = row[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < ATT_S; j++) { row[j] = exp_f(row[j] - maxv); sum += row[j]; }
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < ATT_S; j++) row[j] *= inv_sum;
    // per-32-group power-of-two scale, then fp8 RTZ encode of row[j]/scale
    for (int g = 0; g < GROUPS_S; g++) {
      float gmax = 0.0f;
      for (int j = g * 32; j < (g + 1) * 32; j++)
        if (f_abs(row[j]) > gmax) gmax = f_abs(row[j]);
      int e = 0;
      // Target a MODERATE fp8 magnitude (gmax -> ~16), NOT fp8-max (448): the MX block
      // accumulator (mx_loop_ws_spad acc_e=4, max ~488) and product format (prod_e=4)
      // saturate to inf if P_q*V is large. Scaling P to fp8-max made P_q~256 and overflowed
      // ~half the P@V outputs to inf/nan (masked by golden-check). gmax/16 keeps P_q codes
      // O(1) like the matmul's header operands; the e8m0 group scale (P_scales) carries
      // magnitude, so the dequantized result is unchanged. Verified: 0 inf, 47%->88% vs fp32.
      if (gmax > 0.0f) e = f_exp(gmax / 16.0f) + 1;
      float inv_scale = pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++)
        P_q[i][j] = fp8_e4m3_rtz(row[j] * inv_scale);
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
    gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D -> slot (k*tiles_J+j)
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, DIM, DIM);
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
}
