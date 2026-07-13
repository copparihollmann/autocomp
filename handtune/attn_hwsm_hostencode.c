// AUTO-GENERATED baseline MX attention kernel (S=128, D=64).
void solution(void) {
  // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_config_norm(MX_QLN2,     0,0,1,0,MX_QB,MX_QC);
    gemmini_config_norm(MX_QLN2_INV, 1,0,1,0,MX_QB,MX_QC);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);
    gemmini_config_ld(ATT_D * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
    gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S -> slot (k*tiles_J+j)
    for (int k = 0; k < tiles_K; k++)
      for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
    gemmini_config_st(S_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, SOFTMAX, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  // ---- 2) FLOAT-FREE integer softmax + fp8 requant (I-BERT iexp; host has no FPU) ----
  for (int i = 0; i < ATT_S; i++) {
    int64_t p[ATT_S];
    int64_t sm = 0;
    for (int j = 0; j < ATT_S; j += 4) {  // HW softmax already applied: S1_hw holds bf16 P; just unpack+sum (no max/iexp)
      uint64_t w = S1_hw[i][j / 4];
      for (int l = 0; l < 4; l++) { int64_t pv = mx_ibf16_to_q((uint16_t)((w >> (l*16)) & 0xFFFF)); if(pv<0)pv=0; p[j+l]=pv; sm+=pv; }
    }
    int ls = mx_ilog2((uint64_t)sm);
    uint64_t inv_sm = (((uint64_t)1 << (ls + MX_FRAC)) + ((uint64_t)sm >> 1)) / (uint64_t)sm;   // one rounded reciprocal/row
    for (int g = 0; g < GROUPS_S; g++) {
      int64_t gmax = 0;
      for (int j = g * 32; j < (g + 1) * 32; j++) if (p[j] > gmax) gmax = p[j];
      if (gmax == 0) { P_scales[g][i] = 0; for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = 0; continue; }
      // group e8m0 scale so the group-max lands at ~2^MX_P_TARGET_LOG2 (moderate; avoids e4 overflow)
      int e = mx_ilog2((uint64_t)gmax) - ls - MX_P_TARGET_LOG2;
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = mx_ienc_fp8(p[j], inv_sm, ls, e);
    }
  }

  gemmini_extended_config_st(DIM*sizeof(elem_t), RELU, ACC_SCALE_IDENTITY);  // clear sticky SOFTMAX before PV
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
      for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
    gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D -> slot (k*tiles_J+j)
    for (int k = 0; k < tiles_K; k++)
      for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
}
