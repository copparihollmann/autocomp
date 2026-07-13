// AUTO-GENERATED FLASH attention baseline (S=128, D=64, tile=64).
void solution(void) {
  // ---- flash attention: tile K/V along sequence, online softmax ----
  for (int i = 0; i < ATT_S; i++) { M_run[i] = -((int64_t)1 << 62); L_run[i] = 0; }
  memset(O_acc, 0, ATT_S * ATT_D * sizeof(int64_t));

  for (int t = 0; t < N_TILES; t++) {
    // 1) scores tile S1 = Q @ K_t^T   [S x BT] on Gemmini
    {
      int tiles_I = ATT_S / DIM, tiles_J = BT / DIM, tiles_K = ATT_D / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
      // KT_scales_col is [ATT_D/32][ATT_S] (row stride ATT_S, NOT BT): a contiguous slice at
      // [0][t*BT] reads group 1's scales from group 0's row -> wrong scales on half the
      // K-reduction. Repack the tile's column scales into the group-major [g*BT + n] layout
      // the HW expects for an (ATT_D x BT) matmul (scale_b_mem[group*N_DIM + n], N_DIM=BT).
      static uint8_t KT_sc_tile[(ATT_D / 32) * BT];
      for (int sg = 0; sg < ATT_D / 32; sg++)
        for (int sn = 0; sn < BT; sn++)
          KT_sc_tile[sg * BT + sn] = KT_scales_col[sg][t * BT + sn];
      gemmini_mx_load_scales((uint64_t)KT_sc_tile, (ATT_D / 32) * BT, 1);
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S; tile cols [t*BT..]
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + t * BT + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * BT);
      gemmini_fence();
    }

    // 2) FLOAT-FREE integer online softmax update + fp8 requant of P_t (host has no FPU)
    for (int i = 0; i < ATT_S; i++) {
      int64_t q[BT];
      int64_t m_new = M_run[i];
      for (int j = 0; j < BT; j++) {
        int64_t qq = mx_ibf16_to_q((uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF));
        q[j] = qq; if (qq > m_new) m_new = qq;
      }
      int64_t corr = mx_iexp(M_run[i] - m_new);          // = exp(M_old - m_new) * IEXP0
      int64_t p[BT];
      int64_t st = 0;
      for (int j = 0; j < BT; j++) { int64_t pv = mx_iexp(q[j] - m_new); p[j] = pv; st += pv; }
      L_run[i] = (L_run[i] * corr) / MX_IEXP0 + st;       // rescale prev denom + add this tile
      for (int d = 0; d < ATT_D; d++) O_acc[i][d] = (O_acc[i][d] * corr) / MX_IEXP0;  // rescale prev output
      M_run[i] = m_new;
      // per-32 group e8m0 scale + integer fp8 e4m3 RNE of P_t (UNNORMALIZED; normalize deferred to /Lq)
      for (int g = 0; g < GROUPS_BT; g++) {
        int64_t gmax = 0;
        for (int j = g * 32; j < (g + 1) * 32; j++) if (p[j] > gmax) gmax = p[j];
        int eg = (gmax > 0) ? (mx_ilog2((uint64_t)gmax) - MX_P_TARGET_LOG2) : 0;
        P_s[g][i] = (uint8_t)(eg + 127);
        for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = mx_enc_unnorm(p[j], eg);
      }
    }

    // 3) O_tile = P_t @ V_t  [S x D] on Gemmini; accumulate in fp32
    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = BT / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)P_s, GROUPS_BT * ATT_S, 0);
      gemmini_mx_load_scales((uint64_t)&V_scales_col[t * GROUPS_BT][0], GROUPS_BT * ATT_D, 1);
      gemmini_config_ld(BT * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * BT + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D; tile rows [t*BT..]
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)V_in) + (t * BT + k * DIM) * ATT_D + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_config_st(D_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
      gemmini_fence();
      for (int i = 0; i < ATT_S; i++)
        for (int d = 0; d < ATT_D; d++) {
          uint16_t b = (uint16_t)((O_hw[i][d / 4] >> ((d % 4) * 16)) & 0xFFFF);
          O_acc[i][d] += mx_bf16_to_fx(b, MX_O_F);   // bf16 P_t@V_t -> Q(MX_O_F) integer
        }
    }
  }

  // final normalization by the online denominator
  for (int i = 0; i < ATT_S; i++) {
    int64_t l = L_run[i]; if (l == 0) l = 1;
    for (int d = 0; d < ATT_D; d++) O_acc[i][d] = O_acc[i][d] / l;  // O*2^MX_O_F (fixed-point); integer divide
  }
}
