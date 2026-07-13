// AUTO-GENERATED outer-tiled MX matmul baseline (1024x768x768).
void solution(void) {
  for (int i0 = 0; i0 < BLOCKS_M; i0++)
   for (int j0 = 0; j0 < BLOCKS_N; j0++) {
      // PROF cpu_stage
      // Stage only the tiny per-32-group scales; A/B data is mvin'd DIRECTLY
      // from DRAM (no host-side block memcpy — that was 98% of the cycles).
      for (int g = 0; g < MATMUL_K / 32; g++) {
        memcpy(&A_s[g][0], &A_scales_row[g][i0 * BLK], BLK);
        memcpy(&B_s[g][0], &B_scales_col[g][j0 * BLK], BLK);
      }
      int tiles = BLK / DIM, tiles_K = MATMUL_K / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
      int SPAD_DEST = 128;
      // PROF accel
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)A_s, A_S_BYTES, 0);
      gemmini_mx_load_scales((uint64_t)B_s, B_S_BYTES, 1);
      gemmini_config_ld(MATMUL_K * sizeof(elem_t));      // A rows: stride K
      for (int i = 0; i < tiles; i++)
        for (int k = 0; k < tiles_K; k += 3) { int bb = (tiles_K - k < 3) ? (tiles_K - k) : 3;
          gemmini_extended_mvin((void*)(((elem_t*)A_in) + (i0 * BLK + i * DIM) * MATMUL_K + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
      gemmini_config_ld(MATMUL_N * sizeof(elem_t));      // B rows: stride N
      for (int k = 0; k < tiles_K; k++)                  // B[k][j0*BLK+j] -> slot (k*tiles + j); batch along j (contiguous cols)
        for (int j = 0; j < tiles; j += 3) { int bb = (tiles - j < 3) ? (tiles - j) : 3;
          gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + (j0 * BLK + j * DIM)),
                                b_base + (k * tiles + j) * DIM, bb * DIM, DIM); }
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles, tiles, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles, tiles, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      // Accelerator-centric readout: per-row hardware DMA from smem straight into the
      // bf16-packed row-major output. The old CPU bf16->fp32 assembly was ~99% of cycles
      // (gemmini was ~1%); this DMA path is ~74x faster on 256x256x512 and exposes the
      // accelerator work for autocomp to optimize.
      for (int r = 0; r < BLK; r++)
        gemmini_mx_read_smem(&C_hw[i0 * BLK + r][(j0 * BLK) / BF16_PER_WORD],
                             SPAD_DEST * 16 + r * BLK, BLK);
      gemmini_fence();
    }
}
