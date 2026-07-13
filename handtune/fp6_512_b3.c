// AUTO-GENERATED baseline from Rakanic fork kernel matmul_tiled_fp6_128x128x512 (fp6:e3m2).
void solution(void) {

  // gemmini_config_ex(WEIGHT_STATIONARY, 0, 0); // Full version used instead to configure format
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) // A stride
    | (GEMMINI_FORMAT << 14) // C format
    | (GEMMINI_FORMAT << 12) // B format
    | (GEMMINI_FORMAT << 10) // A format
    | (0 << 9) // B transpose
    | (0 << 8) // A transpose
    | ((false) << 7) // Set only strides
    | ((USE_LUT) << 4)
    | ((0) << 3) // Activation function
    | ((WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
    ((uint64_t)(1) << 48) // C stride
    | (0),
    k_CONFIG);
  int tiles_I = MATMUL_M / 32;                       // = 4
  int tiles_K = MATMUL_K / 16;                         // = 8
  int tiles_J = MATMUL_N / 32 ;                           // = 4  (each n-tile = DIM*2 original cols)
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);
  // gemmini_extended_mvin((void *) B_in, GEMMINI_SPAD_ADDR_B, MATMUL_N / VALUES_PER_BYTE, MATMUL_K); // TODO: Half one dimension for fp4/6
  //gemmini_mxquant_config_mvout(1024, (uint64_t)scale_factors);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);

   // MVIN B

#ifdef SPIKE_SIM
  gemmini_mx_load_lut((uint64_t)&B_lut[0][0], (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY), 0);
  gemmini_mx_load_lut((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1);
  gemmini_mx_load_lut((uint64_t)&C_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 2);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
#ifdef USE_LUT_DEF
  for (size_t i = 0; i < (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i;
    dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
  }
  for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i;
    dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
  }
  for (size_t i = 0; i < (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i;
    dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
  }
#endif
#endif

#ifndef SPIKE_SIM
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M*MATMUL_GK);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N*MATMUL_GK);
  gemmini_fence();
#endif
  //gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  // Tile counts
  // A_in_hw[MATMUL_M/2][MATMUL_K]: tiles_I m-tiles x tiles_K k-tiles, each K_TILE hw-rows x DIM bytes
  // B_in[MATMUL_K][MATMUL_N/2]:    tiles_K k-tiles x tiles_J n-tiles, each K_TILE rows x DIM bytes


  uint32_t a_base = 0;
  uint32_t b_base = 8192 - tiles_K * tiles_J * K_TILE;

  // MVIN A: stride = MATMUL_K (full hw-row width); each tile = K_TILE hw-rows x DIM bytes
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k += 3) { int bb = (tiles_K - k < 3) ? (tiles_K - k) : 3;
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, bb * DIM, DIM);
    }
  }

  // MVIN B: stride = MATMUL_N/2 (full packed row width); each tile = K_TILE rows x DIM bytes
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j += 3) { int bb = (tiles_J - j < 3) ? (tiles_J - j) : 3;
      uint8_t *dram_ptr = (uint8_t *)B_in + k * K_TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, bb * DIM, DIM);
    }
  }

  int SPAD_DEST = 0;

 uint32_t acc_addr = (1u << (ADDR_LEN - 1));
 gemmini_loop_ws_spad( tiles_I, tiles_J, tiles_K,
      0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
      a_base,               // A scratchpad address
      8192,                 // B scratchpad end address
      0,                    // D (bias) - none
      SPAD_DEST,   // C accumulator address
      false, false,         // A_transpose, B_transpose
      false, false, false,  // full_C, low_D, ex_accumulate
      NO_ACTIVATION,        // activation
      0, 0,                 // a_spad_id, b_spad_id, double buffer setting
      false,                // is_resadd
      0x38);       //now skip the, ldA, ldB, and st

//   ┌─────┬──────┬────────────────────────────────┐
//   │ Bit │ Mask │             Skips              │
//   ├─────┼──────┼────────────────────────────────┤
//   │ 3   │ 0x08 │ ldA (skip loading A from DRAM) │
//   ├─────┼──────┼────────────────────────────────┤
//   │ 4   │ 0x10 │ ldB (skip loading B from DRAM) │
//   ├─────┼──────┼────────────────────────────────┤
//   │ 5   │ 0x20 │ ldD (skip loading D/bias)      │
//   ├─────┼──────┼────────────────────────────────┤
//   │ 6   │ 0x40 │ ex  (skip compute)             │Z
//   ├─────┼──────┼────────────────────────────────┤
//   │ 7   │ 0x80 │ st (skip acc→spad store)       │
//   └─────┴──────┴────────────────────────────────┘
// gemmini_config_ld(DIM * sizeof(welem_t));
// gemmini_preload(1 * DIM, (1u << (ADDR_LEN - 1)));
// gemmini_config_ld(DIM * sizeof(elem_t));
// gemmini_compute_preloaded(0 * DIM, GARBAGE_ADDR);

        // gemmini_extended4_config_ld(DIM * sizeof(welem_t), MVIN_SCALE_IDENTITY, false, TILE, 0);
        // gemmini_preload(GEMMINI_SPAD_ADDR_B + k, GEMMINI_ACC_ADDR_C + m);
        // gemmini_extended4_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, TILE, 0);
  //       // gemmini_compute_preloaded(GEMMINI_SPAD_ADDR_A + k, k == 0 ? GARBAGE_ADDR : GEMMINI_ACC_ADDR_C);
  //     }
  //   }
  // }

  gemmini_fence();

#ifdef SPIKE_SIM
  // BF16 output: M*N bf16 values.
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  printf("Address: %p \n", smem_start_addr);
  for (int i = 0; i < MATMUL_M; i ++) {
    for (int j = 0; j < OUT_COLS; j++) {
        C_hw[i][j] = *(smem_start_addr + (i*OUT_COLS + j));
    }
  }

  gemmini_fence();
#endif

  }
