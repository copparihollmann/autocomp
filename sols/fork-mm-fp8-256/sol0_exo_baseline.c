// AUTO-GENERATED baseline from Rakanic fork kernel matmul_tiled_fp8_128x128x256 (fp8:e4m3).
void solution(void) {

  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  gemmini_config_ld(MATMUL_N * sizeof(elem_t));

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
//      printf("j: %d, k: %d, sp_addr: %d \n", j, k, sp_addr);
//      printf("first elem: %x \n", dram_ptr[0]);
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 0;

  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // ---- Compute ----
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

//  for (int i = 0; i < tiles_I; i++) {
//    for (int j = 0; j < tiles_J; j++) {
//      uint32_t acc_tile_addr = acc_addr + (i * tiles_J + j) * DIM;
//      out_t *dram_ptr = &C_hw[i * DIM][j * DIM];
//      gemmini_mvout((void *) dram_ptr, acc_tile_addr);
//    }
//  }


//  gemmini_mvout((void*)&C_hw[0][0], 128 )

  gemmini_fence();

//  uint64_t* smem_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2 + 448 / 8;
//  printf("SMEM at address %x = %lx \n", ((uint64_t*)SMEM), *((uint64_t*)SMEM));
//  printf("SMEM at address %x = %lx \n", smem_addr, *smem_addr);
//  printf("SMEM at address %x = %lx \n", smem_addr + 2, *(smem_addr+2));


#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  printf("Moving out:\n");
  for (int i = 0; i < tiles_J*tiles_I*2; i++) {
      gemmini_mvout((void*)((uint64_t*) C_hw + i*2*DIM), SPAD_DEST + i*DIM);
  }
  gemmini_fence();
#endif

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----
  }
