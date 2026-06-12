// AUTO-GENERATED outer-tiled MX matmul baseline (128x128x128, fp4:e2m1).
void solution(void) {
  for(int i0=0;i0<BLOCKS_M;i0++)for(int j0=0;j0<BLOCKS_N;j0++){
    for(int g=0;g<MATMUL_K/32;g++)for(int b=0;b<BLK;b++){A_s[g][b]=A_scales_row[g][i0*BLK+b];B_s[g][b]=B_scales_col[g][j0*BLK+b];}
    for(int m=0;m<BLK;m++)for(int k=0;k<MATMUL_K;k++){uint8_t by=A_in[i0*BLK+m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);
      if((m&1)==0)A_hw[m>>1][k]=(A_hw[m>>1][k]&0xF0)|cd; else A_hw[m>>1][k]=(A_hw[m>>1][k]&0x0F)|(cd<<4);}
    int ti=BLK/DIM/2, tj=BLK/DIM/2, tk=MATMUL_K/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,2,2,3,0);
    gemmini_mx_load_scales((uint64_t)A_s,(MATMUL_K/32)*BLK,0);
    gemmini_mx_load_scales((uint64_t)B_s,(MATMUL_K/32)*BLK,1);
    gemmini_config_st(BLK_OUTC*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,1);
    gemmini_config_ld(MATMUL_K*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)A_hw)+i*DIM*MATMUL_K+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
    gemmini_config_ld(MATMUL_N*sizeof(elem_t)/2);
    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)B_in)+k*DIM*(MATMUL_N/2)+(j0*BLK)/2+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    for(int r=0;r<BLK;r++) gemmini_mx_read_smem(&C_hw[i0*BLK+r][(j0*BLK)/BF16_PER_WORD], 128*16 + r*BLK, BLK);
    gemmini_fence();
  }
}
