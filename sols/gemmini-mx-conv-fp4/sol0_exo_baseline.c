// AUTO-GENERATED baseline MX patch-embed conv kernel (C=2 H=32 k=4 OC=32, fp4:e2m1).
void solution(void) {
  for(int pi=0;pi<CH/CK;pi++)for(int pj=0;pj<PX;pj++)for(int c=0;c<CC;c++)for(int dy=0;dy<CK;dy++)for(int dx=0;dx<CK;dx++)
    A_buf[pi*PX+pj][c*CK*CK+dy*CK+dx]=X_in[c][pi*CK+dy][pj*CK+dx];
  for(int m=0;m<MATMUL_M;m++)for(int kk=0;kk<MATMUL_K;kk++){uint8_t cd=A_buf[m][kk]&0xF;
    if((m&1)==0)A_in_hw[m>>1][kk]=(A_in_hw[m>>1][kk]&0xF0)|cd; else A_in_hw[m>>1][kk]=(A_in_hw[m>>1][kk]&0x0F)|(cd<<4);}
  { int ti=MATMUL_M/DIM/2, tj=MATMUL_N/DIM/2, tk=MATMUL_K/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,2,2,3,0);
    gemmini_mx_load_scales((uint64_t)&A_scales_row,sizeof(A_scales_row),0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col,sizeof(B_scales_col),1);
    gemmini_config_st(OUT_COLS*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,1);
    gemmini_config_ld(MATMUL_K*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int kk=0;kk<tk;kk++) gemmini_extended_mvin((void*)(((elem_t*)A_in_hw)+i*DIM*MATMUL_K+kk*DIM),ab+(i*tk+kk)*DIM,DIM,DIM);
    gemmini_config_ld(MATMUL_N*sizeof(elem_t)/2);
    for(int kk=0;kk<tk;kk++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)B_in)+kk*DIM*(MATMUL_N/2)+j*DIM),bb+(kk*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_mx_read_smem(&C_hw[0][0],128*16,MATMUL_M*MATMUL_N); gemmini_fence(); }
}
