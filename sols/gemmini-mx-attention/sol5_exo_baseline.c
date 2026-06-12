// AUTO-GENERATED baseline MX attention kernel (S=64, D=64, fp4:e2m1).
void solution(void) {
  { int ti=ATT_S/DIM/2, tj=ATT_S/DIM/2, tk=ATT_D/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,2,2,3,0);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col,sizeof(KT_scales_col),1);
    gemmini_config_st((ATT_S/BF16_PER_WORD)*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,1);
    gemmini_config_ld(ATT_D*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)Q_hw)+i*DIM*ATT_D+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
    gemmini_config_ld(ATT_S*sizeof(elem_t)/2);
    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)KT_in)+k*DIM*(ATT_S/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_mx_read_smem(&S1_hw[0][0],SD*16,ATT_S*ATT_S); gemmini_fence(); }
  for(int i=0;i<ATT_S;i++){
    float row[ATT_S], mx=-1e30f;
    for(int j=0;j<ATT_S;j++){uint16_t b=(uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF);row[j]=bf16f(b);if(row[j]>mx)mx=row[j];}
    float sm=0.f; for(int j=0;j<ATT_S;j++){row[j]=expf2(row[j]-mx);sm+=row[j];}
    for(int j=0;j<ATT_S;j++) row[j]/=sm;
    for(int g=0;g<GKP;g++){ float gm=0.f; for(int j=g*32;j<g*32+32;j++){float a=fabsf2(row[j]);if(a>gm)gm=a;}
      int e=0; if(gm>0.f)e=fexp2(gm/0.5f)+1; float isc=pow2i(-e); P_scales[g][i]=(uint8_t)(e+127);
      for(int j=g*32;j<g*32+32;j++){ uint8_t idx=enc_P(row[j]*isc);
        if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|idx; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(idx<<4); } } }
  { int ti=ATT_S/DIM/2, tj=ATT_D/DIM/2, tk=ATT_S/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,2,2,3,0);
    gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col,sizeof(V_scales_col),1);
    gemmini_config_st((ATT_D/BF16_PER_WORD)*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,1);
    gemmini_config_ld(ATT_S*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)P_hw)+i*DIM*ATT_S+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
    gemmini_config_ld(ATT_D*sizeof(elem_t)/2);
    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)V_in)+k*DIM*(ATT_D/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_mx_read_smem(&O_hw[0][0],SD*16,ATT_S*ATT_D); gemmini_fence(); }
}
