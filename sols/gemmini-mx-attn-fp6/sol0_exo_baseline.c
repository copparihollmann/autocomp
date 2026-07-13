// AUTO-GENERATED baseline MX attention kernel (S=128, D=64, fp6:e3m2).
void solution(void) {
  { int ti=ATT_S/DIM/2, tj=ATT_S/DIM/2, tk=ATT_D/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,1,1,3,1);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col,sizeof(KT_scales_col),1);
    gemmini_config_st((ATT_S/BF16_PER_WORD)*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,8);
    gemmini_mx_load_lut((uint64_t)KT_lp,1,0); gemmini_mx_load_lut((uint64_t)Q_lp,1,1);
    gemmini_config_ld(ATT_D*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)Q_hw)+i*DIM*ATT_D+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
    gemmini_config_ld(ATT_S*sizeof(elem_t)/2);
    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)KT_in)+k*DIM*(ATT_S/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_mx_read_smem(&S1_hw[0][0],SD*16,ATT_S*ATT_S); gemmini_fence(); }
  for(int i=0;i<ATT_S;i++){  /* FLOAT-FREE integer softmax + fp6 requant (host has no FPU) */
    int64_t q[ATT_S]; int64_t mq=-((int64_t)1<<62);
    for(int j=0;j<ATT_S;j++){int64_t qq=mx_ibf16_to_q((uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF));q[j]=qq;if(qq>mq)mq=qq;}
    int64_t p[ATT_S],sm=0; int64_t gmarr[GKP];   /* fused group-max: capture per-group max in the iexp pass */
    for(int g=0;g<GKP;g++){{int64_t gm=0;for(int j=g*32;j<g*32+32;j++){{int64_t pv=mx_iexp(q[j]-mq);p[j]=pv;sm+=pv;if(pv>gm)gm=pv;}}gmarr[g]=gm;}}
    int ls=mx_ilog2((uint64_t)sm); uint64_t inv=(((uint64_t)1<<(ls+MX_FRAC))+((uint64_t)sm>>1))/(uint64_t)sm;
    for(int g=0;g<GKP;g++){ int64_t gm=gmarr[g];
      int e=(gm>0)?(mx_ilog2((uint64_t)gm)-ls+2):0; P_scales[g][i]=(uint8_t)(e+127);
      for(int j=g*32;j<g*32+32;j++){ uint64_t R=(((uint64_t)p[j]*inv)+((uint64_t)1<<(ls-1)))>>ls;
        uint64_t Rt=(e>=0)?(R>>e):(R<<(-e)); uint8_t idx=mx_enc_P(Rt);
        if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|idx; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(idx<<4); } } }
  { int ti=ATT_S/DIM/2, tj=ATT_D/DIM/2, tk=ATT_S/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,1,1,3,1);
    gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col,sizeof(V_scales_col),1);
    gemmini_config_st((ATT_D/BF16_PER_WORD)*sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,8);
    gemmini_mx_load_lut((uint64_t)V_lp,1,0); gemmini_mx_load_lut((uint64_t)P_lp,1,1);
    gemmini_config_ld(ATT_S*sizeof(elem_t));
    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)P_hw)+i*DIM*ATT_S+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
    gemmini_config_ld(ATT_D*sizeof(elem_t)/2);
    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)V_in)+k*DIM*(ATT_D/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_mx_read_smem(&O_hw[0][0],SD*16,ATT_S*ATT_D); gemmini_fence(); }
}
