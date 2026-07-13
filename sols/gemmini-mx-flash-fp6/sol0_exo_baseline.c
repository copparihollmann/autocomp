// AUTO-GENERATED baseline MX FLASH attention kernel (S=128, D=64, fp6:e3m2).
void solution(void) {
  for(int i=0;i<ATT_S;i++){Mr[i]=-((int64_t)1<<62);Lr[i]=0;for(int d=0;d<ATT_D;d++)O_acc[i][d]=0;}
  for(int t=0;t<NT;t++){
    { int ti=ATT_S/DIM/2, tj=BT/DIM/2, tk=ATT_D/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;
      for(int g=0;g<ATT_D/32;g++)for(int n=0;n<BT;n++)KTsc[g*BT+n]=KT_scales_col[g][t*BT+n];
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,1,1,3,1);
      gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);
      gemmini_mx_load_scales((uint64_t)KTsc,(ATT_D/32)*BT,1);
      gemmini_config_st((BT/BF16_PER_WORD)*sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,8);
      gemmini_mx_load_lut((uint64_t)KT_lp,1,0); gemmini_mx_load_lut((uint64_t)Q_lp,1,1);
      gemmini_config_ld(ATT_D*sizeof(elem_t));
      for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)Q_hw)+i*DIM*ATT_D+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
      gemmini_config_ld(ATT_S*sizeof(elem_t)/2);
      for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)KT_in)+k*DIM*(ATT_S/2)+(t*BT)/2+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
      gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
      gemmini_mx_read_smem(&S1_hw[0][0],128*16,ATT_S*BT); gemmini_fence(); }
    for(int i=0;i<ATT_S;i++){  /* FLOAT-FREE integer online softmax + fp6/4 requant */
      int64_t q[BT]; int64_t mn=Mr[i];
      for(int j=0;j<BT;j++){int64_t qq=mx_ibf16_to_q((uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF));q[j]=qq;if(qq>mn)mn=qq;}
      int64_t corr=mx_iexp(Mr[i]-mn); int64_t p[BT],st=0;
      for(int j=0;j<BT;j++){int64_t pv=mx_iexp(q[j]-mn);p[j]=pv;st+=pv;}
      Lr[i]=(Lr[i]*corr)/MX_IEXP0+st;
      for(int d=0;d<ATT_D;d++)O_acc[i][d]=(O_acc[i][d]*corr)/MX_IEXP0; Mr[i]=mn;
      for(int g=0;g<GBT;g++){int64_t gm=0;for(int j=g*32;j<g*32+32;j++)if(p[j]>gm)gm=p[j];
        int e=(gm>0)?(mx_ilog2((uint64_t)gm)-0):0;P_scales[g][i]=(uint8_t)(e+127);
        for(int j=g*32;j<g*32+32;j++){uint64_t Rt=(e>=0)?(((uint64_t)p[j]<<MX_FRAC)>>e):((uint64_t)p[j]<<(MX_FRAC-e));uint8_t idx=mx_enc_P(Rt);
          if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|idx; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(idx<<4);}}}
    { int ti=ATT_S/DIM/2, tj=ATT_D/DIM/2, tk=BT/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,1,1,3,1);
      gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);
      gemmini_mx_load_scales((uint64_t)&V_scales_col[t*GBT][0],GBT*ATT_D,1);
      gemmini_config_st((ATT_D/BF16_PER_WORD)*sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,8);
      gemmini_mx_load_lut((uint64_t)V_lp,1,0); gemmini_mx_load_lut((uint64_t)P_lp,1,1);
      gemmini_config_ld(BT*sizeof(elem_t));
      for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)P_hw)+i*DIM*BT+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);
      gemmini_config_ld(ATT_D*sizeof(elem_t)/2);
      for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)V_in)+(t*BT+k*DIM)*(ATT_D/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);
      gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
      gemmini_mx_read_smem(&O_hw[0][0],128*16,ATT_S*ATT_D); gemmini_fence();
      for(int i=0;i<ATT_S;i++)for(int d=0;d<ATT_D;d++){uint16_t b=(uint16_t)((O_hw[i][d/4]>>((d%4)*16))&0xFFFF);O_acc[i][d]+=mx_bf16_to_fx(b,MX_O_F);} } }
  for(int i=0;i<ATT_S;i++){int64_t l=Lr[i];if(l==0)l=1;for(int d=0;d<ATT_D;d++)O_acc[i][d]=O_acc[i][d]/l;}
}
