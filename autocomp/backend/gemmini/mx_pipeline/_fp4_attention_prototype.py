import re
from autocomp.backend.gemmini.mx_pipeline import matmul_gen as MG
from autocomp.backend.gemmini import gemmini_eval as E
gpath = MG.GEMMINI_PATH
S,D=64,64
HQK=f"mxgen_fp4_att_qk_{S}x{D}.h"; HV=f"mxgen_fp4_att_v_{S}x{D}.h"
def build(T):
  return r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/%HQK%"
#include "include/%HV%"
#define DIM 16
#define ATT_S %S%
#define ATT_D %D%
#define GKP (ATT_S/32)
typedef uint8_t elem_t; typedef uint64_t out_t;
#define BF16_PER_WORD 4
static elem_t Q_hw[ATT_S/2][ATT_D];
static elem_t P_hw[ATT_S/2][ATT_S];
static uint8_t P_scales[GKP][ATT_S];
static out_t S1_hw[ATT_S][ATT_S/BF16_PER_WORD];
static inline uint32_t fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}
static inline float bf(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}
static inline float fabsf2(float x){return bf(fb(x)&0x7FFFFFFFu);}
static inline int fexp2(float x){return (int)((fb(x)>>23)&0xFF)-127;}
static inline float pow2i(int e){if(e<-126)return 0.f;if(e>127)e=127;return bf((uint32_t)(e+127)<<23);}
static inline float expf2(float x){if(x<-87.f)return 0.f;if(x>88.f)x=88.f;float z=x*1.44269504f;int n=(int)(z+(z>=0?0.5f:-0.5f));float f=z-(float)n;float p=0.9999999916f+f*(0.6931471825f+f*(0.2401536316f+f*(0.0558263185f+f*(0.0089893397f+f*0.0018775767f))));return p*pow2i(n);}
static inline float bf16f(uint16_t b){return bf(((uint32_t)b)<<16);}
static inline uint8_t e2m1rtz(float x){if(x==0.f)return 0;uint8_t s=(fb(x)>>31)?0x8:0;float a=fabsf2(x);uint8_t c;if(a<0.5f)c=0;else if(a<1.f)c=1;else if(a<1.5f)c=2;else if(a<2.f)c=3;else if(a<3.f)c=4;else if(a<4.f)c=5;else if(a<6.f)c=6;else c=7;return s|c;}
static inline float fp4d(uint8_t c){int s=(c>>3)&1,e=(c>>1)&3,m=c&1;float sg=s?-1.f:1.f;if(e==0)return sg*(m*0.5f);return sg*(1.f+m*0.5f)*pow2i(e-1);}
int main(){
#ifndef BAREMETAL
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror("mlockall");return 1;}
#endif
  // M-pack Q (K-packed Q_in[S][D/2] -> Q_hw[S/2][D])
  for(int m=0;m<ATT_S;m++)for(int k=0;k<ATT_D;k++){uint8_t by=Q_in[m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);
    if((m&1)==0)Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0xF0)|cd; else Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0x0F)|(cd<<4);}
  static out_t O_hw[ATT_S][ATT_D/BF16_PER_WORD];
  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};
  // ---- QK: S1 = Q@K^T (fp4) ----
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
    gemmini_mx_read_smem(&S1_hw[0][0],SD*16,ATT_S*ATT_S);
    gemmini_fence(); }
  // ---- softmax + e2m1 encode of P, M-pack ----
  for(int i=0;i<ATT_S;i++){
    float row[ATT_S], mx=-1e30f;
    for(int j=0;j<ATT_S;j++){uint16_t b=(uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF);row[j]=bf16f(b);if(row[j]>mx)mx=row[j];}
    float sm=0.f; for(int j=0;j<ATT_S;j++){row[j]=expf2(row[j]-mx);sm+=row[j];}
    for(int j=0;j<ATT_S;j++) row[j]/=sm;
    for(int g=0;g<GKP;g++){ float gm=0.f; for(int j=g*32;j<g*32+32;j++){float a=fabsf2(row[j]);if(a>gm)gm=a;}
      int e=0; if(gm>0.f)e=fexp2(gm/%T%f)+1; float isc=pow2i(-e); P_scales[g][i]=(uint8_t)(e+127);
      for(int j=g*32;j<g*32+32;j++){ uint8_t cd=e2m1rtz(row[j]*isc);
        if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|cd; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(cd<<4); } } }
  // ---- PV: O = P@V (fp4) ----
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
    gemmini_mx_read_smem(&O_hw[0][0],SD*16,ATT_S*ATT_D);
    gemmini_fence(); }
  // ---- TRUE fp4 ref: mirror QK(fp4)->softmax->e2m1 P->PV(fp4) ----
  int near=0,tot=0,inf=0;
  for(int i=0;i<ATT_S;i++){
    float s1[ATT_S],mx=-1e30f;
    for(int j=0;j<ATT_S;j++){float acc=0.f; for(int d=0;d<ATT_D;d++){ uint8_t qb=Q_in[i][d>>1],qi=(d&1)?((qb>>4)&0xF):(qb&0xF);
        uint8_t kb=KT_in[d][j>>1],ki=(j&1)?((kb>>4)&0xF):(kb&0xF);
        acc+=fp4d(qi)*pow2i((int)Q_scales_row[d/32][i]-127)*fp4d(ki)*pow2i((int)KT_scales_col[d/32][j]-127);}
      s1[j]=acc; if(acc>mx)mx=acc;}
    float p[ATT_S],sm=0.f; for(int j=0;j<ATT_S;j++){p[j]=expf2(s1[j]-mx);sm+=p[j];} for(int j=0;j<ATT_S;j++)p[j]/=sm;
    for(int g=0;g<GKP;g++){float gm=0.f;for(int j=g*32;j<g*32+32;j++){float a=fabsf2(p[j]);if(a>gm)gm=a;}
      int e=0;if(gm>0.f)e=fexp2(gm/%T%f)+1;float isc=pow2i(-e),sc=pow2i(e);
      for(int j=g*32;j<g*32+32;j++)p[j]=fp4d(e2m1rtz(p[j]*isc))*sc;}
    for(int d=0;d<ATT_D;d++){float acc=0.f; for(int s=0;s<ATT_S;s++){uint8_t vb=V_in[s][d>>1],vi=(d&1)?((vb>>4)&0xF):(vb&0xF);
        acc+=p[s]*fp4d(vi)*pow2i((int)V_scales_col[s/32][d]-127);}
      uint16_t hb=(uint16_t)((O_hw[i][d/4]>>((d%4)*16))&0xFFFF); float hw=bf16f(hb);
      unsigned u=fb(hw),ex=(u>>23)&0xFF; if(ex==0xFF){inf++;tot++;continue;}
      float df=hw-acc,dn=acc>0?acc:-acc; if(dn<1e-6f)dn=1e-6f; tot++; if((df<0?-df:df)/dn<0.05f)near++;}}
  printf("FP4ATT T=%T% near5=%d inf=%d tot=%d\n",near,inf,tot);
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
'''.replace("%HQK%",HQK).replace("%HV%",HV).replace("%S%",str(S)).replace("%D%",str(D)).replace("%T%",str(T))
for T in ["2.0","1.0","0.5"]:
    rd={}; E.run_spike(build(T), rd, gpath, "0", 600)
    out=rd.get("retval","") if isinstance(rd.get("retval"),str) else str(rd.get("retval"))
    m=re.search(r"FP4ATT T=[\d.]+ near5=(\d+) inf=(\d+) tot=(\d+)", out)
    if m: n,i,t=int(m.group(1)),int(m.group(2)),int(m.group(3)); print(f"fp4 attention T={T}: within5%={100*n/t:.0f}% ({n}/{t}), inf={i}")
    else: print(f"T={T} FAIL: {out[:200]}")
