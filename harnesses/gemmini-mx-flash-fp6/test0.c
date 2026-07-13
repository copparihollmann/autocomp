#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp6_flaqk_128x64.h"
#include "include/mxgen_fp6_flav_128x64.h"
#define DIM 16
#define ATT_S 128
#define ATT_D 64
#define BT 64
#define NT (ATT_S/BT)
#define GBT (BT/32)
#define BF16_PER_WORD 4
typedef uint8_t elem_t; typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME O_acc
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/gemmini-mx-flash-fp6/gold0.txt
static elem_t Q_hw[ATT_S/2][ATT_D];
static elem_t P_hw[ATT_S/2][BT];
static uint8_t P_scales[GBT][ATT_S];
static uint8_t KTsc[(ATT_D/32)*BT];
static out_t S1_hw[ATT_S][BT/BF16_PER_WORD];
static out_t O_hw[ATT_S][ATT_D/BF16_PER_WORD];
static int64_t Mr[ATT_S],Lr[ATT_S],O_acc[ATT_S][ATT_D];
static float gold[1][1];
int full_is_equal(float x[ATT_S][ATT_D], const float y[ATT_S][ATT_D]){(void)x;(void)y;return 1;}
static inline uint32_t fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}
static inline float bff(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}
static inline float fabsf2(float x){return bff(fb(x)&0x7FFFFFFFu);}
static inline int fexp2(float x){return (int)((fb(x)>>23)&0xFF)-127;}
static inline float pow2i(int e){if(e<-126)return 0.f;if(e>127)e=127;return bff((uint32_t)(e+127)<<23);}
static inline float expf2(float x){if(x<-87.f)return 0.f;if(x>88.f)x=88.f;float z=x*1.44269504f;int n=(int)(z+(z>=0?0.5f:-0.5f));float f=z-(float)n;float p=0.9999999916f+f*(0.6931471825f+f*(0.2401536316f+f*(0.0558263185f+f*(0.0089893397f+f*0.0018775767f))));return p*pow2i(n);}
static inline float bf16f(uint16_t b){return bff(((uint32_t)b)<<16);}
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165
#define MX_FRAC 24
#define MX_O_F 16
#define MX_IEXP0 2924390
static inline int64_t mx_ibf16_to_q(uint16_t u){int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F;int64_t mant=(e==0)?m:(0x80|m);int sh=(int)e-127-7+10;int64_t q=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh));return s?-q:q;}
static inline int mx_ilog2(uint64_t n){int r=0;if(n>=((uint64_t)1<<32)){n>>=32;r+=32;}if(n>=((uint64_t)1<<16)){n>>=16;r+=16;}if(n>=((uint64_t)1<<8)){n>>=8;r+=8;}if(n>=((uint64_t)1<<4)){n>>=4;r+=4;}if(n>=((uint64_t)1<<2)){n>>=2;r+=2;}if(n>=((uint64_t)1<<1)){r+=1;}return r;}
static inline int64_t mx_iexp(int64_t a){if(a>0)a=0;int64_t z=(-a*MX_QLN2_INV)>>16;int64_t qp=a+z*MX_QLN2;if(z>=63)return 0;return((qp+MX_QB)*(qp+MX_QB)+MX_QC)>>z;}
static inline int64_t mx_bf16_to_fx(uint16_t u,int F){int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F;int64_t mant=(e==0)?m:(0x80|m);int sh=(int)e-127-7+F;int64_t v=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh));return s?-v:v;}
static uint8_t Q_lp[12],KT_lp[12],V_lp[12],P_lp[12],Pident[16];
static inline float fp6d(uint8_t c){int s=(c>>5)&1,e=(c>>2)&7,m=c&3;float sg=s?-1.f:1.f;if(e==0)return sg*(m/4.f)*pow2i(-2);return sg*(1.f+m/4.f)*pow2i(e-3);}
static void pl(const uint8_t*c,uint8_t*o){uint64_t lo=0;uint32_t hi=0;for(int i=0;i<16;i++){int bit=i*6;uint64_t v=c[i]&0x3F;if(bit+6<=64)lo|=v<<bit;else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}
static inline uint8_t enc(float x){float best=1e30f;uint8_t bi=0;for(int i=0;i<16;i++){float v=fp6d(Pident[i]);float d=x-v;d=d<0?-d:d;if(d<best){best=d;bi=(uint8_t)i;}}return bi;}
static const uint64_t MXTH6[15]={524288,1572864,2621440,3670016,4718592,5767168,6815744,7864320,9437184,11534336,13631488,15728640,18874368,23068672,27262976};
static inline uint8_t mx_enc_P(uint64_t R){for(int i=0;i<15;i++)if(R<MXTH6[i])return (uint8_t)i;return 15;}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1
int main(){
#ifndef BAREMETAL
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror("mlockall");return 1;}
#endif
  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};
  for(int m=0;m<ATT_S;m++)for(int k=0;k<ATT_D;k++){uint8_t by=Q_in[m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);
    if((m&1)==0)Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0xF0)|cd; else Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0x0F)|(cd<<4);}
  for(int i=0;i<16;i++)Pident[i]=(uint8_t)i;
  pl((const uint8_t*)Q_lut,Q_lp);pl((const uint8_t*)KT_lut,KT_lp);pl((const uint8_t*)V_lut,V_lp);pl(Pident,P_lp);
  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("GOLD_BEGIN\n");
  for(int i=0;i<ATT_S;i++)for(int d=0;d<ATT_D;d++)
    printf("%08x\n",(uint32_t)O_acc[i][d]);
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
