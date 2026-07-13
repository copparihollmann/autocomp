#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp4_flaqk_128x64.h"
#include "include/mxgen_fp4_flav_128x64.h"
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
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/gemmini-mx-flash-attn/gold5.txt
static elem_t Q_hw[ATT_S/2][ATT_D];
static elem_t P_hw[ATT_S/2][BT];
static uint8_t P_scales[GBT][ATT_S];
static uint8_t KTsc[(ATT_D/32)*BT];
static out_t S1_hw[ATT_S][BT/BF16_PER_WORD];
static out_t O_hw[ATT_S][ATT_D/BF16_PER_WORD];
static float Mr[ATT_S],Lr[ATT_S],O_acc[ATT_S][ATT_D];
static float gold[1][1];
int full_is_equal(float x[ATT_S][ATT_D], const float y[ATT_S][ATT_D]){(void)x;(void)y;return 1;}
static inline uint32_t fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}
static inline float bff(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}
static inline float fabsf2(float x){return bff(fb(x)&0x7FFFFFFFu);}
static inline int fexp2(float x){return (int)((fb(x)>>23)&0xFF)-127;}
static inline float pow2i(int e){if(e<-126)return 0.f;if(e>127)e=127;return bff((uint32_t)(e+127)<<23);}
static inline float expf2(float x){if(x<-87.f)return 0.f;if(x>88.f)x=88.f;float z=x*1.44269504f;int n=(int)(z+(z>=0?0.5f:-0.5f));float f=z-(float)n;float p=0.9999999916f+f*(0.6931471825f+f*(0.2401536316f+f*(0.0558263185f+f*(0.0089893397f+f*0.0018775767f))));return p*pow2i(n);}
static inline float bf16f(uint16_t b){return bff(((uint32_t)b)<<16);}
static inline float fp4d(uint8_t c){int s=(c>>3)&1,e=(c>>1)&3,m=c&1;float sg=s?-1.f:1.f;if(e==0)return sg*(m*0.5f);return sg*(1.f+m*0.5f)*pow2i(e-1);}
static inline uint8_t enc(float x){if(x==0.f)return 0;uint8_t s=(fb(x)>>31)?0x8:0;float a=fabsf2(x);uint8_t c;if(a<0.5f)c=0;else if(a<1.f)c=1;else if(a<1.5f)c=2;else if(a<2.f)c=3;else if(a<3.f)c=4;else if(a<4.f)c=5;else if(a<6.f)c=6;else c=7;return s|c;}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1
int main(){
#ifndef BAREMETAL
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror("mlockall");return 1;}
#endif
  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};
  for(int m=0;m<ATT_S;m++)for(int k=0;k<ATT_D;k++){uint8_t by=Q_in[m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);
    if((m&1)==0)Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0xF0)|cd; else Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0x0F)|(cd<<4);}
  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }
  printf("GOLD_BEGIN\n");
  for(int i=0;i<ATT_S;i++)for(int d=0;d<ATT_D;d++)
    printf("%08x\n",(unsigned)fb(O_acc[i][d]));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
