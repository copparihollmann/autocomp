"""Generate an autocomp MX-Gemmini FLASH ATTENTION problem.

True flash attention: tiles K/V along sequence, never materializes the full
S x S score matrix. Per K-tile: S1 = Q @ K_t^T on Gemmini; online softmax
(running max + denominator) and rescaling on CPU; O += P_t @ V_t on Gemmini.

This makes long sequences (S = 128/256+) tractable on MX-Gemmini where the
non-flash baseline needs the whole S x S in shared memory.

Reuses two golden_model.py headers (Q/KT and V), renamed; gold = hardware-
captured via Python comparison (GOLD_FILE), like the tiled matmul.

Usage:
    python -m autocomp.backend.gemmini.mx_pipeline.flash_attention_gen --S 128 --D 64 --id 0
"""

import argparse
import re

from autocomp.common import HARNESSES_DIR, SOLS_DIR
from autocomp.backend.gemmini import gemmini_eval as E
from autocomp.backend.gemmini.mx_pipeline.matmul_gen import (
    GEMMINI_PATH, GEMMINI_SW, gen_input_header, _FMT_TAG, _FMT_HW)
from autocomp.backend.gemmini.mx_pipeline.attention_gen import _rename_header

BT = 64  # K/V tile size along the sequence


_HARNESS = r'''// AUTO-GENERATED MX-Gemmini FLASH attention harness (S={S}, D={D}, tile={BT}).
// O = softmax(Q K^T) V with K/V tiling + online softmax (no S x S buffer).
// Correctness vs hardware-captured gold (GOLD_FILE) checked by the runner.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/{header_qk}"
#include "include/{header_v}"

#define DIM 16
#define ATT_S {S}
#define ATT_D {D}
#define BT {BT}
#define N_TILES (ATT_S / BT)
#define GROUPS_BT (BT / 32)
#define BLK_OUTC (BT / 4)
#define D_OUT_COLS (ATT_D / 4)

typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME O_acc
// GOLD_FILE: {gold_file}
int full_is_equal(float x[ATT_S][ATT_D], const float y[ATT_S][ATT_D]) {{
  (void)x; (void)y; return 1;  // verified vs GOLD_FILE by the runner
}}
static float gold[1][1];

// fixed scratch buffers (pinned addresses; spike behavior is layout-sensitive)
#define S1_hw   ((out_t (*)[BLK_OUTC]) 0xA0000000UL)   // S x BT scores tile (bf16-packed)
#define P_q     ((elem_t (*)[BT])      0xA0100000UL)
#define P_s     ((uint8_t (*)[ATT_S])  0xA0200000UL)
#define O_hw    ((out_t (*)[D_OUT_COLS]) 0xA0300000UL)
#define O_acc   ((int64_t (*)[ATT_D])  0xA0400000UL)   // integer fixed-point Q(O_F) (host has no FPU)
#define M_run   ((int64_t *)           0xA0600000UL)   // Mq: running max in Q(1/1024) integer domain
#define L_run   ((int64_t *)           0xA0610000UL)   // Lq: running denom, int64 sum of iexp pseudo-probs
#define scale_factors ((uint32_t *)    0xA0700000UL)
#define SCALE_FACTORS_BYTES (1 << 22)

static inline float bf16_to_f(uint16_t b) {{
  union {{ uint32_t u; float f; }} v; v.u = ((uint32_t)b) << 16; return v.f;
}}
static inline uint32_t f_bits(float f) {{ union {{ uint32_t u; float f; }} v; v.f = f; return v.u; }}
static inline float bits_f(uint32_t u) {{ union {{ uint32_t u; float f; }} v; v.u = u; return v.f; }}
static inline float f_abs(float x) {{ return bits_f(f_bits(x) & 0x7FFFFFFFu); }}
static inline int f_exp(float x) {{ return (int)((f_bits(x) >> 23) & 0xFF) - 127; }}
static inline float pow2i(int e) {{
  if (e < -126) return 0.0f;
  if (e > 127) e = 127;
  return bits_f((uint32_t)(e + 127) << 23);
}}
static inline float exp_f(float x) {{
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float z = x * 1.44269504f;
  int n = (int)(z + (z >= 0 ? 0.5f : -0.5f));
  float f = z - (float)n;
  float p = 0.9999999916f + f*(0.6931471825f + f*(0.2401536316f + f*(0.0558263185f + f*(0.0089893397f + f*0.0018775767f))));
  return p * pow2i(n);
}}
static inline uint8_t fp8_e4m3_rtz(float x) {{
  if (x == 0.0f) return 0;
  uint8_t s = (f_bits(x) >> 31) ? 0x80 : 0;
  float a = f_abs(x);
  if (a >= 448.0f) return s | 0x7E;
  int e = f_exp(a);
  if (e < -6) return s;
  float m = a * pow2i(-e);
  int mant = (int)((m - 1.0f) * 8.0f);
  return s | (uint8_t)(((e + 7) << 3) | (mant & 7));
}}

// --- float-FREE integer online softmax (I-BERT iexp; host has NO FPU, float traps on RTL) ---
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165
#define MX_FRAC 24
#define MX_P_TARGET_LOG2 4
#define MX_O_F 16                 /* O_acc fixed-point fractional bits */
#define MX_IEXP0 2924390          /* iexp(0) = QB*QB+QC */
static inline int64_t mx_ibf16_to_q(uint16_t u) {{
  int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F; int64_t mant=(e==0)?m:(0x80|m); int sh=(int)e-127-7+10;
  int64_t q=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh)); return s?-q:q;
}}
static inline int mx_ilog2(uint64_t n) {{ int r=0;
  if(n>=((uint64_t)1<<32)){{n>>=32;r+=32;}} if(n>=((uint64_t)1<<16)){{n>>=16;r+=16;}}
  if(n>=((uint64_t)1<<8)){{n>>=8;r+=8;}} if(n>=((uint64_t)1<<4)){{n>>=4;r+=4;}}
  if(n>=((uint64_t)1<<2)){{n>>=2;r+=2;}} if(n>=((uint64_t)1<<1)){{r+=1;}} return r; }}
static inline int64_t mx_iexp(int64_t qa) {{ if(qa>0)qa=0;
  int64_t z=(-qa*MX_QLN2_INV)>>16; int64_t qp=qa+z*MX_QLN2; if(z>=63)return 0; return ((qp+MX_QB)*(qp+MX_QB)+MX_QC)>>z; }}
static inline int64_t mx_bf16_to_fx(uint16_t u,int F) {{   // bf16 -> value * 2^F (integer, rounded)
  int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F; int64_t mant=(e==0)?m:(0x80|m); int sh=(int)e-127-7+F;
  int64_t v=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh)); return s?-v:v; }}
static inline uint8_t mx_enc_unnorm(int64_t pt,int eg) {{   // fp8 e4m3 RNE of (pt * 2^-eg), pt>=0
  if(pt<=0)return 0; int sh=MX_FRAC-eg;
  uint64_t R=(sh>=0)?((uint64_t)pt<<sh):((uint64_t)pt>>(-sh)); if(R==0)return 0;
  int top=mx_ilog2(R); uint8_t M;
  if(top>=4){{M=(uint8_t)((R>>(top-3))&7); if((R>>(top-4))&1){{M++; if(M==8){{M=0;top++;}}}}}}
  else M=(uint8_t)((R<<(3-top))&7);
  int E=top-MX_FRAC+7;
  if(E<1){{int rs=1-E,s2=top-3+rs; return (rs<=3&&s2>=0)?(uint8_t)((R>>s2)&7):0;}}
  if(E>15)return 0x7E; if(E==15&&M==7)M=6; return (uint8_t)((E<<3)|(M&7));
}}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  memset(O_acc, 0, ATT_S * ATT_D * sizeof(int64_t));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {{
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }}
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < ATT_S; i++)
    for (int j = 0; j < ATT_D; j++)
      printf("%08x\n", (uint32_t)O_acc[i][j]);   // integer fixed-point Q(MX_O_F), low 32b; no float
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''

_KERNEL_BODY = r'''  // ---- flash attention: tile K/V along sequence, online softmax ----
  for (int i = 0; i < ATT_S; i++) { M_run[i] = -((int64_t)1 << 62); L_run[i] = 0; }
  memset(O_acc, 0, ATT_S * ATT_D * sizeof(int64_t));

  for (int t = 0; t < N_TILES; t++) {
    // 1) scores tile S1 = Q @ K_t^T   [S x BT] on Gemmini
    {
      int tiles_I = ATT_S / DIM, tiles_J = BT / DIM, tiles_K = ATT_D / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
      // KT_scales_col is [ATT_D/32][ATT_S] (row stride ATT_S, NOT BT): a contiguous slice at
      // [0][t*BT] reads group 1's scales from group 0's row -> wrong scales on half the
      // K-reduction. Repack the tile's column scales into the group-major [g*BT + n] layout
      // the HW expects for an (ATT_D x BT) matmul (scale_b_mem[group*N_DIM + n], N_DIM=BT).
      static uint8_t KT_sc_tile[(ATT_D / 32) * BT];
      for (int sg = 0; sg < ATT_D / 32; sg++)
        for (int sn = 0; sn < BT; sn++)
          KT_sc_tile[sg * BT + sn] = KT_scales_col[sg][t * BT + sn];
      gemmini_mx_load_scales((uint64_t)KT_sc_tile, (ATT_D / 32) * BT, 1);
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S; tile cols [t*BT..]
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + t * BT + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * BT);
      gemmini_fence();
    }

    // 2) FLOAT-FREE integer online softmax update + fp8 requant of P_t (host has no FPU)
    for (int i = 0; i < ATT_S; i++) {
      int64_t q[BT];
      int64_t m_new = M_run[i];
      for (int j = 0; j < BT; j++) {
        int64_t qq = mx_ibf16_to_q((uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF));
        q[j] = qq; if (qq > m_new) m_new = qq;
      }
      int64_t corr = mx_iexp(M_run[i] - m_new);          // = exp(M_old - m_new) * IEXP0
      int64_t p[BT];
      int64_t st = 0;
      for (int j = 0; j < BT; j++) { int64_t pv = mx_iexp(q[j] - m_new); p[j] = pv; st += pv; }
      L_run[i] = (L_run[i] * corr) / MX_IEXP0 + st;       // rescale prev denom + add this tile
      for (int d = 0; d < ATT_D; d++) O_acc[i][d] = (O_acc[i][d] * corr) / MX_IEXP0;  // rescale prev output
      M_run[i] = m_new;
      // per-32 group e8m0 scale + integer fp8 e4m3 RNE of P_t (UNNORMALIZED; normalize deferred to /Lq)
      for (int g = 0; g < GROUPS_BT; g++) {
        int64_t gmax = 0;
        for (int j = g * 32; j < (g + 1) * 32; j++) if (p[j] > gmax) gmax = p[j];
        int eg = (gmax > 0) ? (mx_ilog2((uint64_t)gmax) - MX_P_TARGET_LOG2) : 0;
        P_s[g][i] = (uint8_t)(eg + 127);
        for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = mx_enc_unnorm(p[j], eg);
      }
    }

    // 3) O_tile = P_t @ V_t  [S x D] on Gemmini; accumulate in fp32
    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = BT / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      int SPAD_DEST = 128;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)P_s, GROUPS_BT * ATT_S, 0);
      gemmini_mx_load_scales((uint64_t)&V_scales_col[t * GROUPS_BT][0], GROUPS_BT * ATT_D, 1);
      gemmini_config_ld(BT * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * BT + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D; tile rows [t*BT..]
      for (int j = 0; j < tiles_J; j++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)V_in) + (t * BT + k * DIM) * ATT_D + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_config_st(D_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
      gemmini_fence();
      for (int i = 0; i < ATT_S; i++)
        for (int d = 0; d < ATT_D; d++) {
          uint16_t b = (uint16_t)((O_hw[i][d / 4] >> ((d % 4) * 16)) & 0xFFFF);
          O_acc[i][d] += mx_bf16_to_fx(b, MX_O_F);   // bf16 P_t@V_t -> Q(MX_O_F) integer
        }
    }
  }

  // final normalization by the online denominator
  for (int i = 0; i < ATT_S; i++) {
    int64_t l = L_run[i]; if (l == 0) l = 1;
    for (int d = 0; d < ATT_D; d++) O_acc[i][d] = O_acc[i][d] / l;  // O*2^MX_O_F (fixed-point); integer divide
  }
'''

_BASELINE = '''// AUTO-GENERATED FLASH attention baseline (S={S}, D={D}, tile={BT}).
void solution(void) {{
{body}}}
'''


def gen_flash_attention_problem(S, D, prob_type="gemmini-mx-flash-attn", prob_id=0, fmt="fp8:e4m3"):
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        return _gen_flash_fp64(S, D, prob_type, prob_id, fmt)  # fp6/fp4 path
    if S % BT or D % 32:
        raise ValueError(f"S must be multiple of {BT}, D of 32 (got S={S}, D={D})")

    header_qk = f"mxgen_fla_qk_{S}x{D}.h"
    header_v = f"mxgen_fla_v_{S}x{D}.h"
    gen_input_header(S, D, S, fmt, GEMMINI_SW / "include" / header_qk)
    gen_input_header(S, S, D, fmt, GEMMINI_SW / "include" / header_v)
    _rename_header(GEMMINI_SW / "include" / header_qk, {
        "A_in": "Q_in", "B_in": "KT_in",
        "A_scales_row": "Q_scales_row", "B_scales_col": "KT_scales_col",
        "C_out": "QK_C_out", "C_scales_row": "QK_C_scales_row", "C_out_bf16": "QK_C_out_bf16",
        "MATMUL_M": "QK_M", "MATMUL_K": "QK_K", "MATMUL_N": "QK_N",
        "MATMUL_GK": "QK_GK", "MATMUL_GN": "QK_GN",
    }, "FLA_QK")
    _rename_header(GEMMINI_SW / "include" / header_v, {
        "A_in": "Pdummy_in", "B_in": "V_in",
        "A_scales_row": "Pdummy_scales_row", "B_scales_col": "V_scales_col",
        "C_out": "V_C_out", "C_scales_row": "V_C_scales_row", "C_out_bf16": "V_C_out_bf16",
        "MATMUL_M": "V_M", "MATMUL_K": "V_K", "MATMUL_N": "V_N",
        "MATMUL_GK": "V_GK", "MATMUL_GN": "V_GN",
    }, "FLA_V")

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    gold_file = hdir / f"gold{prob_id}.txt"

    (hdir / f"test{prob_id}.c").write_text(_HARNESS.format(
        S=S, D=D, BT=BT, header_qk=header_qk, header_v=header_v, gold_file=gold_file))
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        _BASELINE.format(S=S, D=D, BT=BT, body=_KERNEL_BODY))

    # capture gold: run the baseline through the same harness path
    from autocomp.search.prob import Prob
    from autocomp.backend.gemmini.gemmini_eval import clean_code
    prob = Prob(prob_type, prob_id)
    body = clean_code((sdir / f"sol{prob_id}_exo_baseline.c").read_text())
    tc = prob.tests[0].get_test_code([body])
    tc = tc.replace("// GOLD_FILE:", "// CAPTURING (no gold check):")
    rd = {}
    E.run_spike(tc, rd, GEMMINI_PATH, "0", 3600)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"flash-attn gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"\b[0-9a-fA-F]{8}\b",
                      out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    if len(vals) != S * D:
        raise RuntimeError(f"flash-attn gold: expected {S*D} words, got {len(vals)}")
    gold_file.write_text("\n".join(vals))
    return prob_type, prob_id


# ----------------------------------------------------------------------------
# fp6 / fp4 flash attention (sub-byte microscaling). Combines: flash tiling +
# online softmax + per-tile KT-scale repack (the fp8 flash fix), with fp6/fp4
# packing (Q M-packed once, KT/V tile-sliced + sub-byte) and per-tile dynamic P
# quantization (fp4 e2m1 / fp6 identity-LUT nearest-index). Validated 2026-06-04:
# fp4 100%, fp6 90% within-5% of a true online-softmax ref (S=128 D=64).
# ----------------------------------------------------------------------------
def _flash_fp64_pre(fmt, hqk, hv, S, D, gold_marker):
    base = ('#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n'
        '#ifndef BAREMETAL\n#include <sys/mman.h>\n#include <stdlib.h>\n#endif\n'
        '#include "include/gemmini_testutils.h"\n#include "include/%HQK%"\n#include "include/%HV%"\n'
        '#define DIM 16\n#define ATT_S %S%\n#define ATT_D %D%\n#define BT %BT%\n'
        '#define NT (ATT_S/BT)\n#define GBT (BT/32)\n#define BF16_PER_WORD 4\n'
        'typedef uint8_t elem_t; typedef uint64_t out_t;\n#ifndef fence\n#define fence() gemmini_fence()\n#endif\n'
        '#define OUTPUT_MATRIX_NAME O_acc\n%GOLD%\n'
        'static elem_t Q_hw[ATT_S/2][ATT_D];\nstatic elem_t P_hw[ATT_S/2][BT];\n'
        'static uint8_t P_scales[GBT][ATT_S];\nstatic uint8_t KTsc[(ATT_D/32)*BT];\n'
        'static out_t S1_hw[ATT_S][BT/BF16_PER_WORD];\nstatic out_t O_hw[ATT_S][ATT_D/BF16_PER_WORD];\n'
        'static int64_t Mr[ATT_S],Lr[ATT_S],O_acc[ATT_S][ATT_D];\nstatic float gold[1][1];\n'
        'int full_is_equal(float x[ATT_S][ATT_D], const float y[ATT_S][ATT_D]){(void)x;(void)y;return 1;}\n'
        'static inline uint32_t fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}\n'
        'static inline float bff(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}\n'
        'static inline float fabsf2(float x){return bff(fb(x)&0x7FFFFFFFu);}\n'
        'static inline int fexp2(float x){return (int)((fb(x)>>23)&0xFF)-127;}\n'
        'static inline float pow2i(int e){if(e<-126)return 0.f;if(e>127)e=127;return bff((uint32_t)(e+127)<<23);}\n'
        'static inline float expf2(float x){if(x<-87.f)return 0.f;if(x>88.f)x=88.f;float z=x*1.44269504f;int n=(int)(z+(z>=0?0.5f:-0.5f));float f=z-(float)n;float p=0.9999999916f+f*(0.6931471825f+f*(0.2401536316f+f*(0.0558263185f+f*(0.0089893397f+f*0.0018775767f))));return p*pow2i(n);}\n'
        'static inline float bf16f(uint16_t b){return bff(((uint32_t)b)<<16);}\n'
        '#define MX_QLN2 710\n#define MX_QLN2_INV 92\n#define MX_QB 1385\n#define MX_QC 1006165\n#define MX_FRAC 24\n#define MX_O_F 16\n#define MX_IEXP0 2924390\n'
        'static inline int64_t mx_ibf16_to_q(uint16_t u){int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F;int64_t mant=(e==0)?m:(0x80|m);int sh=(int)e-127-7+10;int64_t q=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh));return s?-q:q;}\n'
        'static inline int mx_ilog2(uint64_t n){int r=0;if(n>=((uint64_t)1<<32)){n>>=32;r+=32;}if(n>=((uint64_t)1<<16)){n>>=16;r+=16;}if(n>=((uint64_t)1<<8)){n>>=8;r+=8;}if(n>=((uint64_t)1<<4)){n>>=4;r+=4;}if(n>=((uint64_t)1<<2)){n>>=2;r+=2;}if(n>=((uint64_t)1<<1)){r+=1;}return r;}\n'
        'static inline int64_t mx_iexp(int64_t a){if(a>0)a=0;int64_t z=(-a*MX_QLN2_INV)>>16;int64_t qp=a+z*MX_QLN2;if(z>=63)return 0;return((qp+MX_QB)*(qp+MX_QB)+MX_QC)>>z;}\n'
        'static inline int64_t mx_bf16_to_fx(uint16_t u,int F){int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F;int64_t mant=(e==0)?m:(0x80|m);int sh=(int)e-127-7+F;int64_t v=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh));return s?-v:v;}\n')
    if _FMT_HW[fmt]["lut"]:
        base += ('static uint8_t Q_lp[12],KT_lp[12],V_lp[12],P_lp[12],Pident[16];\n'
            'static inline float fp6d(uint8_t c){int s=(c>>5)&1,e=(c>>2)&7,m=c&3;float sg=s?-1.f:1.f;if(e==0)return sg*(m/4.f)*pow2i(-2);return sg*(1.f+m/4.f)*pow2i(e-3);}\n'
            'static void pl(const uint8_t*c,uint8_t*o){uint64_t lo=0;uint32_t hi=0;for(int i=0;i<16;i++){int bit=i*6;uint64_t v=c[i]&0x3F;if(bit+6<=64)lo|=v<<bit;else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}\n'
            'static inline uint8_t enc(float x){float best=1e30f;uint8_t bi=0;for(int i=0;i<16;i++){float v=fp6d(Pident[i]);float d=x-v;d=d<0?-d:d;if(d<best){best=d;bi=(uint8_t)i;}}return bi;}\n'
            'static const uint64_t MXTH6[15]={524288,1572864,2621440,3670016,4718592,5767168,6815744,7864320,9437184,11534336,13631488,15728640,18874368,23068672,27262976};\n'
            'static inline uint8_t mx_enc_P(uint64_t R){for(int i=0;i<15;i++)if(R<MXTH6[i])return (uint8_t)i;return 15;}\n')
    else:
        base += ('static inline float fp4d(uint8_t c){int s=(c>>3)&1,e=(c>>1)&3,m=c&1;float sg=s?-1.f:1.f;if(e==0)return sg*(m*0.5f);return sg*(1.f+m*0.5f)*pow2i(e-1);}\n'
            'static inline uint8_t enc(float x){if(x==0.f)return 0;uint8_t s=(fb(x)>>31)?0x8:0;float a=fabsf2(x);uint8_t c;if(a<0.5f)c=0;else if(a<1.f)c=1;else if(a<1.5f)c=2;else if(a<2.f)c=3;else if(a<3.f)c=4;else if(a<4.f)c=5;else if(a<6.f)c=6;else c=7;return s|c;}\n'
            'static const uint64_t MXTH4[7]={4194304,12582912,20971520,29360128,41943040,58720256,83886080};\n'
            'static inline uint8_t mx_enc_P(uint64_t R){for(int c=0;c<7;c++)if(R<MXTH4[c])return (uint8_t)c;return 7;}\n')
    return (base.replace("%HQK%", hqk).replace("%HV%", hv).replace("%S%", str(S))
            .replace("%D%", str(D)).replace("%BT%", str(BT)).replace("%GOLD%", gold_marker))


def _flash_fp64_prep(fmt):
    s = ('  for(int m=0;m<ATT_S;m++)for(int k=0;k<ATT_D;k++){uint8_t by=Q_in[m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);\n'
         '    if((m&1)==0)Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0xF0)|cd; else Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0x0F)|(cd<<4);}\n')
    if _FMT_HW[fmt]["lut"]:
        s += ('  for(int i=0;i<16;i++)Pident[i]=(uint8_t)i;\n'
              '  pl((const uint8_t*)Q_lut,Q_lp);pl((const uint8_t*)KT_lut,KT_lp);pl((const uint8_t*)V_lut,V_lp);pl(Pident,P_lp);\n')
    return s


def _flash_fp64_body(fmt, S, D):
    cfg = _FMT_HW[fmt]; lut = cfg["lut"]; gl = max(S, D).bit_length() if lut else 1
    qklut = "      gemmini_mx_load_lut((uint64_t)KT_lp,1,0); gemmini_mx_load_lut((uint64_t)Q_lp,1,1);\n" if lut else ""
    pvlut = "      gemmini_mx_load_lut((uint64_t)V_lp,1,0); gemmini_mx_load_lut((uint64_t)P_lp,1,1);\n" if lut else ""
    a, w, u = cfg["actf"], cfg["wgtf"], cfg["uselut"]
    body = (
        '  for(int i=0;i<ATT_S;i++){Mr[i]=-((int64_t)1<<62);Lr[i]=0;for(int d=0;d<ATT_D;d++)O_acc[i][d]=0;}\n'
        '  for(int t=0;t<NT;t++){\n'
        '    { int ti=ATT_S/DIM/2, tj=BT/DIM/2, tk=ATT_D/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;\n'
        '      for(int g=0;g<ATT_D/32;g++)for(int n=0;n<BT;n++)KTsc[g*BT+n]=KT_scales_col[g][t*BT+n];\n'
        '      gemmini_flush(0);\n'
        f'      gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,{a},{w},3,{u});\n'
        '      gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);\n'
        '      gemmini_mx_load_scales((uint64_t)KTsc,(ATT_D/32)*BT,1);\n'
        '      gemmini_config_st((BT/BF16_PER_WORD)*sizeof(out_t));\n'
        f'      gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,{gl});\n'
        f'{qklut}      gemmini_config_ld(ATT_D*sizeof(elem_t));\n'
        '      for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)Q_hw)+i*DIM*ATT_D+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);\n'
        '      gemmini_config_ld(ATT_S*sizeof(elem_t)/2);\n'
        '      for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)KT_in)+k*DIM*(ATT_S/2)+(t*BT)/2+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);\n'
        '      gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n'
        '      gemmini_mx_read_smem(&S1_hw[0][0],128*16,ATT_S*BT); gemmini_fence(); }\n'
        '    for(int i=0;i<ATT_S;i++){  /* FLOAT-FREE integer online softmax + fp6/4 requant */\n'
        '      int64_t q[BT]; int64_t mn=Mr[i];\n'
        '      for(int j=0;j<BT;j++){int64_t qq=mx_ibf16_to_q((uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF));q[j]=qq;if(qq>mn)mn=qq;}\n'
        '      int64_t corr=mx_iexp(Mr[i]-mn); int64_t p[BT],st=0;\n'
        '      for(int j=0;j<BT;j++){int64_t pv=mx_iexp(q[j]-mn);p[j]=pv;st+=pv;}\n'
        '      Lr[i]=(Lr[i]*corr)/MX_IEXP0+st;\n'
        '      for(int d=0;d<ATT_D;d++)O_acc[i][d]=(O_acc[i][d]*corr)/MX_IEXP0; Mr[i]=mn;\n'
        '      for(int g=0;g<GBT;g++){int64_t gm=0;for(int j=g*32;j<g*32+32;j++)if(p[j]>gm)gm=p[j];\n'
        '        int e=(gm>0)?(mx_ilog2((uint64_t)gm)-%FTGT%):0;P_scales[g][i]=(uint8_t)(e+127);\n'
        '        for(int j=g*32;j<g*32+32;j++){uint64_t Rt=(e>=0)?(((uint64_t)p[j]<<MX_FRAC)>>e):((uint64_t)p[j]<<(MX_FRAC-e));uint8_t idx=mx_enc_P(Rt);\n'
        '          if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|idx; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(idx<<4);}}}\n'
        '    { int ti=ATT_S/DIM/2, tj=ATT_D/DIM/2, tk=BT/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;\n'
        '      gemmini_flush(0);\n'
        f'      gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,{a},{w},3,{u});\n'
        '      gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);\n'
        '      gemmini_mx_load_scales((uint64_t)&V_scales_col[t*GBT][0],GBT*ATT_D,1);\n'
        '      gemmini_config_st((ATT_D/BF16_PER_WORD)*sizeof(out_t));\n'
        f'      gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,{gl});\n'
        f'{pvlut}      gemmini_config_ld(BT*sizeof(elem_t));\n'
        '      for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)P_hw)+i*DIM*BT+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);\n'
        '      gemmini_config_ld(ATT_D*sizeof(elem_t)/2);\n'
        '      for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)V_in)+(t*BT+k*DIM)*(ATT_D/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);\n'
        '      gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n'
        '      gemmini_mx_read_smem(&O_hw[0][0],128*16,ATT_S*ATT_D); gemmini_fence();\n'
        '      for(int i=0;i<ATT_S;i++)for(int d=0;d<ATT_D;d++){uint16_t b=(uint16_t)((O_hw[i][d/4]>>((d%4)*16))&0xFFFF);O_acc[i][d]+=mx_bf16_to_fx(b,MX_O_F);} } }\n'
        '  for(int i=0;i<ATT_S;i++){int64_t l=Lr[i];if(l==0)l=1;for(int d=0;d<ATT_D;d++)O_acc[i][d]=O_acc[i][d]/l;}\n')
    # fp4 (e2m1) collapses to all-zero P at small T; use 2.0. fp6's finer LUT is fine at 0.5.
    return body.replace("%T%", "0.5" if cfg["lut"] else "2.0").replace("%FTGT%", "0" if cfg["lut"] else "2")


def _gen_flash_fp64(S, D, prob_type, prob_id, fmt):
    if S % BT or D % 32:
        raise ValueError(f"fp6/fp4 flash needs S multiple of {BT}, D of 32; got S={S} D={D}")
    tag = _FMT_TAG[fmt]; lut = _FMT_HW[fmt]["lut"]
    hqk = f"mxgen_{tag}_flaqk_{S}x{D}.h"; hv = f"mxgen_{tag}_flav_{S}x{D}.h"
    gen_input_header(S, D, S, fmt, GEMMINI_SW / "include" / hqk)
    gen_input_header(S, S, D, fmt, GEMMINI_SW / "include" / hv)
    qk_map = {"A_in": "Q_in", "B_in": "KT_in", "A_scales_row": "Q_scales_row", "B_scales_col": "KT_scales_col",
              "MATMUL_M": "QK_M", "MATMUL_K": "QK_K", "MATMUL_N": "QK_N", "MATMUL_GK": "QK_GK", "MATMUL_GN": "QK_GN",
              "C_out": "QKC", "C_scales_row": "QKCS", "C_out_bf16": "QKCB"}
    v_map = {"A_in": "Pd_in", "B_in": "V_in", "A_scales_row": "Pd_sr", "B_scales_col": "V_scales_col",
             "MATMUL_M": "V_M", "MATMUL_K": "V_K", "MATMUL_N": "V_N", "MATMUL_GK": "V_GK", "MATMUL_GN": "V_GN",
             "C_out": "VC", "C_scales_row": "VCS", "C_out_bf16": "VCB"}
    if lut:
        qk_map.update({"A_lut": "Q_lut", "B_lut": "KT_lut", "C_lut": "QKC_lut"})
        v_map.update({"A_lut": "Pd_lut", "B_lut": "V_lut", "C_lut": "VC_lut"})
    _rename_header(GEMMINI_SW / "include" / hqk, qk_map, "FL64QK")
    _rename_header(GEMMINI_SW / "include" / hv, v_map, "FL64V")

    hdir = HARNESSES_DIR / prob_type; sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    gold_file = hdir / f"gold{prob_id}.txt"
    prep = _flash_fp64_prep(fmt); body = _flash_fp64_body(fmt, S, D)
    mlock = "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
    dump = ('  printf("GOLD_BEGIN\\n");\n  for(int i=0;i<ATT_S;i++)for(int d=0;d<ATT_D;d++)\n'
            '    printf("%08x\\n",(uint32_t)O_acc[i][d]);\n  printf("GOLD_END\\n");\n')

    cap_pre = _flash_fp64_pre(fmt, hqk, hv, S, D, "// CAPTURING")
    cap = (cap_pre + "\nint main(){\n" + mlock + "  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};\n"
           + prep + body + dump + "#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 600)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"{tag} flash gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"\b[0-9a-fA-F]{8}\b", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    if len(vals) != S * D:
        raise RuntimeError(f"{tag} flash gold: expected {S*D} words, got {len(vals)}")
    gold_file.write_text("\n".join(vals))

    opt_pre = _flash_fp64_pre(fmt, hqk, hv, S, D, f"// GOLD_FILE: {gold_file}")
    opt = (opt_pre + "\n#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\nint main(){\n" + mlock +
           "  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};\n" + prep +
           "  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){\n"
           "    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n" + dump +
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED baseline MX FLASH attention kernel (S={S}, D={D}, {fmt}).\nvoid solution(void) {{\n" + body + "}\n")
    return prob_type, prob_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--fmt", default="fp8:e4m3")
    a = ap.parse_args()
    pt, pid = gen_flash_attention_problem(a.S, a.D, prob_id=a.id, fmt=a.fmt)
    print(f"generated {pt} id={pid}")
