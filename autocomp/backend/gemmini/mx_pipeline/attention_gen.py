"""Generate an autocomp MX-Gemmini ATTENTION problem.

O = softmax(Q @ K^T) @ V  for a single head, S x D:
  1. MX matmul:  S1 = Q @ K^T            (Gemmini, bf16 out)
  2. CPU:        P  = softmax(S1) rows; quantize to fp8 + per-32-group scales
  3. MX matmul:  O  = P @ V              (Gemmini, bf16 out)

This is the flash-attention-style fused kernel skeleton; autocomp can optimize
tiling/overlap. Inputs come from two golden_model.py matmul headers, renamed
(Q/KT and V). Correctness oracle = hardware-captured gold for O.

Requires S, D multiples of 32 (K-group). Both matmuls must fit the scratchpad.

Usage:
    python -m autocomp.backend.gemmini.mx_pipeline.attention_gen --S 64 --D 64 --id 0
"""

import argparse
import pathlib
import re

from autocomp.common import HARNESSES_DIR, SOLS_DIR
from autocomp.backend.gemmini import gemmini_eval as E
from autocomp.backend.gemmini.mx_pipeline.matmul_gen import (
    GEMMINI_PATH, GEMMINI_SW, check_shape, gen_input_header, _FMT_TAG, _FMT_HW,
)


def _rename_header(path: pathlib.Path, mapping: dict, guard_suffix: str):
    text = path.read_text()
    for old, new in mapping.items():
        text = re.sub(rf"\b{old}\b", new, text)
    text = text.replace("MATMUL_DATA_H", f"MATMUL_DATA_{guard_suffix}_H")
    path.write_text(text)


# All colliding symbols a golden matmul header defines (every include redefines them).
_HDR_SYMS = ["A_in", "B_in", "A_scales_row", "B_scales_col", "C_out", "C_scales_row",
             "C_out_bf16", "MATMUL_M", "MATMUL_K", "MATMUL_N", "MATMUL_GK", "MATMUL_GN"]


def _rename_all(path: pathlib.Path, suffix: str, expose: dict):
    """Make a golden header safe to #include alongside others: rename EVERY colliding symbol to
    a unique suffixed name; `expose` maps the few we keep (e.g. {'A_in':'Q_in_h0'}) to their
    target names. Returns nothing (rewrites in place)."""
    mapping = {s: expose.get(s, f"_{s}_{suffix}") for s in _HDR_SYMS}
    _rename_header(path, mapping, suffix)


_HARNESS = r'''// AUTO-GENERATED MX-Gemmini attention harness (S={S}, D={D}).
// O = softmax(Q K^T) V. gold = baseline hardware output (captured).
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
#define BF16_PER_WORD 4
#define ATT_S {S}
#define ATT_D {D}
#define S_OUT_COLS (ATT_S / BF16_PER_WORD)
#define D_OUT_COLS (ATT_D / BF16_PER_WORD)
#define GROUPS_S (ATT_S / 32)
#define ATT_CAUSAL {causal}    /* 1 = causal mask (future cols -> prob 0); 0 = full attention */

typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME O_hw

static const out_t gold[ATT_S][D_OUT_COLS] = {{
{gold_init}
}};

// scratch: scores (bf16-packed), P quantized + scales
static out_t  S1_hw[ATT_S][S_OUT_COLS];
static elem_t P_q[ATT_S][ATT_S];
static uint8_t P_scales[GROUPS_S][ATT_S];

static inline float bf16_to_f(uint16_t b) {{
  union {{ uint32_t u; float f; }} v; v.u = ((uint32_t)b) << 16; return v.f;
}}

// --- libm-free float helpers (baremetal links without libm) ---
static inline uint32_t f_bits(float f) {{ union {{ uint32_t u; float f; }} v; v.f = f; return v.u; }}
static inline float bits_f(uint32_t u) {{ union {{ uint32_t u; float f; }} v; v.u = u; return v.f; }}
static inline float f_abs(float x) {{ return bits_f(f_bits(x) & 0x7FFFFFFFu); }}
static inline int f_exp(float x) {{ return (int)((f_bits(x) >> 23) & 0xFF) - 127; }}  // floor(log2|x|), normals
static inline float pow2i(int e) {{
  if (e < -126) return 0.0f;
  if (e > 127) e = 127;
  return bits_f((uint32_t)(e + 127) << 23);
}}
static inline float exp_f(float x) {{
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float z = x * 1.44269504f;                  // x / ln2
  int n = (int)(z + (z >= 0 ? 0.5f : -0.5f));
  float f = z - (float)n;                     // [-0.5, 0.5]
  float p = 0.9999999916f + f*(0.6931471825f + f*(0.2401536316f + f*(0.0558263185f + f*(0.0089893397f + f*0.0018775767f))));
  return p * pow2i(n);
}}

// fp8 e4m3 round-toward-zero encode of x (|x| <= 448 after scaling)
static inline uint8_t fp8_e4m3_rtz(float x) {{
  if (x == 0.0f) return 0;
  uint8_t s = (f_bits(x) >> 31) ? 0x80 : 0;
  float a = f_abs(x);
  if (a >= 448.0f) return s | 0x7E;
  int e = f_exp(a);
  if (e < -6) return s;                  // flush subnormal
  float m = a * pow2i(-e);               // [1, 2)
  int mant = (int)((m - 1.0f) * 8.0f);
  return s | (uint8_t)(((e + 7) << 3) | (mant & 7));
}}

// --- float-FREE integer softmax (I-BERT iexp + integer fp8 e4m3 encode) ---
// The host Rocket in RadianceGemminiOnlyConfig has NO FPU (fpu=None); hard-float traps on RTL
// (verified: fpu_probe FAILED on VCS, int_probe passed). So the softmax MUST be integer-only.
// This is the I-BERT integer exp (== gemmini.cc:2293 apply_iexp / gemmini.h:1127 HW softmax),
// with S_q = 1/1024 (validated to match/beat the float softmax). int64 throughout (RV64 host).
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165
#define MX_FRAC 24
#define MX_P_TARGET_LOG2 4   /* P group-max -> ~2^4=16, NOT fp8-max: avoids e4 acc overflow */
static inline int64_t mx_ibf16_to_q(uint16_t u) {{   // bf16 bits -> Q(1/1024) fixed-point int
  int s = (u >> 15) & 1, e = (u >> 7) & 0xFF, m = u & 0x7F;
  int64_t mant = (e == 0) ? m : (0x80 | m);
  int sh = (int)e - 127 - 7 + 10;                    // value * 2^SL, SL=10
  int64_t q = (sh >= 0) ? (mant << sh)
                        : ((mant + ((int64_t)1 << (-sh - 1))) >> (-sh));  // round-nearest
  return s ? -q : q;
}}
static inline int mx_ilog2(uint64_t n) {{   // floor(log2 n), n>0; no __builtin_clz (baremetal -nostdlib has no libgcc __clzdi2)
  int r = 0;
  if (n >= ((uint64_t)1 << 32)) {{ n >>= 32; r += 32; }}
  if (n >= ((uint64_t)1 << 16)) {{ n >>= 16; r += 16; }}
  if (n >= ((uint64_t)1 << 8))  {{ n >>= 8;  r += 8;  }}
  if (n >= ((uint64_t)1 << 4))  {{ n >>= 4;  r += 4;  }}
  if (n >= ((uint64_t)1 << 2))  {{ n >>= 2;  r += 2;  }}
  if (n >= ((uint64_t)1 << 1))  {{ r += 1; }}
  return r;
}}
static inline int64_t mx_iexp(int64_t qa) {{          // qa <= 0; returns integer pseudo-prob
  int64_t z  = (-qa * MX_QLN2_INV) >> 16;
  int64_t qp = qa + z * MX_QLN2;
  int64_t qe = (qp + MX_QB) * (qp + MX_QB) + MX_QC;
  if (z >= 63) return 0;
  return qe >> z;
}}
static inline uint8_t mx_ienc_fp8(int64_t p, uint64_t inv_sm, int ls, int e) {{  // t=(p/sm)*2^-e -> fp8 e4m3 RNE
  if (p <= 0) return 0;
  uint64_t R = (((uint64_t)p * inv_sm) + ((uint64_t)1 << (ls - 1))) >> ls;  // ==round((p<<MX_FRAC)/sm); rounded reciprocal+R
  if (R == 0) return 0;
  int top = mx_ilog2(R);
  uint8_t M;
  if (top >= 4) {{                                   // round-nearest-even on the 3-bit mantissa
    M = (uint8_t)((R >> (top - 3)) & 0x7);
    if ((R >> (top - 4)) & 1) {{ M++; if (M == 8) {{ M = 0; top++; }} }}   // mantissa carry -> bump exponent
  }} else {{
    M = (uint8_t)((R << (3 - top)) & 0x7);
  }}
  int lg = top - MX_FRAC - e, E = lg + 7;
  if (E < 1) {{ int rs = 1 - E, sh = top - 3 + rs; return (rs <= 3 && sh >= 0) ? (uint8_t)((R >> sh) & 0x7) : 0; }}
  if (E > 15) return 0x7E;
  if (E == 15 && M == 7) M = 6;
  return (uint8_t)((E << 3) | (M & 7));
}}

int full_is_equal(out_t x[ATT_S][D_OUT_COLS], const out_t y[ATT_S][D_OUT_COLS]) {{
  for (int i = 0; i < ATT_S; i++)
    for (int j = 0; j < D_OUT_COLS; j++)
      if (x[i][j] != y[i][j]) return 0;
  return 1;
}}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  static out_t O_hw[ATT_S][D_OUT_COLS];
  uint32_t scale_factors[512] = {{0}};
  int SPAD_DEST = 128;
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {{
    memset(O_hw, 0, sizeof(O_hw));
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }}
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''

_KERNEL_BODY = r'''  // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);
    gemmini_config_ld(ATT_D * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
    gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S -> slot (k*tiles_J+j)
    for (int k = 0; k < tiles_K; k++)
      for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
    gemmini_config_st(S_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  // ---- 2) FLOAT-FREE integer softmax + fp8 requant (I-BERT iexp; host has no FPU) ----
  for (int i = 0; i < ATT_S; i++) {
    int64_t q[ATT_S];
    int64_t mq = -((int64_t)1 << 62);
    for (int j = 0; j < ATT_S; j += 4) {                       // unpack 4 bf16/word -> Q(1/1024) int + max
      uint64_t w = S1_hw[i][j / 4];
      for (int l = 0; l < 4; l++) {
#if ATT_CAUSAL
        if (j + l > i) { q[j + l] = -((int64_t)1 << 62); continue; }   // causal: future col -> iexp()=0
#endif
        int64_t qq = mx_ibf16_to_q((uint16_t)((w >> (l * 16)) & 0xFFFF));
        q[j + l] = qq; if (qq > mq) mq = qq;
      }
    }
    int64_t p[ATT_S];
    int64_t sm = 0;
    int64_t gmax_arr[GROUPS_S];                                // fused group-max: capture per-group max IN the iexp pass
    for (int g = 0; g < GROUPS_S; g++) {                       // (group-nested, register-scalar gmax) -> removes the
      int64_t gmax = 0;                                        // separate 16384-elem max scan. Output bit-identical.
      for (int j = g * 32; j < (g + 1) * 32; j++) { int64_t pv = mx_iexp(q[j] - mq); p[j] = pv; sm += pv; if (pv > gmax) gmax = pv; }
      gmax_arr[g] = gmax;
    }
    int ls = mx_ilog2((uint64_t)sm);
    uint64_t inv_sm = (((uint64_t)1 << (ls + MX_FRAC)) + ((uint64_t)sm >> 1)) / (uint64_t)sm;   // one rounded reciprocal/row
    for (int g = 0; g < GROUPS_S; g++) {
      int64_t gmax = gmax_arr[g];
      if (gmax == 0) { P_scales[g][i] = 0; for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = 0; continue; }
      // group e8m0 scale so the group-max lands at ~2^MX_P_TARGET_LOG2 (moderate; avoids e4 overflow)
      int e = mx_ilog2((uint64_t)gmax) - ls - MX_P_TARGET_LOG2;
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = mx_ienc_fp8(p[j], inv_sm, ls, e);
    }
  }

  // ---- 3) O = P @ V on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = ATT_S / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col, sizeof(V_scales_col), 1);
    gemmini_config_ld(ATT_S * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
    gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D -> slot (k*tiles_J+j)
    for (int k = 0; k < tiles_K; k++)
      for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
        gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&{OUT}[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
'''

_BASELINE = '''// AUTO-GENERATED baseline MX attention kernel (S={S}, D={D}).
void solution(void) {{
{body}}}
'''


def gen_attention_problem(S, D, prob_type="gemmini-mx-attention", prob_id=0, fmt="fp8:e4m3",
                          causal=False):
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        return _gen_attention_fp64(S, D, prob_type, prob_id, fmt)  # fp6/fp4 path
    for (m, k, n) in [(S, D, S), (S, S, D)]:
        errs = check_shape(m, k, n)
        if errs:
            raise ValueError(f"attention matmul {m}x{k}x{n} unsupported: {'; '.join(errs)}")

    header_qk = f"mxgen_att_qk_{S}x{D}.h"
    header_v = f"mxgen_att_v_{S}x{D}.h"
    gen_input_header(S, D, S, fmt, GEMMINI_SW / "include" / header_qk)
    gen_input_header(S, S, D, fmt, GEMMINI_SW / "include" / header_v)
    _rename_header(GEMMINI_SW / "include" / header_qk, {
        "A_in": "Q_in", "B_in": "KT_in",
        "A_scales_row": "Q_scales_row", "B_scales_col": "KT_scales_col",
        "C_out": "QK_C_out", "C_scales_row": "QK_C_scales_row", "C_out_bf16": "QK_C_out_bf16",
        "MATMUL_M": "QK_M", "MATMUL_K": "QK_K", "MATMUL_N": "QK_N",
        "MATMUL_GK": "QK_GK", "MATMUL_GN": "QK_GN",
    }, "QK")
    _rename_header(GEMMINI_SW / "include" / header_v, {
        "A_in": "Pdummy_in", "B_in": "V_in",
        "A_scales_row": "Pdummy_scales_row", "B_scales_col": "V_scales_col",
        "C_out": "V_C_out", "C_scales_row": "V_C_scales_row", "C_out_bf16": "V_C_out_bf16",
        "MATMUL_M": "V_M", "MATMUL_K": "V_K", "MATMUL_N": "V_N",
        "MATMUL_GK": "V_GK", "MATMUL_GN": "V_GN",
    }, "V")

    base_kwargs = dict(S=S, D=D, header_qk=header_qk, header_v=header_v, causal=int(causal))
    capture_src = _HARNESS.format(gold_init="  {0}", **base_kwargs).replace(
        "// SUBSTITUTE HERE",
        _KERNEL_BODY.replace("{OUT}", "O_hw")
        + '\n  printf("GOLD_BEGIN\\n");\n'
          "  for (int i = 0; i < ATT_S; i++)\n"
          "    for (int j = 0; j < D_OUT_COLS; j++)\n"
          '      printf("%016llx\\n", (unsigned long long) O_hw[i][j]);\n'
          '  printf("GOLD_END\\n");'
    )
    rd = {}
    E.run_spike(capture_src, rd, GEMMINI_PATH, "0", 300)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"attention gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    cols = D // 4
    if len(vals) != S * cols:
        raise RuntimeError(f"attention gold: expected {S*cols} words, got {len(vals)}")
    gold_init = ",\n".join(
        "  { " + ", ".join("0x" + vals[i * cols + j] + "ULL" for j in range(cols)) + " }"
        for i in range(S)
    )

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(_HARNESS.format(gold_init=gold_init, **base_kwargs))
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        _BASELINE.format(S=S, D=D, body=_KERNEL_BODY.replace("{OUT}", "O_hw")))
    return prob_type, prob_id


# ----------------------------------------------------------------------------
# fp6 / fp4 attention (sub-byte microscaling). QK and PV reuse the fp6/fp4
# matmul packing (Q=A M-packed, KT/V=B N/2-packed, fp6 LUTs). The new piece is
# dynamic P quantization: softmax -> per-32-group scale (target 0.5 to dodge the
# e4 block-accumulator overflow) -> fp4 e2m1 RTZ encode / fp6 nearest-index into
# an IDENTITY LUT (e3m2 codes 0..15 == values [0,1.75], covering probabilities)
# -> M-pack. Validated 2026-06-04: fp4 100%, fp6 84% within-5% of true ref.
# ----------------------------------------------------------------------------
def _attn_fp64_preamble(fmt, hqk, hv, S, D):
    base = ('#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n'
        '#ifndef BAREMETAL\n#include <sys/mman.h>\n#include <stdlib.h>\n#endif\n'
        '#include "include/gemmini_testutils.h"\n#include "include/%HQK%"\n#include "include/%HV%"\n'
        '#define DIM 16\n#define ATT_S %S%\n#define ATT_D %D%\n#define GKP (ATT_S/32)\n'
        '#define BF16_PER_WORD 4\n#define OUT_COLS (ATT_D / BF16_PER_WORD)\n'
        'typedef uint8_t elem_t; typedef uint64_t out_t;\n#ifndef fence\n#define fence() gemmini_fence()\n#endif\n'
        'static elem_t Q_hw[ATT_S/2][ATT_D];\nstatic elem_t P_hw[ATT_S/2][ATT_S];\n'
        'static uint8_t P_scales[GKP][ATT_S];\nstatic out_t S1_hw[ATT_S][ATT_S/BF16_PER_WORD];\n'
        'static inline uint32_t fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}\n'
        'static inline float bff(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}\n'
        'static inline float fabsf2(float x){return bff(fb(x)&0x7FFFFFFFu);}\n'
        'static inline int fexp2(float x){return (int)((fb(x)>>23)&0xFF)-127;}\n'
        'static inline float pow2i(int e){if(e<-126)return 0.f;if(e>127)e=127;return bff((uint32_t)(e+127)<<23);}\n'
        'static inline float expf2(float x){if(x<-87.f)return 0.f;if(x>88.f)x=88.f;float z=x*1.44269504f;int n=(int)(z+(z>=0?0.5f:-0.5f));float f=z-(float)n;float p=0.9999999916f+f*(0.6931471825f+f*(0.2401536316f+f*(0.0558263185f+f*(0.0089893397f+f*0.0018775767f))));return p*pow2i(n);}\n'
        'static inline float bf16f(uint16_t b){return bff(((uint32_t)b)<<16);}\n'
        # --- float-FREE integer softmax helpers (host has no FPU); I-BERT iexp, S_q=1/1024 ---
        '#define MX_QLN2 710\n#define MX_QLN2_INV 92\n#define MX_QB 1385\n#define MX_QC 1006165\n#define MX_FRAC 24\n'
        'static inline int64_t mx_ibf16_to_q(uint16_t u){int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F;int64_t mant=(e==0)?m:(0x80|m);int sh=(int)e-127-7+10;int64_t q=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh));return s?-q:q;}\n'
        'static inline int mx_ilog2(uint64_t n){int r=0;if(n>=((uint64_t)1<<32)){n>>=32;r+=32;}if(n>=((uint64_t)1<<16)){n>>=16;r+=16;}if(n>=((uint64_t)1<<8)){n>>=8;r+=8;}if(n>=((uint64_t)1<<4)){n>>=4;r+=4;}if(n>=((uint64_t)1<<2)){n>>=2;r+=2;}if(n>=((uint64_t)1<<1)){r+=1;}return r;}\n'
        'static inline int64_t mx_iexp(int64_t a){if(a>0)a=0;int64_t z=(-a*MX_QLN2_INV)>>16;int64_t qp=a+z*MX_QLN2;if(z>=63)return 0;return((qp+MX_QB)*(qp+MX_QB)+MX_QC)>>z;}\n')
    if _FMT_HW[fmt]["lut"]:
        base += ('static elem_t Q_lp[12],KT_lp[12],V_lp[12],P_lp[12]; static uint8_t Pident[16];\n'
            'static inline float fp6d(uint8_t c){int s=(c>>5)&1,e=(c>>2)&7,m=c&3;float sg=s?-1.f:1.f;if(e==0)return sg*(m/4.f)*pow2i(-2);return sg*(1.f+m/4.f)*pow2i(e-3);}\n'
            'static void packlut(const uint8_t*c,uint8_t*o){uint64_t lo=0;uint32_t hi=0;for(int i=0;i<16;i++){int bit=i*6;uint64_t v=c[i]&0x3F;if(bit+6<=64)lo|=v<<bit;else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}\n'
            # integer fp6 nearest-LUT-index encode of value=Rt/2^MX_FRAC (thresholds = midpoints*2^24)
            'static const uint64_t MXTH6[15]={524288,1572864,2621440,3670016,4718592,5767168,6815744,7864320,9437184,11534336,13631488,15728640,18874368,23068672,27262976};\n'
            'static inline uint8_t mx_enc_P(uint64_t R){for(int i=0;i<15;i++)if(R<MXTH6[i])return (uint8_t)i;return 15;}\n')
    else:
        base += ('static inline float fp4d(uint8_t c){int s=(c>>3)&1,e=(c>>1)&3,m=c&1;float sg=s?-1.f:1.f;if(e==0)return sg*(m*0.5f);return sg*(1.f+m*0.5f)*pow2i(e-1);}\n'
            # integer fp4 e2m1 encode of value=Rt/2^MX_FRAC (thresholds*2^24)
            'static const uint64_t MXTH4[7]={4194304,12582912,20971520,29360128,41943040,58720256,83886080};\n'
            'static inline uint8_t mx_enc_P(uint64_t R){for(int c=0;c<7;c++)if(R<MXTH4[c])return (uint8_t)c;return 7;}\n')
    return base.replace("%HQK%", hqk).replace("%HV%", hv).replace("%S%", str(S)).replace("%D%", str(D))


def _attn_fp64_prep(fmt):
    s = ('  for(int m=0;m<ATT_S;m++)for(int k=0;k<ATT_D;k++){uint8_t by=Q_in[m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);\n'
         '    if((m&1)==0)Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0xF0)|cd; else Q_hw[m>>1][k]=(Q_hw[m>>1][k]&0x0F)|(cd<<4);}\n')
    if _FMT_HW[fmt]["lut"]:
        s += ('  for(int i=0;i<16;i++)Pident[i]=(uint8_t)i;\n'
              '  packlut((const uint8_t*)Q_lut,Q_lp);packlut((const uint8_t*)KT_lut,KT_lp);packlut((const uint8_t*)V_lut,V_lp);packlut(Pident,P_lp);\n')
    return s


def _attn_fp64_body(fmt, S, D):
    cfg = _FMT_HW[fmt]; lut = cfg["lut"]
    # granularity must cover both QK (M=N=S) and PV (M=S,N=D): use max(S,D) so every
    # element maps to the single shared LUT group even when S is not a power of two.
    gl = max(S, D).bit_length() if lut else 1
    qklut = "    gemmini_mx_load_lut((uint64_t)KT_lp,1,0); gemmini_mx_load_lut((uint64_t)Q_lp,1,1);\n" if lut else ""
    pvlut = "    gemmini_mx_load_lut((uint64_t)V_lp,1,0); gemmini_mx_load_lut((uint64_t)P_lp,1,1);\n" if lut else ""
    body = (
        '  { int ti=ATT_S/DIM/2, tj=ATT_S/DIM/2, tk=ATT_D/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;\n'
        '    gemmini_flush(0);\n'
        '    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,%ACTF%,%WGTF%,3,%USELUT%);\n'
        '    gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);\n'
        '    gemmini_mx_load_scales((uint64_t)&KT_scales_col,sizeof(KT_scales_col),1);\n'
        '    gemmini_config_st((ATT_S/BF16_PER_WORD)*sizeof(out_t));\n'
        '    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,%GL%);\n'
        '%QKLUT%    gemmini_config_ld(ATT_D*sizeof(elem_t));\n'
        '    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)Q_hw)+i*DIM*ATT_D+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);\n'
        '    gemmini_config_ld(ATT_S*sizeof(elem_t)/2);\n'
        '    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)KT_in)+k*DIM*(ATT_S/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);\n'
        '    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n'
        '    gemmini_mx_read_smem(&S1_hw[0][0],SD*16,ATT_S*ATT_S); gemmini_fence(); }\n'
        '  for(int i=0;i<ATT_S;i++){  /* FLOAT-FREE integer softmax + fp%FMTBITS% requant (host has no FPU) */\n'
        '    int64_t q[ATT_S]; int64_t mq=-((int64_t)1<<62);\n'
        '    for(int j=0;j<ATT_S;j++){int64_t qq=mx_ibf16_to_q((uint16_t)((S1_hw[i][j/4]>>((j%4)*16))&0xFFFF));q[j]=qq;if(qq>mq)mq=qq;}\n'
        '    int64_t p[ATT_S],sm=0; int64_t gmarr[GKP];   /* fused group-max: capture per-group max in the iexp pass */\n'
        '    for(int g=0;g<GKP;g++){{int64_t gm=0;for(int j=g*32;j<g*32+32;j++){{int64_t pv=mx_iexp(q[j]-mq);p[j]=pv;sm+=pv;if(pv>gm)gm=pv;}}gmarr[g]=gm;}}\n'
        '    int ls=mx_ilog2((uint64_t)sm); uint64_t inv=(((uint64_t)1<<(ls+MX_FRAC))+((uint64_t)sm>>1))/(uint64_t)sm;\n'
        '    for(int g=0;g<GKP;g++){ int64_t gm=gmarr[g];\n'
        '      int e=(gm>0)?(mx_ilog2((uint64_t)gm)-ls+%POFF%):0; P_scales[g][i]=(uint8_t)(e+127);\n'
        '      for(int j=g*32;j<g*32+32;j++){ uint64_t R=(((uint64_t)p[j]*inv)+((uint64_t)1<<(ls-1)))>>ls;\n'
        '        uint64_t Rt=(e>=0)?(R>>e):(R<<(-e)); uint8_t idx=mx_enc_P(Rt);\n'
        '        if((i&1)==0)P_hw[i>>1][j]=(P_hw[i>>1][j]&0xF0)|idx; else P_hw[i>>1][j]=(P_hw[i>>1][j]&0x0F)|(idx<<4); } } }\n'
        '  { int ti=ATT_S/DIM/2, tj=ATT_D/DIM/2, tk=ATT_S/DIM; uint32_t ab=0, bb=BANK_NUM*BANK_ROWS-tk*tj*DIM; int SD=128;\n'
        '    gemmini_flush(0);\n'
        '    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,%ACTF%,%WGTF%,3,%USELUT%);\n'
        '    gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);\n'
        '    gemmini_mx_load_scales((uint64_t)&V_scales_col,sizeof(V_scales_col),1);\n'
        '    gemmini_config_st((ATT_D/BF16_PER_WORD)*sizeof(out_t));\n'
        '    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,%GL%);\n'
        '%PVLUT%    gemmini_config_ld(ATT_S*sizeof(elem_t));\n'
        '    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)P_hw)+i*DIM*ATT_S+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);\n'
        '    gemmini_config_ld(ATT_D*sizeof(elem_t)/2);\n'
        '    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)V_in)+k*DIM*(ATT_D/2)+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);\n'
        '    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,SD,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n'
        '    gemmini_mx_read_smem(&O_hw[0][0],SD*16,ATT_S*ATT_D); gemmini_fence(); }\n')
    # P-quant scale target: fp4 (e2m1, min nonzero 0.5) collapses to all-zero at small T,
    # so use 2.0; fp6's finer LUT (down to 0.0625) is fine at 0.5.
    tval = "0.5" if cfg["lut"] else "2.0"
    # integer group-scale offset matching the float e=fexp2(gm/T)+1: fp6(T=0.5)->+2, fp4(T=2.0)->0
    poff = "2" if cfg["lut"] else "0"
    fmtbits = "6" if cfg["lut"] else "4"
    return (body.replace("%ACTF%", str(cfg["actf"])).replace("%WGTF%", str(cfg["wgtf"]))
            .replace("%USELUT%", str(cfg["uselut"])).replace("%GL%", str(gl))
            .replace("%QKLUT%", qklut).replace("%PVLUT%", pvlut).replace("%T%", tval)
            .replace("%POFF%", poff).replace("%FMTBITS%", fmtbits))


def _gen_attention_fp64(S, D, prob_type, prob_id, fmt):
    if S % 32 or D % 32:
        raise ValueError(f"fp6/fp4 attention needs S,D multiples of 32; got S={S} D={D}")
    tag = _FMT_TAG[fmt]; lut = _FMT_HW[fmt]["lut"]
    hqk = f"mxgen_{tag}_attqk_{S}x{D}.h"; hv = f"mxgen_{tag}_attv_{S}x{D}.h"
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
    _rename_header(GEMMINI_SW / "include" / hqk, qk_map, "FP64QK")
    _rename_header(GEMMINI_SW / "include" / hv, v_map, "FP64V")

    preamble = _attn_fp64_preamble(fmt, hqk, hv, S, D)
    prep = _attn_fp64_prep(fmt)
    body = _attn_fp64_body(fmt, S, D)
    decl = ("  static out_t O_hw[ATT_S][OUT_COLS];\n"
            "  uint32_t scale_factors[512] __attribute__((aligned(32))) = {0};\n")
    mlock = "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"

    cap = (preamble + "\nint main(){\n" + mlock + decl + "  memset(O_hw,0,sizeof(O_hw));\n" + prep + body +
           '  printf("GOLD_BEGIN\\n");\n  for(int i=0;i<ATT_S;i++)for(int j=0;j<OUT_COLS;j++)\n'
           '    printf("%016llx\\n",(unsigned long long)O_hw[i][j]);\n  printf("GOLD_END\\n");\n'
           "#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 360)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"{tag} attention gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    cols = D // 4
    if len(vals) != S * cols:
        raise RuntimeError(f"{tag} attn gold: expected {S*cols} words, got {len(vals)}")
    gold_init = ",\n".join("  { " + ", ".join("0x" + vals[i*cols + j] + "ULL" for j in range(cols)) + " }"
                           for i in range(S))

    opt = (preamble + "\n#define OUTPUT_MATRIX_NAME O_hw\n"
           "static const out_t gold[ATT_S][OUT_COLS] = {\n" + gold_init + "\n};\n"
           "int full_is_equal(out_t x[ATT_S][OUT_COLS], const out_t y[ATT_S][OUT_COLS]){\n"
           "  for(int i=0;i<ATT_S;i++)for(int j=0;j<OUT_COLS;j++) if(x[i][j]!=y[i][j]) return 0;\n  return 1;\n}\n"
           "#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\nint main(){\n" + mlock + decl + prep +
           "  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){\n"
           "    memset(O_hw,0,sizeof(O_hw));\n    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n"
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')

    hdir = HARNESSES_DIR / prob_type; sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED baseline MX attention kernel (S={S}, D={D}, {fmt}).\nvoid solution(void) {{\n" + body + "}\n")
    return prob_type, prob_id


# ============================================================================
# Multi-head / GQA / MQA attention (fp8). H query heads, H_kv key/value heads
# (H_kv==H -> MHA; H_kv<H -> GQA; H_kv==1 -> MQA). Each head has DISTINCT data
# (golden_model --seed varied), so the GQA KV-reuse indexing is actually exercised.
# Reuses the single-head QK->softmax->PV pipeline (_KERNEL_BODY) inside an H-loop
# with per-head pointer arrays. Correctness oracle = hardware-captured gold (all heads).
# ============================================================================

# Integer-softmax helper block (MX_* consts + mx_ibf16_to_q/ilog2/iexp/ienc_fp8),
# sliced verbatim from the single-head harness so the two stay in sync.
_INT_HELPERS = _HARNESS[_HARNESS.find("// --- float-FREE integer softmax"):_HARNESS.find("int full_is_equal")]


def _mha_kernel_body(out_name="O_hw"):
    """The single-head QK->softmax->PV body, retargeted to per-head pointers + an H-loop."""
    b = _KERNEL_BODY
    b = b.replace("(uint64_t)&Q_scales_row, sizeof(Q_scales_row)", "(uint64_t)Q_sc, QSS_BYTES")
    b = b.replace("(uint64_t)&KT_scales_col, sizeof(KT_scales_col)", "(uint64_t)KT_sc, QSS_BYTES")
    b = b.replace("(uint64_t)&V_scales_col, sizeof(V_scales_col)", "(uint64_t)V_sc, VSS_BYTES")
    b = b.replace("&{OUT}[0][0]", "&%s[h][0][0]" % out_name)
    return ("  for (int h = 0; h < ATT_H; h++) {\n"
            "    int kv = h / (ATT_H / ATT_HKV);\n"
            "    const elem_t *Q_in = Qs[h];  const uint8_t *Q_sc = Qss[h];\n"
            "    const elem_t *KT_in = KTs[kv]; const uint8_t *KT_sc = KTss[kv];\n"
            "    const elem_t *V_in = Vs[kv];  const uint8_t *V_sc = Vss[kv];\n"
            + b + "  }\n")


def gen_mha_attention_problem(S, D, H, H_kv=None, causal=False,
                              prob_type="gemmini-mx-mha", prob_id=0, fmt="fp8:e4m3"):
    """Multi-head attention. H_kv=None -> MHA (H_kv=H). Requires H % H_kv == 0."""
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        raise NotImplementedError("MHA generator is fp8-only for now")
    H_kv = H_kv or H
    if H % H_kv:
        raise ValueError(f"H ({H}) must be a multiple of H_kv ({H_kv})")
    for (m, k, n) in [(S, D, S), (S, S, D)]:
        errs = check_shape(m, k, n, fmt)
        if errs:
            raise ValueError(f"MHA matmul {m}x{k}x{n} unsupported: {'; '.join(errs)}")
    inc = GEMMINI_SW / "include"

    # Per-head DISTINCT data via golden --seed. Q from H QK-headers (expose A_in/A_scales);
    # KT/V from H_kv headers (expose B_in/B_scales). Each header's other symbols are uniquified.
    q_ptrs, qss_ptrs, kt_ptrs, ktss_ptrs, v_ptrs, vss_ptrs, includes = [], [], [], [], [], [], []
    for h in range(H):
        hdr = f"mxgen_mha_q_{prob_type}_{prob_id}_h{h}.h"
        gen_input_header(S, D, S, fmt, inc / hdr, seed=h)
        _rename_all(inc / hdr, f"qh{h}", {"A_in": f"Q_in_h{h}", "A_scales_row": f"Q_scales_row_h{h}"})
        includes.append(hdr); q_ptrs.append(f"(const elem_t*)Q_in_h{h}"); qss_ptrs.append(f"(const uint8_t*)Q_scales_row_h{h}")
    for kv in range(H_kv):
        hkt = f"mxgen_mha_kt_{prob_type}_{prob_id}_kv{kv}.h"
        gen_input_header(S, D, S, fmt, inc / hkt, seed=500 + kv)
        _rename_all(inc / hkt, f"ktv{kv}", {"B_in": f"KT_in_kv{kv}", "B_scales_col": f"KT_scales_col_kv{kv}"})
        includes.append(hkt); kt_ptrs.append(f"(const elem_t*)KT_in_kv{kv}"); ktss_ptrs.append(f"(const uint8_t*)KT_scales_col_kv{kv}")
        hv = f"mxgen_mha_v_{prob_type}_{prob_id}_kv{kv}.h"
        gen_input_header(S, S, D, fmt, inc / hv, seed=900 + kv)
        _rename_all(inc / hv, f"vv{kv}", {"B_in": f"V_in_kv{kv}", "B_scales_col": f"V_scales_col_kv{kv}"})
        includes.append(hv); v_ptrs.append(f"(const elem_t*)V_in_kv{kv}"); vss_ptrs.append(f"(const uint8_t*)V_scales_col_kv{kv}")

    inc_str = "\n".join(f'#include "include/{h}"' for h in includes)
    arr = lambda lst: ", ".join(lst)
    head = (
        "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
        "#ifndef BAREMETAL\n#include <sys/mman.h>\n#include <stdlib.h>\n#endif\n"
        '#include "include/gemmini_testutils.h"\n' + inc_str + "\n"
        f"#define DIM 16\n#define BF16_PER_WORD 4\n#define ATT_S {S}\n#define ATT_D {D}\n"
        f"#define ATT_H {H}\n#define ATT_HKV {H_kv}\n#define ATT_CAUSAL {int(causal)}\n"
        "#define S_OUT_COLS (ATT_S / BF16_PER_WORD)\n#define D_OUT_COLS (ATT_D / BF16_PER_WORD)\n"
        "#define GROUPS_S (ATT_S / 32)\n#define QSS_BYTES ((ATT_D/32)*ATT_S)\n#define VSS_BYTES ((ATT_S/32)*ATT_D)\n"
        "typedef uint8_t elem_t; typedef uint64_t out_t;\n#ifndef fence\n#define fence() gemmini_fence()\n#endif\n"
        "#define OUTPUT_MATRIX_NAME O_hw\n"
        "static out_t S1_hw[ATT_S][S_OUT_COLS];\nstatic elem_t P_q[ATT_S][ATT_S];\nstatic uint8_t P_scales[GROUPS_S][ATT_S];\n"
        f"static const elem_t* const Qs[ATT_H]   = {{ {arr(q_ptrs)} }};\n"
        f"static const uint8_t* const Qss[ATT_H] = {{ {arr(qss_ptrs)} }};\n"
        f"static const elem_t* const KTs[ATT_HKV]   = {{ {arr(kt_ptrs)} }};\n"
        f"static const uint8_t* const KTss[ATT_HKV] = {{ {arr(ktss_ptrs)} }};\n"
        f"static const elem_t* const Vs[ATT_HKV]   = {{ {arr(v_ptrs)} }};\n"
        f"static const uint8_t* const Vss[ATT_HKV] = {{ {arr(vss_ptrs)} }};\n"
    )
    helpers = _INT_HELPERS.replace("{{", "{").replace("}}", "}")  # un-escape (we won't .format this)
    feq = ("int full_is_equal(out_t x[ATT_H][ATT_S][D_OUT_COLS], const out_t y[ATT_H][ATT_S][D_OUT_COLS]){\n"
           "  for(int h=0;h<ATT_H;h++)for(int i=0;i<ATT_S;i++)for(int j=0;j<D_OUT_COLS;j++) if(x[h][i][j]!=y[h][i][j]) return 0;\n  return 1;\n}\n")
    decls = ("  static out_t O_hw[ATT_H][ATT_S][D_OUT_COLS];\n  uint32_t scale_factors[512] = {0};\n  int SPAD_DEST = 128;\n")
    mlock = "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
    kbody = _mha_kernel_body("O_hw")

    # capture program (dump gold for all heads)
    dump = ('  printf("GOLD_BEGIN\\n");\n  for(int h=0;h<ATT_H;h++)for(int i=0;i<ATT_S;i++)for(int j=0;j<D_OUT_COLS;j++)\n'
            '    printf("%016llx\\n",(unsigned long long)O_hw[h][i][j]);\n  printf("GOLD_END\\n");\n')
    cap = (head + helpers + feq + "int main(){\n" + mlock + decls + "  memset(O_hw,0,sizeof(O_hw));\n  gemmini_flush(0);\n"
           + kbody + dump + "#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 600)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"MHA gold capture failed: {str(out)[:400]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    nexp = H * S * (D // 4)
    if len(vals) != nexp:
        raise RuntimeError(f"MHA gold: expected {nexp} words, got {len(vals)}")
    cols = D // 4
    gold_init = ",\n".join(
        "  { " + ",\n    ".join(
            "{ " + ", ".join("0x" + vals[(h * S + i) * cols + j] + "ULL" for j in range(cols)) + " }"
            for i in range(S)) + " }"
        for h in range(H))

    opt = (head + helpers + f"static const out_t gold[ATT_H][ATT_S][D_OUT_COLS] = {{\n{gold_init}\n}};\n" + feq +
           "#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\nint main(){\n" + mlock + decls +
           "  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){\n"
           "    memset(O_hw,0,sizeof(O_hw));\n    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n"
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')
    hdir = HARNESSES_DIR / prob_type; sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED baseline MHA kernel (S={S}, D={D}, H={H}, H_kv={H_kv}, causal={int(causal)}, {fmt}).\n"
        "void solution(void) {\n" + kbody + "}\n")
    return prob_type, prob_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--fmt", default="fp8:e4m3")
    ap.add_argument("--H", type=int, default=0, help="multi-head: number of query heads (0=single-head)")
    ap.add_argument("--H-kv", type=int, default=0, help="KV heads for GQA/MQA (0=MHA)")
    ap.add_argument("--causal", action="store_true")
    a = ap.parse_args()
    if a.H:
        pt, pid = gen_mha_attention_problem(a.S, a.D, a.H, a.H_kv or None, a.causal, prob_id=a.id, fmt=a.fmt)
    else:
        pt, pid = gen_attention_problem(a.S, a.D, prob_id=a.id, fmt=a.fmt)
    print(f"generated {pt} id={pid}")
