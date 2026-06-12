"""Generate an autocomp MX-Gemmini conv2d problem (patch-embed convolution).

Targets stride == kernel ("patch embed") conv2d, the form that appears in real
vision/transformer models (e.g. smolvla's 16x16/stride-16 conv): non-overlapping
im2col, so the kernel = im2col (CPU) + MX matmul (Gemmini).

Reuses the matmul machinery: golden_model.py generates the equivalent matmul
inputs (M=patches, K=C*k*k, N=OC); the conv input X[C][H][W] is reconstructed
from A_in (im2col is bijective when stride==kernel). Correctness oracle =
hardware-captured gold (baseline run once on spike, output baked into harness).

Usage:
    python -m autocomp.backend.gemmini.mx_pipeline.conv_gen --C 2 --H 64 --k 8 --OC 32 --id 0
"""

import argparse
import pathlib
import re

from autocomp.common import HARNESSES_DIR, SOLS_DIR
from autocomp.backend.gemmini import gemmini_eval as E
from autocomp.backend.gemmini.mx_pipeline.matmul_gen import (
    GEMMINI_PATH, GEMMINI_SW, DIM, GROUP, check_shape, gen_input_header, _FMT_TAG, _FMT_HW,
)


def conv_dims(C, H, W, k, OC):
    """Map a patch-embed conv (stride == kernel) to matmul dims."""
    assert H % k == 0 and W % k == 0, "H, W must be multiples of the patch size"
    M = (H // k) * (W // k)
    K = C * k * k
    N = OC
    return M, K, N


def _parse_2d(header: str, name: str):
    body = header.split(f"{name}[")[1].split("= {", 1)[1].split("};", 1)[0]
    return [[int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]+)", row)]
            for row in body.split("{")[1:]]


_HARNESS = r'''// AUTO-GENERATED MX-Gemmini patch-embed conv harness (C={C} H={H} W={W} k={k} OC={OC}).
// Equivalent matmul: M={M} K={K} N={N}. gold = baseline hardware output.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/{header}"

#define DIM 16
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
#define CONV_C {C}
#define CONV_H {H}
#define CONV_W {W}
#define CONV_KSZ {k}
#define PATCHES_X (CONV_W / CONV_KSZ)

typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME C_hw

// Conv input feature map (fp8 codes), reconstructed from the im2col matrix.
static const elem_t X_in[CONV_C][CONV_H][CONV_W] = {{
{x_init}
}};

static const out_t gold[MATMUL_M][OUT_COLS] = {{
{gold_init}
}};

static elem_t A_buf[MATMUL_M][MATMUL_K];

int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {{
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < OUT_COLS; j++)
      if (x[i][j] != y[i][j]) return 0;
  return 1;
}}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {{0}};
  int tiles_I = MATMUL_M / DIM, tiles_J = MATMUL_N / DIM, tiles_K = MATMUL_K / DIM;
  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  int SPAD_DEST = 128;
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {{
    memset(C_hw, 0, sizeof(C_hw));
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

# Conv kernel body: im2col (CPU, byte moves) + MX matmul, parameterized by output.
_KERNEL_BODY = r'''  // im2col: patch (pi,pj) -> row pi*PATCHES_X+pj; column c*k*k + dy*k + dx.
  // The innermost dx span is contiguous in BOTH A_buf and X_in (stride==kernel), so it is
  // a memcpy of CONV_KSZ bytes -- not CONV_KSZ scalar stores (the conv bottleneck, ~1.36x).
  for (int pi = 0; pi < CONV_H / CONV_KSZ; pi++)
    for (int pj = 0; pj < PATCHES_X; pj++)
      for (int c = 0; c < CONV_C; c++)
        for (int dy = 0; dy < CONV_KSZ; dy++)
          memcpy(&A_buf[pi * PATCHES_X + pj][c * CONV_KSZ * CONV_KSZ + dy * CONV_KSZ],
                 &X_in[c][pi * CONV_KSZ + dy][pj * CONV_KSZ], CONV_KSZ * sizeof(elem_t));

  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // batched ("block") mvin at BATCH=2 (cols=2*DIM=32, within the 6-bit hw cols field; RTL-legal).
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k += 2) {{ int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
      gemmini_extended_mvin((void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }}
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));   // B[k][j] stride N -> slot (k*tiles_J + j)
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j += 2) {{ int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }}

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                       SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_mx_read_smem(&{OUT}[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
'''

_CAPTURE = r'''// capture program: run conv baseline once, dump output
{harness_no_gold}
'''

_BASELINE = '''// AUTO-GENERATED baseline MX patch-embed conv kernel (C={C} H={H} k={k} OC={OC}).
void solution(void) {{
{body}}}
'''


def _fmt_rows(rows, per_row_fmt):
    return ",\n".join("  { " + ", ".join(per_row_fmt(v) for v in row) + " }" for row in rows)


def gen_conv_problem(C, H, k, OC, prob_type="gemmini-mx-conv", prob_id=0, fmt="fp8:e4m3"):
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        return _gen_conv_fp64(C, H, k, OC, prob_type, prob_id, fmt)  # fp6/fp4 path
    W = H
    M, K, N = conv_dims(C, H, W, k, OC)
    errs = check_shape(M, K, N)
    if errs:
        raise ValueError(f"conv→matmul {M}x{K}x{N} unsupported: {'; '.join(errs)}")
    header = f"mxgen_conv_{C}x{H}x{k}_{OC}.h"
    gen_input_header(M, K, N, fmt, GEMMINI_SW / "include" / header)
    hdr_text = (GEMMINI_SW / "include" / header).read_text()

    # Reconstruct X[C][H][W] from im2col rows (bijective for stride == kernel).
    A = _parse_2d(hdr_text, "A_in")
    X = [[[0] * W for _ in range(H)] for _ in range(C)]
    px = W // k
    for p in range(M):
        pi, pj = divmod(p, px)
        for c in range(C):
            for dy in range(k):
                for dx in range(k):
                    X[c][pi * k + dy][pj * k + dx] = A[p][c * k * k + dy * k + dx]
    x_init = ",\n".join(
        "  {\n" + _fmt_rows(X[c], lambda v: f"0x{v:02x}") + "\n  }" for c in range(C)
    )

    body_gold = _KERNEL_BODY.format(OUT="C_hw")
    capture_src = _HARNESS.format(
        C=C, H=H, W=W, k=k, OC=OC, M=M, K=K, N=N, header=header,
        x_init=x_init, gold_init="  {0}",
    ).replace("// SUBSTITUTE HERE", body_gold +
              '\n  printf("GOLD_BEGIN\\n");\n'
              "  for (int i = 0; i < MATMUL_M; i++)\n"
              "    for (int j = 0; j < OUT_COLS; j++)\n"
              '      printf("%016llx\\n", (unsigned long long) C_hw[i][j]);\n'
              '  printf("GOLD_END\\n");')

    rd = {}
    E.run_spike(capture_src, rd, GEMMINI_PATH, "0", 240)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"conv gold capture failed: {str(out)[:200]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    cols = N // 4
    if len(vals) != M * cols:
        raise RuntimeError(f"conv gold capture: expected {M*cols} words, got {len(vals)}")
    gold_init = ",\n".join(
        "  { " + ", ".join("0x" + vals[i * cols + j] + "ULL" for j in range(cols)) + " }"
        for i in range(M)
    )

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(_HARNESS.format(
        C=C, H=H, W=W, k=k, OC=OC, M=M, K=K, N=N, header=header,
        x_init=x_init, gold_init=gold_init))
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(_BASELINE.format(
        C=C, H=H, k=k, OC=OC, body=_KERNEL_BODY.format(OUT="C_hw")))
    return prob_type, prob_id, (M, K, N)


# ----------------------------------------------------------------------------
# fp6 / fp4 patch-embed conv (sub-byte microscaling). im2col (host) builds the
# logical code matrix A_buf[M][K], M-packed to A_in_hw, then the fp6/fp4 matmul
# (B weights packed + scales + fp6 LUTs). Validated 2026-06-04: conv fp4 100% vs
# golden C_out_bf16.
# ----------------------------------------------------------------------------
def _gen_conv_fp64(C, H, k, OC, prob_type, prob_id, fmt):
    W = H
    M, K, N = conv_dims(C, H, W, k, OC)
    if M % 32 or N % 32:
        raise ValueError(f"fp6/fp4 conv needs M(patches),N(OC) multiples of 32; got M={M} N={N}")
    if K % GROUP:
        raise ValueError(f"K=C*k*k must be a multiple of {GROUP}; got K={K}")
    tag = _FMT_TAG[fmt]; cfg = _FMT_HW[fmt]; lut = cfg["lut"]
    header = f"mxgen_conv_{tag}_{C}x{H}x{k}_{OC}.h"
    gen_input_header(M, K, N, fmt, GEMMINI_SW / "include" / header)
    hdr = (GEMMINI_SW / "include" / header).read_text()
    A = _parse_2d(hdr, "A_in")  # [M][K/2] packed nibbles

    def code_at(p, kk):
        b = A[p][kk >> 1]
        return (b >> 4) & 0xF if (kk & 1) else b & 0xF
    px = W // k
    X = [[[0] * W for _ in range(H)] for _ in range(C)]
    for p in range(M):
        pi, pj = divmod(p, px)
        for c in range(C):
            for dy in range(k):
                for dx in range(k):
                    X[c][pi * k + dy][pj * k + dx] = code_at(p, c * k * k + dy * k + dx)
    x_init = ",\n".join("  {\n" + _fmt_rows(X[c], lambda v: f"0x{v:02x}") + "\n  }" for c in range(C))

    gl = max(M, N).bit_length() if lut else 1
    lutdecl = ("static uint8_t A_lp[12], B_lp[12];\n"
               "static void mx_pack_lut(const uint8_t *cc, uint8_t *o){uint64_t lo=0;uint32_t hi=0;"
               "for(int i=0;i<16;i++){int bit=i*6;uint64_t v=cc[i]&0x3F;if(bit+6<=64)lo|=v<<bit;"
               "else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}"
               "for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}\n") if lut else ""
    lutpack = "  mx_pack_lut((const uint8_t*)A_lut,A_lp); mx_pack_lut((const uint8_t*)B_lut,B_lp);\n" if lut else ""
    lutload = "    gemmini_mx_load_lut((uint64_t)B_lp,1,0); gemmini_mx_load_lut((uint64_t)A_lp,1,1);\n" if lut else ""

    preamble = (
        '#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n'
        '#ifndef BAREMETAL\n#include <sys/mman.h>\n#include <stdlib.h>\n#endif\n'
        '#include "include/gemmini_testutils.h"\n#include "include/%H%"\n'
        '#define DIM 16\n#define CC %C%\n#define CH %HH%\n#define CK %K%\n#define PX (CH/CK)\n'
        '#define BF16_PER_WORD 4\n#define OUT_COLS (MATMUL_N/BF16_PER_WORD)\n'
        'typedef uint8_t elem_t; typedef uint64_t out_t;\n#ifndef fence\n#define fence() gemmini_fence()\n#endif\n'
        'static const elem_t X_in[CC][CH][CH] = {\n%XINIT%\n};\n'
        'static elem_t A_buf[MATMUL_M][MATMUL_K];\nstatic elem_t A_in_hw[MATMUL_M/2][MATMUL_K];\n' + lutdecl
    ).replace("%H%", header).replace("%C%", str(C)).replace("%HH%", str(H)).replace("%K%", str(k)).replace("%XINIT%", x_init)

    prep = lutpack
    body = (
        "  for(int pi=0;pi<CH/CK;pi++)for(int pj=0;pj<PX;pj++)for(int c=0;c<CC;c++)for(int dy=0;dy<CK;dy++)for(int dx=0;dx<CK;dx++)\n"
        "    A_buf[pi*PX+pj][c*CK*CK+dy*CK+dx]=X_in[c][pi*CK+dy][pj*CK+dx];\n"
        "  for(int m=0;m<MATMUL_M;m++)for(int kk=0;kk<MATMUL_K;kk++){uint8_t cd=A_buf[m][kk]&0xF;\n"
        "    if((m&1)==0)A_in_hw[m>>1][kk]=(A_in_hw[m>>1][kk]&0xF0)|cd; else A_in_hw[m>>1][kk]=(A_in_hw[m>>1][kk]&0x0F)|(cd<<4);}\n"
        "  { int ti=MATMUL_M/DIM/2, tj=MATMUL_N/DIM/2, tk=MATMUL_K/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;\n"
        "    gemmini_flush(0);\n"
        f"    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,{cfg['actf']},{cfg['wgtf']},3,{cfg['uselut']});\n"
        "    gemmini_mx_load_scales((uint64_t)&A_scales_row,sizeof(A_scales_row),0);\n"
        "    gemmini_mx_load_scales((uint64_t)&B_scales_col,sizeof(B_scales_col),1);\n"
        "    gemmini_config_st(OUT_COLS*sizeof(out_t));\n"
        f"    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,{gl});\n"
        f"{lutload}    gemmini_config_ld(MATMUL_K*sizeof(elem_t));\n"
        "    for(int i=0;i<ti;i++)for(int kk=0;kk<tk;kk++) gemmini_extended_mvin((void*)(((elem_t*)A_in_hw)+i*DIM*MATMUL_K+kk*DIM),ab+(i*tk+kk)*DIM,DIM,DIM);\n"
        "    gemmini_config_ld(MATMUL_N*sizeof(elem_t)/2);\n"
        "    for(int kk=0;kk<tk;kk++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)B_in)+kk*DIM*(MATMUL_N/2)+j*DIM),bb+(kk*tj+j)*DIM,DIM,DIM);\n"
        "    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n"
        "    gemmini_mx_read_smem(&C_hw[0][0],128*16,MATMUL_M*MATMUL_N); gemmini_fence(); }\n")

    mlock = "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
    decl = "  static out_t C_hw[MATMUL_M][OUT_COLS];\n  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};\n"
    cap = (preamble + "\nint main(){\n" + mlock + decl + "  memset(C_hw,0,sizeof(C_hw));\n" + prep + body +
           '  printf("GOLD_BEGIN\\n");\n  for(int i=0;i<MATMUL_M;i++)for(int j=0;j<OUT_COLS;j++)\n'
           '    printf("%016llx\\n",(unsigned long long)C_hw[i][j]);\n  printf("GOLD_END\\n");\n'
           "#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 300)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"{tag} conv gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    cols = N // 4
    if len(vals) != M * cols:
        raise RuntimeError(f"{tag} conv gold: expected {M*cols} words, got {len(vals)}")
    gold_init = ",\n".join("  { " + ", ".join("0x" + vals[i*cols + j] + "ULL" for j in range(cols)) + " }"
                           for i in range(M))

    opt = (preamble + "\n#define OUTPUT_MATRIX_NAME C_hw\n"
           "static const out_t gold[MATMUL_M][OUT_COLS] = {\n" + gold_init + "\n};\n"
           "int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]){\n"
           "  for(int i=0;i<MATMUL_M;i++)for(int j=0;j<OUT_COLS;j++) if(x[i][j]!=y[i][j]) return 0;\n  return 1;\n}\n"
           "#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\nint main(){\n" + mlock + decl + prep +
           "  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){\n"
           "    memset(C_hw,0,sizeof(C_hw));\n    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n"
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')

    hdir = HARNESSES_DIR / prob_type; sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED baseline MX patch-embed conv kernel (C={C} H={H} k={k} OC={OC}, {fmt}).\n"
        "void solution(void) {\n" + body + "}\n")
    return prob_type, prob_id, (M, K, N)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--OC", type=int, default=32)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--fmt", default="fp8:e4m3")
    a = ap.parse_args()
    pt, pid, dims = gen_conv_problem(a.C, a.H, a.k, a.OC, prob_id=a.id, fmt=a.fmt)
    print(f"generated {pt} id={pid} matmul dims={dims}")
