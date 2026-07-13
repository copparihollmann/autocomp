"""True-correctness gate for MX matmul kernels (independent of baseline-as-gold).

Runs a kernel, then computes a CPU fp32 reference from the DEQUANTIZED fp8 inputs
(fp8 e4m3 decode · 2^(fpE8M0 scale exp)) and reports how many output elements fall
within 5% of the true matmul. This distinguishes a real matmul from self-consistent
garbage that "passes" baseline-as-gold.

Finding (2026-06-04): N==K shapes match the reference ~57% (MX-vs-fp32 quant gap at
5% tol); N!=K shapes match ~2% → the B-tile addressing is only correct when N==K.
Caveat: baremetal printf lacks %f, so we report integer within-5% COUNTS, not maxrel.

  python -m autocomp.backend.gemmini.mx_pipeline.verify_truth --M 64 --K 64 --N 64
"""

import argparse
import re

from autocomp.backend.gemmini.mx_pipeline import matmul_gen as G
from autocomp.backend.gemmini import gemmini_eval as E

_REF_HELPERS = (
    "\nstatic inline uint32_t _fb(float f){union{uint32_t u;float f;}v;v.f=f;return v.u;}"
    "\nstatic inline float _bf(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}"
    "\nstatic inline float _pow2(int e){ if(e<-126)return 0.f; if(e>127)e=127; return _bf((uint32_t)(e+127)<<23);}"
    "\nstatic inline float fp8d(uint8_t b){ int s=(b>>7)&1,e=(b>>3)&0xF,m=b&7; float sg=s?-1.f:1.f;"
    " if(e==0) return sg*(m/8.f)*_pow2(-6); return sg*(1.f+m/8.f)*_pow2(e-7);}"
    "\nstatic inline float bf16f2(uint16_t b){return _bf(((uint32_t)b)<<16);}\n"
)
_CHECK = (
    "  int OKc=0,TOT=0;\n"
    "  for(int m=0;m<REAL_M;m++)for(int n=0;n<REAL_N;n++){\n"
    "    float acc=0.f;\n"
    "    for(int k=0;k<REAL_K;k++){\n"
    "      float a=fp8d(((const uint8_t*)A_in)[m*MATMUL_K+k])*_pow2((int)A_scales_row[k/32][m]-127);\n"
    "      float b=fp8d(((const uint8_t*)B_in)[k*MATMUL_N+n])*_pow2((int)B_scales_col[k/32][n]-127);\n"
    "      acc+=a*b;\n    }\n"
    "    uint16_t hb=(uint16_t)((C_hw[m][n/4]>>((n%4)*16))&0xFFFF); float hw=bf16f2(hb);\n"
    "    float d=hw-acc, dn=acc>0?acc:-acc; if(dn<1e-6f)dn=1e-6f;\n"
    "    if((d<0?-d:d)/dn<0.05f)OKc++; TOT++;\n  }\n"
    '  printf("TRUEREF within5pct=%d/%d\\n",OKc,TOT);'
)


def verify(M, K, N, kernel_body=None, fmt="fp8:e4m3"):
    """Returns (within5, total) for the FITTING kernel (default = the standard baseline). Padding-aware:
    header/golden at padded dims, fp32 ref over the REAL M*N*K sub-block (the true oracle, independent
    of baseline-as-gold)."""
    Mp, Kp, Np = G.padded_dims(M, K, N, fmt)
    header = f"mxgen_{G._FMT_TAG.get(fmt,'fp8')}_{Mp}x{Kp}x{Np}.h"
    G.gen_input_header(Mp, Kp, Np, fmt, G.GEMMINI_SW / "include" / header)
    body = kernel_body or G._KERNEL_BODY.replace("{OUT}", "C_hw")
    src = G._CAPTURE.format(header=header, body=body, real_m=M, real_n=N, real_k=K).replace(
        '#include "include/' + header + '"', '#include "include/' + header + '"' + _REF_HELPERS)
    src = re.sub(r'  printf\("GOLD_BEGIN.*?printf\("GOLD_END\\n"\);', lambda m: _CHECK, src, flags=re.S)
    rd = {}
    E.run_spike(src, rd, G.GEMMINI_PATH, "0", 900)
    out = rd.get("retval", "")
    m = re.search(r"within5pct=(\d+)/(\d+)", out if isinstance(out, str) else "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


_CHECK_TILED = _CHECK.replace('printf("TRUEREF', 'printf("TRUEREF')  # same body (REAL_M/N/K loop)


def verify_tiled(M, K, N, fmt="fp8:e4m3"):
    """True fp32 oracle for the OUTER-TILED kernel (padding-aware: real M*N*K sub-block). Builds the
    tiled capture harness + kernel, replaces the gold dump with the fp32-within-5% check."""
    Mp, Np = G.ceil_to(M, G.BLK), G.ceil_to(N, G.BLK)
    Kp = K
    header = f"mxgen_{G._FMT_TAG.get(fmt,'fp8')}_{Mp}x{Kp}x{Np}.h"
    G.gen_input_header(Mp, Kp, Np, fmt, G.GEMMINI_SW / "include" / header)
    pre = G._TILED_PREAMBLE.format(header=header, real_m=M, real_n=N) + _REF_HELPERS + "#define REAL_K MATMUL_K\n"
    src = (pre + "\nint main(){\n"
           "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
           "  memset(C_hw,0,MATMUL_M*OUT_COLS*sizeof(out_t));\n  memset(scale_factors,0,SCALE_FACTORS_BYTES);\n"
           "  gemmini_flush(0); fence();\n  {\n" + G._TILED_KERNEL + "  }\n  fence();\n"
           + _CHECK_TILED + "\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(src, rd, G.GEMMINI_PATH, "0", 1800)
    out = rd.get("retval", "")
    m = re.search(r"within5pct=(\d+)/(\d+)", out if isinstance(out, str) else "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tiled", action="store_true", help="verify the outer-tiled kernel")
    a = ap.parse_args()
    ok, tot = (verify_tiled if a.tiled else verify)(a.M, a.K, a.N)
    frac = 100 * ok / tot if tot else 0
    verdict = "REAL matmul" if frac > 25 else "WRONG (likely N!=K B-addressing)"
    print(f"{a.M}x{a.K}x{a.N}{' tiled' if a.tiled else ''}: within5%={ok}/{tot} ({frac:.0f}%) -> {verdict}")
