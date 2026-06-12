"""Generate an autocomp MX-Gemmini matmul problem (harness + golden + baseline)
for an arbitrary (M, K, N) shape, driven by a model2MLIR OpSpec.

Correctness oracle = "hardware-captured gold": we run the baseline MX kernel once
on spike, capture its bf16-packed output, and bake it into the harness as the
`gold` reference. Optimized candidates must reproduce it bit-for-bit. This is
hardware-faithful by construction for ANY shape, needs no golden_model.py<->hw
numeric match, and keeps each program to a single kernel run (running the MX
kernel twice in one program leaks accelerator state).

golden_model.py is still used to generate the INPUT header (A_in/B_in + fpe8m0
scales in the right C format); its C_out_bf16 is ignored.

Baseline pattern = "load all A+B tiles, one fused WS loop" -> shape must fit the
scratchpad: M,N % 16 == 0, K % 32 == 0, (tiles_I+tiles_J)*tiles_K*16 <= 8192.

Usage:
    python -m autocomp.backend.gemmini.mx_pipeline.matmul_gen --M 64 --K 64 --N 64 --id 0
"""

import argparse
import pathlib
import re
import subprocess

from autocomp.common import HARNESSES_DIR, SOLS_DIR
from autocomp.backend.gemmini import gemmini_eval as E
from autocomp.backend.gemmini.gemmini_eval import INT8_16PE_CHIPYARD_PATH

CHIPYARD = pathlib.Path(INT8_16PE_CHIPYARD_PATH)
GEMMINI_PATH = CHIPYARD / "generators" / "gemmini"
GEMMINI_SW = GEMMINI_PATH / "software" / "gemmini-rocc-tests"
GOLDEN_PY = GEMMINI_SW / "golden_model.py"
GOLDEN_PYTHON = "/scratch/agustin/projects/model2MLIR/.venv/bin/python"

DIM = 16
GROUP = 32
SPAD_ROWS = 4 * 2048
_FMT_TAG = {"fp8:e4m3": "fp8", "fp6:e3m2": "fp6", "fp4:e2m1": "fp4"}


def ceil_to(x, g):
    return (x + g - 1) // g * g


def padded_dims(M, K, N, fmt="fp8:e4m3"):
    """Round (M,N) up to the systolic-tile granularity (DIM for fp8, 32 for fp6/fp4) and
    K up to DIM. Real models have M%16!=0 in 61.6% of matmuls; the header/golden are emitted
    at the padded shape, while the KERNEL is told the real dims and uses loop_ws pad_I/pad_J/
    pad_K so only the real M*N*K is computed/moved. Output row i depends only on A row i, so
    the real M*N sub-block of golden's C_out is correct regardless of the padding rows/cols."""
    g = DIM if _FMT_TAG.get(fmt, "fp8") == "fp8" else 32
    return ceil_to(M, g), ceil_to(K, DIM), ceil_to(N, g)


def fits_scratchpad(M, K, N, fmt="fp8:e4m3"):
    Mp, Kp, Np = padded_dims(M, K, N, fmt)
    tI, tJ, tK = Mp // DIM, Np // DIM, Kp // DIM
    return (tI * tK + tJ * tK) * DIM <= SPAD_ROWS


def check_shape(M, K, N, fmt="fp8:e4m3"):
    """Arbitrary M,N,K are accepted (padded to tile granularity via loop_ws pad fields).
    The only hard limit is the padded shape fitting the scratchpad (else use outer tiling)."""
    errs = []
    if M <= 0 or N <= 0 or K <= 0:
        errs.append(f"M,K,N must be positive (got M={M}, K={K}, N={N})")
    if not errs and not fits_scratchpad(M, K, N, fmt):
        errs.append("padded shape exceeds scratchpad with the load-all baseline (needs outer tiling)")
    return errs


def gen_input_header(M, K, N, fmt, out_path: pathlib.Path, seed=0):
    """golden_model.py -> header with A_in/B_in/A_scales_row/B_scales_col (inputs).

    fp8 (1 byte/elem): --tile 16. fp6/fp4 (2 elems/byte, 32-wide HW tiles): --tile 32,
    and fp6 stores 4-bit LUT indices (--lut-index-bits 4) + A_lut/B_lut/C_lut tables.
    `seed` varies the random inputs (used by multi-head attention to make heads distinct).
    """
    tag = _FMT_TAG.get(fmt, "fp8")
    tile = "16" if tag == "fp8" else "32"
    cmd = [
        GOLDEN_PYTHON, str(GOLDEN_PY),
        "--input", fmt, "--prod", fmt, "--acc", "bf16", "--acc-rounding", "q_bf16_rne",
        "--scaled-spec", "bf16", "--scale-spec", "fpe8m0", "--prod-mant-bits", "7",
        "--input-rounding", "zero", "--prod-rounding", "zero", "--scale-exp", "2",
        "--tile", tile, "--M", str(M), "--K", str(K), "--N", str(N), "--seed", str(seed),
        "--header-path", str(out_path),
    ]
    if tag == "fp6":
        cmd += ["--lut-index-bits", "4"]
    p = subprocess.run(cmd, cwd=GEMMINI_SW, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"golden_model.py failed for {M}x{K}x{N}:\n{p.stdout[-1500:]}")


# Kernel body parameterized by the output array name.
_KERNEL_BODY = r'''  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // mvin FULL padded tiles (block-aligned DIMxDIM -> 1 cheap DMA/tile; sub-DIM mvin is modeled
  // as a slow/unaligned path on spike). The header holds padded data; loop_ws pad_I/pad_J/pad_K
  // then skip the padded rows/cols in COMPUTE, so the real M*N*K result is unaffected.
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));      // A rows: stride K
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);
  // B tile (k,j) = B_in[k][j] (stride N) into scratchpad slot (k*tiles_J + j) —
  // the layout gemmini's MX loop reads (B_t = B_sp + (k*TJ + j)*DIM). The old
  // (j*tiles_K + k) slot was a transpose: correct only when N==K, garbage otherwise.
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));      // B rows: stride N
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, DIM, DIM);

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, PAD_I, PAD_J, PAD_K, a_base, BANK_NUM * BANK_ROWS, 0,
                       SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  // read only the REAL rows (smem row stride is the padded MATMUL_N); padded rows stay 0 (memset).
  gemmini_mx_read_smem(&{OUT}[0][0], SPAD_DEST * 16, REAL_M * MATMUL_N);
  gemmini_fence();
'''

# Common preamble (types/macros/decls) shared by capture + opt harnesses.
_PREAMBLE = r'''#include <stdint.h>
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
// MATMUL_M/K/N from the header are the PADDED dims (tile-aligned). REAL_* are the true
// problem dims; the kernel computes/moves only the real region and pads the rest via loop_ws.
#define REAL_M {real_m}
#define REAL_N {real_n}
#define REAL_K {real_k}
#define REAL_OUT_COLS (REAL_N / BF16_PER_WORD)
#define PAD_I (MATMUL_M - REAL_M)
#define PAD_J (MATMUL_N - REAL_N)
#define PAD_K (MATMUL_K - REAL_K)
typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
'''

# Standalone program: run the baseline once, dump C_hw between markers for capture.
_CAPTURE = _PREAMBLE + r'''
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
  memset(C_hw, 0, sizeof(C_hw));
  gemmini_flush(0);
{body}
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < OUT_COLS; j++)
      printf("%016llx\n", (unsigned long long) C_hw[i][j]);
  printf("GOLD_END\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''

# Optimization harness: bake captured gold, run candidate once, compare.
_OPT = _PREAMBLE + r'''
#define OUTPUT_MATRIX_NAME C_hw
static const out_t gold[MATMUL_M][OUT_COLS] = {{
{gold_init}
}};

int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {{
  for (int i = 0; i < REAL_M; i++)            // compare only the real M*N sub-block
    for (int j = 0; j < REAL_OUT_COLS; j++)
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

_BASELINE = '''// AUTO-GENERATED baseline MX matmul kernel ({M}x{K}x{N}, {fmt}).
void solution(void) {{
{body}}}
'''


def _capture_gold(M, K, N, Mp, Np, header):
    """Build+run the capture program on spike, parse the dumped gold words, and return only
    the REAL M*(N/4) sub-block (the capture dumps the padded Mp*(Np/4) grid)."""
    src = _CAPTURE.format(header=header, body=_KERNEL_BODY.replace("{OUT}", "C_hw"),
                          real_m=M, real_n=N, real_k=K)
    rd = {}
    E.run_spike(src, rd, GEMMINI_PATH, "0", 180)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"gold capture failed (compile/run): {str(out)[:200]}")
    body = out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0]
    allvals = re.findall(r"[0-9a-fA-F]{16}", body)
    pcols = Np // 4
    if len(allvals) != Mp * pcols:
        raise RuntimeError(f"gold capture: expected {Mp*pcols} words, got {len(allvals)}")
    rcols = N // 4
    return [allvals[i * pcols + j] for i in range(M) for j in range(rcols)]


def _parse_c_out_bf16(header_path: pathlib.Path):
    """Parse golden_model.py's `C_out_bf16[M][N]` (independent fp32-ref gold) from a
    generated header -> list of M rows, each a list of N ints (one bf16 per element)."""
    txt = header_path.read_text()
    m = re.search(r"C_out_bf16\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", txt, re.S)
    if not m:
        raise RuntimeError(f"C_out_bf16 not found in {header_path}")
    body = m.group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    out = []
    for r in rows:
        vals = re.findall(r"0x[0-9a-fA-F]+|\d+", r)
        if vals:
            out.append([int(v, 0) for v in vals])
    return out


def _independent_gold(M, N, header_path: pathlib.Path):
    """Pack golden_model.py's C_out_bf16 into the same 4-bf16/uint64 layout the harness
    compares (word j of row i = cols 4j..4j+3, col 4j in the low 16 bits) -> 16-hex words.
    This is the INDEPENDENT fp32-reference oracle (vs self-captured spike output)."""
    grid = _parse_c_out_bf16(header_path)
    if len(grid) < M or any(len(r) < N for r in grid[:M]):   # grid may be padded (>= real)
        raise RuntimeError(f"C_out_bf16 shape {len(grid)}x{len(grid[0]) if grid else 0} < {M}x{N}")
    vals = []
    for i in range(M):
        for j in range(N // 4):
            w = (grid[i][4*j+3] << 48) | (grid[i][4*j+2] << 32) | (grid[i][4*j+1] << 16) | grid[i][4*j]
            vals.append(f"{w:016x}")
    return vals


def gen_matmul_problem(M, K, N, prob_type="gemmini-mx-matmul", prob_id=0, fmt="fp8:e4m3",
                       gold_src="capture"):
    """gold_src: "capture" = self-captured spike output (legacy); "independent" =
    golden_model.py's C_out_bf16 (true fp32-reference oracle, cross-checked vs the
    Rakanic fork). "validate" = build both and require they match bit-for-bit."""
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        return _gen_matmul_fp64(M, K, N, prob_type, prob_id, fmt)  # fp6/fp4 path
    errs = check_shape(M, K, N, fmt)
    if errs:
        raise ValueError(f"unsupported shape {M}x{K}x{N}: {'; '.join(errs)}")
    tag = _FMT_TAG.get(fmt, "fp8")
    Mp, Kp, Np = padded_dims(M, K, N, fmt)   # header/golden are emitted at the padded shape
    header = f"mxgen_{tag}_{Mp}x{Kp}x{Np}.h"
    header_path = GEMMINI_SW / "include" / header
    gen_input_header(Mp, Kp, Np, fmt, header_path)

    rows, cols = M, N // 4                    # gold covers only the real M*(N/4) sub-block
    if gold_src == "independent":
        vals = _independent_gold(M, N, header_path)
    elif gold_src == "validate":
        cap = _capture_gold(M, K, N, Mp, Np, header)
        ind = _independent_gold(M, N, header_path)
        mism = sum(1 for a, b in zip(cap, ind) if a.lstrip("0") != b.lstrip("0"))
        if mism:
            raise RuntimeError(f"GOLD MISMATCH {M}x{K}x{N} {fmt}: {mism}/{len(cap)} words "
                               f"differ between spike-capture and golden_model.py C_out_bf16")
        vals = ind
    else:
        vals = _capture_gold(M, K, N, Mp, Np, header)
    gold_init = ",\n".join(
        "  { " + ", ".join("0x" + vals[i * cols + j] + "ULL" for j in range(cols)) + " }"
        for i in range(rows)
    )

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(
        _OPT.format(header=header, gold_init=gold_init, real_m=M, real_n=N, real_k=K))
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        _BASELINE.format(M=M, K=K, N=N, fmt=fmt, body=_KERNEL_BODY.replace("{OUT}", "C_hw")))
    return prob_type, prob_id, header


# ----------------------------------------------------------------------------
# fp6 / fp4 path (sub-byte microscaling). Differs from fp8: 32-wide HW tiles
# (2 elems/byte), M-packed A operand (A_in_hw, rebuilt in C from golden's
# K-packed A_in), B packed 2/byte, act/wgt format bits, and for fp6 a loaded
# LUT (4-bit index -> 6-bit e3m2 code). Validated 2026-06-04: fp4 99% exact vs
# golden, fp6 55% within-5% of a true fp6 reference (== fp8's quant level).
# ----------------------------------------------------------------------------
_FMT_HW = {
    "fp6:e3m2": dict(actf=1, wgtf=1, uselut=1, lut=True),
    "fp4:e2m1": dict(actf=2, wgtf=2, uselut=0, lut=False),
}

_FP64_PREAMBLE = r'''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/%HEADER%"

#define DIM 16
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif

// A in HW layout: 2 M-rows packed per byte (golden emits K-packed A_in[M][K/2]).
static elem_t A_in_hw[MATMUL_M / 2][MATMUL_K];
%LUTDECL%'''

_FP64_LUTDECL = r'''static uint8_t A_lut_p[12], B_lut_p[12];   // golden A_lut/B_lut[16] packed to 96-bit
static void mx_pack_lut(const uint8_t *codes, uint8_t *out) {
  uint64_t lo = 0; uint32_t hi = 0;
  for (int i = 0; i < 16; i++) { int bit = i * 6; uint64_t c = codes[i] & 0x3F;
    if (bit + 6 <= 64) lo |= c << bit;
    else if (bit >= 64) hi |= (uint32_t)(c << (bit - 64));
    else { lo |= c << bit; hi |= (uint32_t)(c >> (64 - bit)); } }
  for (int b = 0; b < 8; b++) out[b] = (lo >> (b * 8)) & 0xFF;
  for (int b = 0; b < 4; b++) out[8 + b] = (hi >> (b * 8)) & 0xFF;
}
'''

# CPU data marshaling, run once before the gemmini kernel.
_FP64_PREP = r'''  for (int m = 0; m < MATMUL_M; m++)
    for (int k = 0; k < MATMUL_K; k++) {
      uint8_t byte = A_in[m][k >> 1];
      uint8_t code = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
      if ((m & 1) == 0) A_in_hw[m >> 1][k] = (A_in_hw[m >> 1][k] & 0xF0) | code;
      else              A_in_hw[m >> 1][k] = (A_in_hw[m >> 1][k] & 0x0F) | (code << 4);
    }
%LUTPACK%'''


def _fp64_body(fmt, M, N):
    cfg = _FMT_HW[fmt]
    # LUT granularity must shift the largest A(M) or B(N) index to 0 so every element
    # maps to the single shared LUT group; max(M,N).bit_length() covers both (M>N too).
    glut = max(M, N).bit_length() if cfg["lut"] else 1
    lut_load = ("    gemmini_mx_load_lut((uint64_t)B_lut_p, 1, 0);\n"
                "    gemmini_mx_load_lut((uint64_t)A_lut_p, 1, 1);\n") if cfg["lut"] else ""
    return (
        "  {\n"
        "    int ti = MATMUL_M / DIM / 2, tj = MATMUL_N / DIM / 2, tk = MATMUL_K / DIM;\n"
        "    uint32_t ab = 0;\n"
        "    uint32_t bb = BANK_NUM * BANK_ROWS - tk * tj * DIM;\n"
        "    int SDST = 128;\n"
        f"    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, {cfg['actf']}, {cfg['wgtf']}, 3, {cfg['uselut']});\n"
        "    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);\n"
        "    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);\n"
        "    gemmini_config_st(OUT_COLS * sizeof(out_t));\n"
        f"    gemmini_mxquant_config_mvout((uint64_t)scale_factors, ti, tj, tk, 0, 0, {glut});\n"
        f"{lut_load}"
        "    gemmini_config_ld(MATMUL_K * sizeof(elem_t));\n"
        "    for (int i = 0; i < ti; i++)\n"
        "      for (int k = 0; k < tk; k++)\n"
        "        gemmini_extended_mvin((void*)(((elem_t*)A_in_hw) + i*DIM*MATMUL_K + k*DIM), ab + (i*tk + k)*DIM, DIM, DIM);\n"
        "    gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);\n"
        "    for (int k = 0; k < tk; k++)\n"
        "      for (int j = 0; j < tj; j++)\n"
        "        gemmini_extended_mvin((void*)(((elem_t*)B_in) + k*DIM*(MATMUL_N/2) + j*DIM), bb + (k*tj + j)*DIM, DIM, DIM);\n"
        "    gemmini_loop_ws_spad(ti, tj, tk, 0, 0, 0, ab, BANK_NUM * BANK_ROWS, 0, SDST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);\n"
        "    gemmini_mx_read_smem(&C_hw[0][0], SDST * 16, MATMUL_M * MATMUL_N);\n"
        "    gemmini_fence();\n"
        "  }\n"
    )


def _fp64_pieces(fmt):
    cfg = _FMT_HW[fmt]
    preamble = _FP64_PREAMBLE.replace("%LUTDECL%", _FP64_LUTDECL if cfg["lut"] else "")
    prep = _FP64_PREP.replace("%LUTPACK%",
        "  mx_pack_lut((const uint8_t*)A_lut, A_lut_p);\n  mx_pack_lut((const uint8_t*)B_lut, B_lut_p);\n"
        if cfg["lut"] else "")
    return preamble, prep


def _gen_matmul_fp64(M, K, N, prob_type, prob_id, fmt):
    """fp6/fp4 fitting matmul: generate header, capture gold from baseline, write opt harness."""
    if M % 32 or N % 32:
        raise ValueError(f"fp6/fp4 needs M,N multiples of 32 (32-wide HW tiles); got M={M} N={N}")
    if K % GROUP:
        raise ValueError(f"K must be a multiple of {GROUP}; got K={K}")
    tag = _FMT_TAG[fmt]
    header = f"mxgen_{tag}_{M}x{K}x{N}.h"
    gen_input_header(M, K, N, fmt, GEMMINI_SW / "include" / header)
    preamble, prep = _fp64_pieces(fmt)
    body = _fp64_body(fmt, M, N)
    preamble = preamble.replace("%HEADER%", header)

    # capture: prep + body + dump
    cap = (preamble + "\nint main() {\n"
           "#ifndef BAREMETAL\n  if (mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
           "  static out_t C_hw[MATMUL_M][OUT_COLS];\n  uint32_t scale_factors[512] = {0};\n"
           "  memset(C_hw, 0, sizeof(C_hw));\n  gemmini_flush(0);\n" + prep + body +
           '  printf("GOLD_BEGIN\\n");\n'
           "  for (int i = 0; i < MATMUL_M; i++) for (int j = 0; j < OUT_COLS; j++)\n"
           '    printf("%016llx\\n", (unsigned long long) C_hw[i][j]);\n'
           '  printf("GOLD_END\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 240)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"{tag} gold capture failed: {str(out)[:200]}")
    vals = re.findall(r"[0-9a-fA-F]{16}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    rows, cols = M, N // 4
    if len(vals) != rows * cols:
        raise RuntimeError(f"{tag} gold: expected {rows*cols} words, got {len(vals)}")
    gold_init = ",\n".join("  { " + ", ".join("0x" + vals[i*cols + j] + "ULL" for j in range(cols)) + " }"
                           for i in range(rows))

    # opt harness: prep fixed before the optimizable region (SUBSTITUTE = the gemmini kernel)
    opt = (preamble + "\n#define OUTPUT_MATRIX_NAME C_hw\n"
           "static const out_t gold[MATMUL_M][OUT_COLS] = {\n" + gold_init + "\n};\n\n"
           "int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {\n"
           "  for (int i = 0; i < MATMUL_M; i++) for (int j = 0; j < OUT_COLS; j++)\n"
           "    if (x[i][j] != y[i][j]) return 0;\n  return 1;\n}\n\n"
           "#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\n\nint main() {\n"
           "#ifndef BAREMETAL\n  if (mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
           "  static out_t C_hw[MATMUL_M][OUT_COLS];\n  uint32_t scale_factors[512] = {0};\n" + prep +
           "  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {\n"
           "    memset(C_hw, 0, sizeof(C_hw));\n    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n"
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED baseline MX matmul kernel ({M}x{K}x{N}, {fmt}).\n"
        "void solution(void) {\n" + body + "}\n")
    return prob_type, prob_id, header


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--prob-type", default="gemmini-mx-matmul")
    ap.add_argument("--fmt", default="fp8:e4m3")
    a = ap.parse_args()
    pt, pid, hdr = gen_matmul_problem(a.M, a.K, a.N, a.prob_type, a.id, a.fmt)
    print(f"generated prob_type={pt} id={pid} header={hdr}")


# ============================================================================
# Outer-tiled matmul (P2b): for shapes that exceed the scratchpad.
# Blocks the matmul into BM x BK x BN sub-matmuls (each = the proven load-all
# kernel). Partial sums accumulate on the CPU in fp32 (deterministic, so the
# hardware-captured gold rule still applies). Latency includes the CPU loops -
# exactly the part autocomp can optimize (block size, reuse, fewer reloads).
# ============================================================================

BLK = 64

_TILED_PREAMBLE = r'''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/{header}"

#define DIM 16
#define BLK 64
#define BLK_OUTC (BLK / 4)
#define BLOCKS_M (MATMUL_M / BLK)
#define BLOCKS_N (MATMUL_N / BLK)
#define BLOCKS_K (MATMUL_K / BLK)
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
// MATMUL_M/K/N are PADDED to BLK (M,N) / DIM (K); REAL_* are the true dims. The block loop
// covers padded blocks; the last M/N block is partial and uses loop_ws pad to skip the rest.
#define REAL_M {real_m}
#define REAL_N {real_n}
typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif

// All DMA-visible buffers at FIXED DRAM addresses so the capture and the
// optimization harness use bit-identical addresses regardless of binary layout
// (the spike MX model's behavior depends on buffer addresses).
#define A_blk          ((elem_t (*)[MATMUL_K])      0xA0000000UL)
#define B_blk          ((elem_t (*)[MATMUL_K])      0xA0100000UL)
#define A_s            ((uint8_t (*)[BLK])          0xA0200000UL)
#define B_s            ((uint8_t (*)[BLK])          0xA0280000UL)
#define C_blk          ((out_t (*)[BLK_OUTC])       0xA0300000UL)
#define scale_factors  ((uint32_t *)                0xA0400000UL)
#define C_acc          ((float (*)[MATMUL_N])       0xA0800000UL)
#define C_hw           ((out_t (*)[OUT_COLS])       0xA1000000UL)  // bf16-packed row-major output
#define SCALE_FACTORS_BYTES (1 << 22)
#define A_S_BYTES   ((MATMUL_K / 32) * BLK)
#define B_S_BYTES   ((MATMUL_K / 32) * BLK)   // requantizer scale stream; advances per launch -- keep large + zeroed

static inline float bf16f(uint16_t b) {{
  union {{ uint32_t u; float f; }} v; v.u = ((uint32_t)b) << 16; return v.f;
}}
static inline uint32_t fbits(float f) {{
  union {{ uint32_t u; float f; }} v; v.f = f; return v.u;
}}
'''

_TILED_KERNEL = r'''  for (int i0 = 0; i0 < BLOCKS_M; i0++)
   for (int j0 = 0; j0 < BLOCKS_N; j0++) {
      // Partial last M/N block: compute only the real rows/cols, pad the rest.
      int mrows = REAL_M - i0 * BLK; if (mrows > BLK) mrows = BLK; if (mrows <= 0) continue;
      int ncols = REAL_N - j0 * BLK; if (ncols > BLK) ncols = BLK; if (ncols <= 0) continue;
      // PROF cpu_stage
      // Stage only the tiny per-32-group scales; A/B data is mvin'd DIRECTLY
      // from DRAM (no host-side block memcpy — that was 98% of the cycles).
      for (int g = 0; g < MATMUL_K / 32; g++) {
        memcpy(&A_s[g][0], &A_scales_row[g][i0 * BLK], BLK);
        memcpy(&B_s[g][0], &B_scales_col[g][j0 * BLK], BLK);
      }
      int tiles = BLK / DIM, tiles_K = MATMUL_K / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
      int SPAD_DEST = 128;
      int pad_i = BLK - mrows, pad_j = BLK - ncols;
      // PROF accel
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)A_s, A_S_BYTES, 0);
      gemmini_mx_load_scales((uint64_t)B_s, B_S_BYTES, 1);
      gemmini_config_ld(MATMUL_K * sizeof(elem_t));      // A rows: stride K (full block-aligned tiles)
      for (int i = 0; i < tiles; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)A_in) + (i0 * BLK + i * DIM) * MATMUL_K + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_config_ld(MATMUL_N * sizeof(elem_t));      // B rows: stride N
      for (int j = 0; j < tiles; j++)                    // B[k][j0*BLK+j] -> slot (k*tiles + j)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + (j0 * BLK + j * DIM)),
                                b_base + (k * tiles + j) * DIM, DIM, DIM);
      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles, tiles, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles, tiles, tiles_K, pad_i, pad_j, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      // Accelerator-centric readout: per-row hardware DMA from smem straight into the
      // bf16-packed row-major output (block-internal smem row stride is BLK).
      for (int r = 0; r < mrows; r++)
        gemmini_mx_read_smem(&C_hw[i0 * BLK + r][(j0 * BLK) / BF16_PER_WORD],
                             SPAD_DEST * 16 + r * BLK, ncols);
      gemmini_fence();
    }
'''

_TILED_CAPTURE = _TILED_PREAMBLE + r'''
int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  // Replicate the autocomp harness wrapper exactly (flush + fence before, fence
  // after): gemmini_fence() advances MX scale-stream state, so capture and
  // candidate must execute identical fence sequences.
  gemmini_flush(0);
  fence();
  {{
{body}
  }}
  fence();
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < OUT_COLS; j++)
      printf("%08x\n%08x\n", (unsigned)(C_hw[i][j] >> 32), (unsigned)(C_hw[i][j] & 0xffffffffu));
  printf("GOLD_END\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''

_TILED_OPT = _TILED_PREAMBLE + r"""
#define OUTPUT_MATRIX_NAME C_hw
// GOLD_FILE: {gold_file}
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {{
  (void)x; (void)y; return 1;  // correctness checked vs GOLD_FILE by the runner
}}
static out_t gold[1][1];

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  // SUBSTITUTE HERE
  // SUBSTITUTE END
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < REAL_M; i++)                 // dump only the REAL sub-block: padding writes by a
    for (int j = 0; j < REAL_N / BF16_PER_WORD; j++)  // candidate can't cause a false mismatch
      printf("%08x\n%08x\n", (unsigned)(C_hw[i][j] >> 32), (unsigned)(C_hw[i][j] & 0xffffffffu));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
"""

def gen_matmul_problem_tiled(M, K, N, prob_type="gemmini-mx-matmul-tiled", prob_id=0, fmt="fp8:e4m3"):
    """Outer-tiled matmul problem for shapes that exceed the scratchpad."""
    if _FMT_TAG.get(fmt, "fp8") != "fp8":
        return _gen_matmul_tiled_fp64(M, K, N, prob_type, prob_id, fmt)  # fp6/fp4 path
    if K % DIM:
        raise ValueError(f"K must be a multiple of {DIM} (got K={K})")
    Mp, Np = ceil_to(M, BLK), ceil_to(N, BLK)   # pad M,N to the block size; K stays (must be %16)
    Kp = K
    if 2 * BLK * (Kp // DIM) > SPAD_ROWS:
        raise ValueError(f"K={K} too large: blocks exceed scratchpad")
    tag = _FMT_TAG.get(fmt, "fp8")
    header = f"mxgen_{tag}_{Mp}x{Kp}x{Np}.h"
    gen_input_header(Mp, Kp, Np, fmt, GEMMINI_SW / "include" / header)

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    gold_file = hdir / f"gold{prob_id}.txt"

    harness = _TILED_OPT.format(header=header, gold_file=gold_file, real_m=M, real_n=N)
    (hdir / f"test{prob_id}.c").write_text(harness)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        "// AUTO-GENERATED outer-tiled MX matmul baseline (%dx%dx%d).\n"
        "void solution(void) {\n%s}\n" % (M, K, N, _TILED_KERNEL))

    # gold = baseline run through the very same harness + wrapper path
    from autocomp.search.prob import Prob
    from autocomp.backend.gemmini.gemmini_eval import clean_code
    prob = Prob(prob_type, prob_id)
    body = clean_code((sdir / f"sol{prob_id}_exo_baseline.c").read_text())
    tc = prob.tests[0].get_test_code([body])
    tc = tc.replace("// GOLD_FILE:", "// CAPTURING (no gold check):")
    rd = {}
    E.run_spike(tc, rd, GEMMINI_PATH, "0", 1800)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"tiled gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"[0-9a-fA-F]{8}", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    if len(vals) != M * N // 2:   # REAL sub-block dump: M*(N/4) uint64, 2 words each
        raise RuntimeError(f"tiled gold: expected {M*N//2} words, got {len(vals)}")
    gold_file.write_text("\n".join(vals))
    return prob_type, prob_id, header


def _gen_matmul_tiled_fp64(M, K, N, prob_type, prob_id, fmt):
    """Outer-tiled fp6/fp4 matmul: block M,N into BLK; each block M-packs its A rows,
    stages its scales, mvins B sub-byte (+fp6 LUTs), runs the full-K block matmul, and
    assembles the bf16 result into C_acc (fp32). For shapes exceeding the scratchpad."""
    if M % BLK or N % BLK or K % GROUP:
        raise ValueError(f"M,N must be multiples of {BLK}, K of {GROUP} (got {M}x{K}x{N})")
    tag = _FMT_TAG[fmt]; cfg = _FMT_HW[fmt]; lut = cfg["lut"]
    header = f"mxgen_{tag}_{M}x{K}x{N}.h"
    gen_input_header(M, K, N, fmt, GEMMINI_SW / "include" / header)
    gl = BLK.bit_length() if lut else 1   # block-local LUT index (BLK rows/cols) -> 1 group
    lutdecl = ("static uint8_t A_lp[12], B_lp[12];\n"
               "static void mx_pack_lut(const uint8_t *cc, uint8_t *o){uint64_t lo=0;uint32_t hi=0;"
               "for(int i=0;i<16;i++){int bit=i*6;uint64_t v=cc[i]&0x3F;if(bit+6<=64)lo|=v<<bit;"
               "else if(bit>=64)hi|=(uint32_t)(v<<(bit-64));else{lo|=v<<bit;hi|=(uint32_t)(v>>(64-bit));}}"
               "for(int b=0;b<8;b++)o[b]=(lo>>(b*8))&0xFF;for(int b=0;b<4;b++)o[8+b]=(hi>>(b*8))&0xFF;}\n") if lut else ""
    lutpack = "  mx_pack_lut((const uint8_t*)A_lut,A_lp); mx_pack_lut((const uint8_t*)B_lut,B_lp);\n" if lut else ""
    lutload = "      gemmini_mx_load_lut((uint64_t)B_lp,1,0); gemmini_mx_load_lut((uint64_t)A_lp,1,1);\n" if lut else ""

    def pre(gold_marker):
        return (
            '#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n'
            '#ifndef BAREMETAL\n#include <sys/mman.h>\n#include <stdlib.h>\n#endif\n'
            '#include "include/gemmini_testutils.h"\n#include "include/%H%"\n'
            '#define DIM 16\n#define BLK 64\n#define BLK_OUTC (BLK/4)\n'
            '#define BLOCKS_M (MATMUL_M/BLK)\n#define BLOCKS_N (MATMUL_N/BLK)\n'
            '#define BF16_PER_WORD 4\ntypedef uint8_t elem_t; typedef uint64_t out_t;\n'
            '#ifndef fence\n#define fence() gemmini_fence()\n#endif\n'
            'static elem_t A_hw[BLK/2][MATMUL_K];\n'
            'static uint8_t A_s[MATMUL_K/32][BLK], B_s[MATMUL_K/32][BLK];\n'
            '#define OUT_COLS (MATMUL_N/BF16_PER_WORD)\nstatic out_t C_hw[MATMUL_M][OUT_COLS];\n' + lutdecl +
            '#define OUTPUT_MATRIX_NAME C_hw\n' + gold_marker + '\n'
            'int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]){(void)x;(void)y;return 1;}\n'
            'static out_t gold[1][1];\n'
        ).replace("%H%", header)

    body = (lutload +
        "  for(int i0=0;i0<BLOCKS_M;i0++)for(int j0=0;j0<BLOCKS_N;j0++){\n"
        "    for(int g=0;g<MATMUL_K/32;g++)for(int b=0;b<BLK;b++){A_s[g][b]=A_scales_row[g][i0*BLK+b];B_s[g][b]=B_scales_col[g][j0*BLK+b];}\n"
        "    for(int m=0;m<BLK;m++)for(int k=0;k<MATMUL_K;k++){uint8_t by=A_in[i0*BLK+m][k>>1];uint8_t cd=(k&1)?((by>>4)&0xF):(by&0xF);\n"
        "      if((m&1)==0)A_hw[m>>1][k]=(A_hw[m>>1][k]&0xF0)|cd; else A_hw[m>>1][k]=(A_hw[m>>1][k]&0x0F)|(cd<<4);}\n"
        "    int ti=BLK/DIM/2, tj=BLK/DIM/2, tk=MATMUL_K/DIM; uint32_t ab=0,bb=BANK_NUM*BANK_ROWS-tk*tj*DIM;\n"
        "    gemmini_flush(0);\n"
        f"    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,{cfg['actf']},{cfg['wgtf']},3,{cfg['uselut']});\n"
        "    gemmini_mx_load_scales((uint64_t)A_s,(MATMUL_K/32)*BLK,0);\n"
        "    gemmini_mx_load_scales((uint64_t)B_s,(MATMUL_K/32)*BLK,1);\n"
        "    gemmini_config_st(BLK_OUTC*sizeof(out_t));\n"
        f"    gemmini_mxquant_config_mvout((uint64_t)scale_factors,ti,tj,tk,0,0,{gl});\n"
        "    gemmini_config_ld(MATMUL_K*sizeof(elem_t));\n"
        "    for(int i=0;i<ti;i++)for(int k=0;k<tk;k++) gemmini_extended_mvin((void*)(((elem_t*)A_hw)+i*DIM*MATMUL_K+k*DIM),ab+(i*tk+k)*DIM,DIM,DIM);\n"
        "    gemmini_config_ld(MATMUL_N*sizeof(elem_t)/2);\n"
        "    for(int k=0;k<tk;k++)for(int j=0;j<tj;j++) gemmini_extended_mvin((void*)(((elem_t*)B_in)+k*DIM*(MATMUL_N/2)+(j0*BLK)/2+j*DIM),bb+(k*tj+j)*DIM,DIM,DIM);\n"
        "    gemmini_loop_ws_spad(ti,tj,tk,0,0,0,ab,BANK_NUM*BANK_ROWS,0,128,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);\n"
        "    for(int r=0;r<BLK;r++) gemmini_mx_read_smem(&C_hw[i0*BLK+r][(j0*BLK)/BF16_PER_WORD], 128*16 + r*BLK, BLK);\n"
        "    gemmini_fence();\n"
        "  }\n")

    hdir = HARNESSES_DIR / prob_type; sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    gold_file = hdir / f"gold{prob_id}.txt"
    mlock = "#ifndef BAREMETAL\n  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0){perror(\"mlockall\");return 1;}\n#endif\n"
    decl = "  uint32_t scale_factors[512] __attribute__((aligned(32)))={0};\n"
    dump = ('  printf("GOLD_BEGIN\\n");\n  for(int i=0;i<MATMUL_M;i++)for(int j=0;j<MATMUL_N/BF16_PER_WORD;j++)\n'
            '    printf("%08x\\n%08x\\n",(unsigned)(C_hw[i][j]>>32),(unsigned)(C_hw[i][j]&0xffffffffu));\n  printf("GOLD_END\\n");\n')

    cap = (pre("// CAPTURING") + "\nint main(){\n" + mlock + decl +
           "  memset(C_hw,0,sizeof(C_hw));\n" + lutpack + body + dump +
           "#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n")
    rd = {}
    E.run_spike(cap, rd, GEMMINI_PATH, "0", 1800)
    out = rd.get("retval", "")
    if not isinstance(out, str) or "GOLD_BEGIN" not in out:
        raise RuntimeError(f"{tag} tiled gold capture failed: {str(out)[:300]}")
    vals = re.findall(r"\b[0-9a-fA-F]{8}\b", out.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
    if len(vals) != M * N // 2:
        raise RuntimeError(f"{tag} tiled gold: expected {M*N//2} words, got {len(vals)}")
    gold_file.write_text("\n".join(vals))

    opt = (pre(f"// GOLD_FILE: {gold_file}") + "\n#define REPEAT_TEST_ITERS 1\n#define RUN_BASELINE_CODE 1\nint main(){\n"
           + mlock + decl + "  memset(C_hw,0,sizeof(C_hw));\n" + lutpack +
           "  for(int repeat_iters=0;repeat_iters<REPEAT_TEST_ITERS;repeat_iters++){\n"
           "    // SUBSTITUTE HERE\n    // SUBSTITUTE END\n  }\n" + dump +
           '  printf("Correct result\\n");\n#ifndef BAREMETAL\n  exit(0);\n#else\n  return 0;\n#endif\n}\n')
    (hdir / f"test{prob_id}.c").write_text(opt)
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        f"// AUTO-GENERATED outer-tiled MX matmul baseline ({M}x{K}x{N}, {fmt}).\nvoid solution(void) {{\n" + body + "}\n")
    return prob_type, prob_id, header
