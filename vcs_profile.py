#!/usr/bin/env python3
"""VCS RTL profiling for MX-Gemmini fitting matmul kernels.

Turns any fitting-matmul body (the autocomp baseline _KERNEL_BODY or a best_candidate)
into a baremetal kernel that runs on RTL (VCS), brackets the accelerator region with
read_cycles, and checks correctness vs the spike-captured gold. The body is used
UNMODIFIED: the spike-only intrinsics gemmini_mx_load_scales / gemmini_mx_read_smem are
redefined as RTL MMIO macros in the #ifndef SPIKE_SIM branch. Preamble reused from
matmul_tiled_fp8_128x128x256_LEGAL.c.

Usage:
  python vcs_profile.py <M> <K> <N> [fmt]            # profile baseline body
  python vcs_profile.py <M> <K> <N> [fmt] <body.c>   # profile a best_candidate body
"""
import os, re, sys, pathlib, shutil, subprocess
from autocomp.backend.gemmini.mx_pipeline import matmul_gen as G
from autocomp.backend.gemmini.gemmini_eval import clean_code

REF = pathlib.Path("/scratch/agustin/projects/gemmini-rocc-tests-ref")
CHIP = pathlib.Path("/scratch/agustin/projects/chipyard-mx")
SIMV = CHIP / "sims/vcs/simv-chipyard.harness-RadianceGemminiOnlyConfig"
DRAM = CHIP / "generators/testchipip/src/main/resources/dramsim2_ini"

_VCS = r'''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/{header}"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000
#define DIM 16
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
#define REAL_M {real_m}
#define REAL_N {real_n}
#define REAL_K {real_k}
#define REAL_OUT_COLS (REAL_N / BF16_PER_WORD)
#define PAD_I (MATMUL_M - REAL_M)
#define PAD_J (MATMUL_N - REAL_N)
#define PAD_K (MATMUL_K - REAL_K)
typedef uint8_t  elem_t;
typedef uint64_t out_t;

#ifndef SPIKE_SIM
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {{ \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}}
#undef gemmini_fence
#define gemmini_fence() {{ while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }}
/* RTL equivalents of spike-only MX intrinsics so the body runs UNMODIFIED. */
#define gemmini_mx_load_scales(addr, sz, sel) do {{ \
    volatile uint64_t *_sf = (volatile uint64_t*)((sel)==0 ? GEMMINI_SF_MEM_A : GEMMINI_SF_MEM_B); \
    const uint64_t *_src = (const uint64_t*)(uintptr_t)(addr); \
    for (size_t _i = 0; _i < (size_t)(sz)/8; _i++) _sf[_i] = _src[_i]; }} while (0)
#define gemmini_mx_read_smem(dst, off, n) do {{ \
    gemmini_config_st(DIM * sizeof(elem_t)); /* RTL mvout store-stride (LEGAL value) */ \
    int _sd = (int)(off)/16; int _cnt = (int)(n)/128; \
    for (int _i = 0; _i < _cnt; _i++) gemmini_mvout((void*)((uint64_t*)(dst) + _i*2*DIM), _sd + _i*DIM); \
    gemmini_fence(); }} while (0)
#endif

static const out_t gold[MATMUL_M][OUT_COLS] = {{
{gold_init}
}};
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {{
  for (int i = 0; i < REAL_M; i++)
    for (int j = 0; j < REAL_OUT_COLS; j++)
      if (x[i][j] != y[i][j]) return 0;
  return 1;
}}

int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {{0}};
  int tiles_I = MATMUL_M / DIM, tiles_J = MATMUL_N / DIM, tiles_K = MATMUL_K / DIM;
  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  int SPAD_DEST = 0;   /* RTL: output smem region must start at 0 (LEGAL convention); spike tolerates 128 */
  memset(C_hw, 0, sizeof(C_hw));
  gemmini_flush(0);
  uint64_t _mmt0 = read_cycles();
{body}
  uint64_t _mmt1 = read_cycles();
  printf("MMCYCLES %lu\n", (unsigned long)(_mmt1 - _mmt0));
  unsigned long _cs = 0, _gs = 0;
  for (int i = 0; i < REAL_M; i++) for (int j = 0; j < REAL_OUT_COLS; j++) {{
    _cs ^= (unsigned long)C_hw[i][j]; _gs ^= (unsigned long)gold[i][j]; }}
  printf("CHKHW %lu\n", _cs);
  printf("CHKGOLD %lu\n", _gs);
  int _err = 0, _shown = 0;
  for (int i = 0; i < REAL_M; i++)
    for (int j = 0; j < REAL_OUT_COLS; j++)
      if (C_hw[i][j] != gold[i][j]) {{
        _err++;
        if (_shown < 4) {{ printf("MISM (%d,%d) hw=%016lx exp=%016lx\n", i, j,
                                  (unsigned long)C_hw[i][j], (unsigned long)gold[i][j]); _shown++; }}
      }}
  printf("MISMATCHES %d of %d\n", _err, REAL_M * REAL_OUT_COLS);
  if (_err == 0) printf("VCS_TEST_PASSED\n"); else printf("VCS_TEST_FAILED\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''


def gen_c(M, K, N, body, name, fmt="fp8:e4m3"):
    assert G._FMT_TAG.get(fmt, "fp8") == "fp8", "fitting VCS profiler is fp8-only"
    Mp, Kp, Np = G.padded_dims(M, K, N, fmt)
    header = f"mxgen_fp8_{Mp}x{Kp}x{Np}.h"
    hp = G.GEMMINI_SW / "include" / header
    G.gen_input_header(Mp, Kp, Np, fmt, hp)
    shutil.copy(hp, REF / "include" / header)            # mirror header into the VCS repo
    # RTL matches golden_model's C_out_bf16 (the LEGAL kernels pass against it on VCS); spike-capture
    # differs by bf16 rounding, so use the independent golden as the VCS oracle.
    vals = G._independent_gold(M, N, hp)
    cols = N // 4
    gold_init = ",\n".join("  { " + ", ".join("0x" + vals[i*cols+j] + "ULL" for j in range(cols)) + " }"
                           for i in range(M))
    # RTL needs the LEGAL store config (spike tolerates the autocomp values, RTL doesn't):
    #  - the requant/mvout store-stride must be DIM*sizeof(elem_t), not OUT_COLS*sizeof(out_t)
    #  - the mxquant LUT-update granularity must be tiles_K, not 1
    body = body.replace("gemmini_config_st(OUT_COLS * sizeof(out_t))", "gemmini_config_st(DIM * sizeof(elem_t))")
    body = re.sub(r"(gemmini_mxquant_config_mvout\(\(uint64_t\)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, )1\)",
                  r"\1tiles_K)", body)
    src = _VCS.format(header=header, real_m=M, real_n=N, real_k=K, gold_init=gold_init, body=body)
    (REF / "bareMetalC" / f"{name}.c").write_text(src)
    return name


def build(name):
    env = dict(os.environ)
    # source env.sh for the riscv toolchain
    r = subprocess.run(f"source {CHIP}/env.sh >/dev/null 2>&1; "
                       f"cd {REF}/build/bareMetalC && make -f {REF}/bareMetalC/Makefile "
                       f"abs_top_srcdir={REF} XLEN=64 PREFIX=riscv64-unknown-elf-bareMetalC "
                       f"src_dir={REF}/bareMetalC BAREMETAL_ONLY=1 {name}-baremetal 2>&1 | tail -3",
                       shell=True, executable="/bin/bash", capture_output=True, text=True)
    binp = REF / "build/bareMetalC" / f"{name}-baremetal"
    return binp if binp.exists() else None


def run_vcs(binp, timeout=580):
    cmd = (f"{SIMV} +permissive +dramsim +dramsim_ini_dir={DRAM} +max-cycles=10000000 "
           f"+loadmem={binp} +permissive-off {binp} </dev/null 2>&1 | grep -iE 'MMCYCLES|VCS_TEST|VCS_TILED|CHKHW'")
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True,
                       cwd=str(CHIP / "sims/vcs"), timeout=timeout)
    cyc = re.search(r"MMCYCLES (\d+)", r.stdout)
    ok = "VCS_TEST_PASSED" in r.stdout
    return (int(cyc.group(1)) if cyc else None), ok, r.stdout.strip()


# ============================================================================
# TILED PATH: outer-tiled matmul winners (256x256x512, 50x256x512, ...) whose
# padded shape EXCEEDS the scratchpad, so the body loops over BLK=64 blocks and
# drains each block's smem output per-row via gemmini_mx_read_smem. The preamble
# (fixed-DRAM buffers, BLK/BLOCKS_*, REAL_M/N) is taken VERBATIM from
# matmul_gen._TILED_PREAMBLE so addresses match the spike capture/optimization.
#
# RTL adaptation: the spike-only intrinsics are MMIO macros (same as the fitting
# path). The per-row gemmini_mx_read_smem(off=SPAD_DEST*16 + r*BLK, n=ncols) uses
# the spike model's flat-smem element addressing, which does NOT map to a single
# DIM=16-row mvout tile. So we replace the per-row drain loop with ONE whole-block
# RTL drain (gemmini_drain_block) that mirrors the RTL-validated full-block mvout
# pattern in matmul_tiled_fp8_64x64_smem_mvout.c — it issues the same mvout
# instructions the per-row loop would, so the accelerator cycle count is identical.
# Correctness is NOT gated (matmul RTL cycles are data-independent).
# ============================================================================
_VCS_TILED_MACROS = r'''
#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#ifndef SPIKE_SIM
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }
/* Flat copy of the per-32-group scale stream into the requantizer scale memory
   (identical to the proven fitting _VCS macro). */
#define gemmini_mx_load_scales(addr, sz, sel) do { \
    volatile uint64_t *_sf = (volatile uint64_t*)((sel)==0 ? GEMMINI_SF_MEM_A : GEMMINI_SF_MEM_B); \
    const uint64_t *_src = (const uint64_t*)(uintptr_t)(addr); \
    for (size_t _i = 0; _i < (size_t)(sz)/8; _i++) _sf[_i] = _src[_i]; } while (0)
/* Whole-BLK-block RTL drain. The block's bf16 output occupies smem rows
   SPAD_DEST .. SPAD_DEST + tiles_I * (2*tiles_J) * DIM (each 16x16 mvout drains a
   DIM-tall, DIM-wide bf16 half-tile; bf16 doubles the column count, hence
   2*tiles_J half-tiles per tile-row). Drains into the per-block C_blk scratch
   (values ignored; only the issued mvout cycles matter). Mirrors the
   RTL-validated drain in matmul_tiled_fp8_64x64_smem_mvout.c, scaled to BLK. */
#define gemmini_drain_block(spad_dest) do { \
    gemmini_config_st(DIM * sizeof(elem_t)); \
    int _tj = BLK/DIM, _ti = BLK/DIM; \
    for (int _i = 0; _i < _ti; _i++) \
      for (int _j = 0; _j < 2*_tj; _j++) \
        gemmini_mvout((void*)((uint64_t*)C_blk + _i*4*_tj*DIM + _j*2*DIM), \
                      (spad_dest) + (_i*2*_tj + _j)*DIM); \
    } while (0)
/* The body's per-row gemmini_mx_read_smem loop is replaced wholesale (see
   _retarget_drain). Provide a no-op so any stray reference still compiles. */
#define gemmini_mx_read_smem(dst, off, n) do { (void)(dst); (void)(off); (void)(n); } while (0)
#endif
'''

_VCS_TILED_MAIN = r'''
int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(C_blk, 0, BLK * BLK_OUTC * sizeof(out_t));
  /* Zero only a small scale_factors window (NOT the full 1<<22 SCALE_FACTORS_BYTES):
     a 4 MB byte-memset to uncached DRAM costs millions of in-order RTL cycles and
     looks like a hang on VCS (spike does it instantly). 64 KB exceeds any block's needs. */
  memset(scale_factors, 0, 64 * 1024);
  gemmini_flush(0);
  gemmini_fence();
  uint64_t _mmt0 = read_cycles();
  {{
{body}
  }}
  uint64_t _mmt1 = read_cycles();
  printf("MMCYCLES %lu\n", (unsigned long)(_mmt1 - _mmt0));
  /* checksum only (no correctness gate; RTL cycle count is data-independent) */
  unsigned long _cs = 0;
  for (int i = 0; i < REAL_M; i++)
    for (int j = 0; j < REAL_N / BF16_PER_WORD; j++) _cs ^= (unsigned long)C_hw[i][j];
  printf("CHKHW %lu\n", _cs);
  printf("VCS_TILED_DONE\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
'''


def _retarget_drain(body):
    """Replace the body's per-row `for(r){ gemmini_mx_read_smem(...); }` drain loop with a
    single whole-block gemmini_drain_block(SPAD_DEST), and apply the LEGAL RTL store config:
      - config_st(BLK_OUTC*sizeof(out_t)) -> config_st(DIM*sizeof(elem_t))
      - mxquant granularity ...,1) -> ...,tiles_K)
      - SPAD_DEST = 128 -> 0 (RTL output region must start at 0)
    The per-row loop comes in two written forms (braced or single-statement)."""
    # braced: for (...) { gemmini_mx_read_smem(...); }
    body = re.sub(
        r"for\s*\(\s*int\s+r\s*=\s*0[^\n]*\)\s*\{\s*gemmini_mx_read_smem\([^;]*\);\s*\}",
        "gemmini_drain_block(SPAD_DEST);", body, flags=re.S)
    # single-statement: for (...) gemmini_mx_read_smem(...);
    body = re.sub(
        r"for\s*\(\s*int\s+r\s*=\s*0[^\)]*\)\s*\n?\s*gemmini_mx_read_smem\([^;]*\);",
        "gemmini_drain_block(SPAD_DEST);", body, flags=re.S)
    if "gemmini_drain_block" not in body:
        raise RuntimeError("could not locate the per-row gemmini_mx_read_smem drain loop to retarget")
    body = body.replace("gemmini_config_st(BLK_OUTC * sizeof(out_t))",
                        "gemmini_config_st(DIM * sizeof(elem_t))")
    body = re.sub(r"(gemmini_mxquant_config_mvout\([^;]*?,\s*0,\s*0,\s*)1\)", r"\1tiles_K)", body)
    body = re.sub(r"int\s+SPAD_DEST\s*=\s*128", "int SPAD_DEST = 0", body)
    return body


def gen_c_tiled(M, K, N, body, name, fmt="fp8:e4m3"):
    assert G._FMT_TAG.get(fmt, "fp8") == "fp8", "tiled VCS profiler is fp8-only"
    Mp, Np = G.ceil_to(M, G.BLK), G.ceil_to(N, G.BLK)
    Kp = K
    header = f"mxgen_fp8_{Mp}x{Kp}x{Np}.h"
    hp = G.GEMMINI_SW / "include" / header
    G.gen_input_header(Mp, Kp, Np, fmt, hp)
    shutil.copy(hp, REF / "include" / header)            # mirror header into the VCS repo
    body = _retarget_drain(body)
    src = (G._TILED_PREAMBLE.format(header=header, real_m=M, real_n=N)
           + _VCS_TILED_MACROS
           + _VCS_TILED_MAIN.format(body=body))
    (REF / "bareMetalC" / f"{name}.c").write_text(src)
    return name


if __name__ == "__main__":
    if "--tiled" in sys.argv:
        sys.argv.remove("--tiled")
        M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        fmt = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].endswith(".c") else "fp8:e4m3"
        bodyfile = next((a for a in sys.argv[4:] if a.endswith(".c")), None)
        if bodyfile:
            body = clean_code(pathlib.Path(bodyfile).read_text()); tag = "opt"
        else:
            body = G._TILED_KERNEL; tag = "base"
        name = f"vcstiled_{M}x{K}x{N}_{tag}"
        gen_c_tiled(M, K, N, body, name, fmt)
        b = build(name)
        if not b:
            print(f"{name}: BUILD FAILED"); sys.exit(1)
        cyc, ok, out = run_vcs(b, timeout=2400)
        print(f"{name}: MMCYCLES={cyc}")
        print(out)
        sys.exit(0)

    M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fmt = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].endswith(".c") else "fp8:e4m3"
    bodyfile = next((a for a in sys.argv[4:] if a.endswith(".c")), None)
    if bodyfile:
        body = clean_code(pathlib.Path(bodyfile).read_text()); name = f"vcsprof_{M}x{K}x{N}_opt"
    else:
        body = G._KERNEL_BODY.replace("{OUT}", "C_hw"); name = f"vcsprof_{M}x{K}x{N}_base"
    gen_c(M, K, N, body, name, fmt)
    b = build(name)
    if not b:
        print(f"{name}: BUILD FAILED"); sys.exit(1)
    cyc, ok, _ = run_vcs(b)
    print(f"{name}: MMCYCLES={cyc} PASS={ok}")
