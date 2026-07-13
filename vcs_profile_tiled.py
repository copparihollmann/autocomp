#!/usr/bin/env python3
"""VCS RTL profiling for MX-Gemmini OUTER-TILED matmul kernels.

Companion to vcs_profile.py (which handles FITTING bodies). This handles the
fixed-DRAM-buffer outer-tiled bodies (matmul_gen._TILED_PREAMBLE / _TILED_KERNEL
style, e.g. autocomp's best_candidate_so_far.c for shapes that exceed the
scratchpad). It brackets the accelerator region with read_cycles and prints
MMCYCLES. Correctness is NOT checked: matmul RTL cycle count is data-independent
(it depends only on shape + instruction schedule), and the on-device fp8 header
arithmetic fidelity is a separate unsolved issue. READ MMCYCLES; IGNORE PASS/FAIL.

The two spike-only intrinsics are redefined as RTL MMIO macros so the body builds:
  * gemmini_mx_load_scales -> flat copy to GEMMINI_SF_MEM_A / _B (same as the
    fitting _VCS macro).
  * gemmini_mx_read_smem -> the RTL has NO k_MX_READ_SMEM op. The tiled body calls
    it PER ROW (num_bf16 = ncols = BLK), which is not a natural 16x16 mvout. So we
    do NOT translate it per-row; instead we POST-PROCESS the body, replacing the
    per-row readout loop with the RTL-validated per-BLOCK tile drain (the same
    `for i in 0..tiles*tiles*2: gemmini_mvout(dst + i*2*DIM, SPAD_DEST + i*DIM)`
    pattern proven on RTL by matmul_tiled_fp8_128x128x256_WINNER.c). One BLKxBLK
    block = (BLK/DIM)^2 logical tiles x2 (bf16 hi/lo) mvouts = the same total
    output data the per-row loop would move, so the drain work — hence cycles — is
    faithful. The drain target is the per-block C_blk scratch buffer (keeps
    addresses in-bounds; values are ignored).

Usage:
  python vcs_profile_tiled.py <M> <K> <N> <body.c>   # profile a tiled candidate body
  python vcs_profile_tiled.py <M> <K> <N>            # profile the _TILED_KERNEL baseline
"""
import os, re, sys, pathlib, shutil, subprocess
from autocomp.backend.gemmini.mx_pipeline import matmul_gen as G
from autocomp.backend.gemmini.gemmini_eval import clean_code

REF = pathlib.Path("/scratch/agustin/projects/gemmini-rocc-tests-ref")
CHIP = pathlib.Path("/scratch/agustin/projects/chipyard-mx")
SIMV = CHIP / "sims/vcs/simv-chipyard.harness-RadianceGemminiOnlyConfig"
DRAM = CHIP / "generators/testchipip/src/main/resources/dramsim2_ini"

# RTL MMIO macros + spike-intrinsic RTL equivalents, inserted after the preamble.
_RTL_MACROS = r'''
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
/* Flat copy of the per-32-group scale stream into the requantizer scale memory,
   exactly like the proven fitting _VCS macro. */
#define gemmini_mx_load_scales(addr, sz, sel) do { \
    volatile uint64_t *_sf = (volatile uint64_t*)((sel)==0 ? GEMMINI_SF_MEM_A : GEMMINI_SF_MEM_B); \
    const uint64_t *_src = (const uint64_t*)(uintptr_t)(addr); \
    for (size_t _i = 0; _i < (size_t)(sz)/8; _i++) _sf[_i] = _src[_i]; } while (0)
/* RTL per-BLOCK tile drain (substituted for the body's per-row read_smem loop).
   Mirrors the RTL-validated drain in matmul_tiled_fp8_64x64_smem_mvout.c:
     for i in tiles_I: for j in 2*tiles_J:
       mvout(C_hw + i*4*tiles_J + j*2,  SPAD_DEST + (i*2*tiles_J + j)*DIM)
   scaled to a BLK x BLK block (tiles_I = tiles_J = BLK/DIM). The block's bf16
   output occupies smem rows SPAD_DEST .. + tiles_I*2*tiles_J*DIM; each 16x16 mvout
   drains one DIM-tall, DIM-wide bf16 tile (2 mvouts per logical tile = hi/lo half).
   dst here is the per-block C_blk scratch (values ignored; cycles are what matter). */
#define gemmini_drain_block(spad_dest) do { \
    gemmini_config_st(DIM * sizeof(elem_t)); \
    int _tj = BLK/DIM, _ti = BLK/DIM; \
    for (int _i = 0; _i < _ti; _i++) \
      for (int _j = 0; _j < 2*_tj; _j++) \
        gemmini_mvout((void*)((uint64_t*)C_blk + _i*4*_tj*DIM + _j*2*DIM), \
                      (spad_dest) + (_i*2*_tj + _j)*DIM); \
    } while (0)
#endif
'''

_MAIN = r'''
int main() {{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {{ perror("mlockall"); return 1; }}
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(C_blk, 0, BLK * BLK_OUTC * sizeof(out_t));
  /* Zero only a small scale_factors window, NOT the full 1<<22 SCALE_FACTORS_BYTES:
     a 4 MB byte-memset to uncached DRAM costs millions of in-order RTL cycles and
     looks like a hang on VCS (spike does it instantly). The requantizer only touches
     i*j*k_bound entries per block; 64 KB is comfortably larger than any block needs. */
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


def _rewrite_readout(body):
    """Replace the body's per-row gemmini_mx_read_smem loop with the RTL per-block
    drain. The loop has the canonical shape:
        for (int r = 0; r < mrows; r++)
          gemmini_mx_read_smem(&C_hw[...][...], SPAD_DEST * 16 + r * BLK, ncols);
    (optionally brace-wrapped). We collapse it to a single gemmini_drain_block(SPAD_DEST*16)."""
    # Match the for-loop (braced or single-statement) whose body is a gemmini_mx_read_smem call.
    pat = re.compile(
        r"for\s*\([^;]*;\s*[^;]*;\s*[^)]*\)\s*\{?\s*"
        r"gemmini_mx_read_smem\s*\([^;]*?SPAD_DEST\s*\*\s*16\s*\+\s*r\s*\*\s*BLK[^;]*?\)\s*;\s*\}?",
        re.DOTALL)
    new, n = pat.subn("gemmini_drain_block(SPAD_DEST);", body)
    if n == 0:
        # Fall back: maybe a single per-row call not in a loop, or different spacing.
        pat2 = re.compile(r"gemmini_mx_read_smem\s*\([^;]*?\)\s*;", re.DOTALL)
        new, n = pat2.subn("gemmini_drain_block(SPAD_DEST);", body)
    return new, n


def gen_c(M, K, N, body, name, fmt="fp8:e4m3"):
    assert G._FMT_TAG.get(fmt, "fp8") == "fp8", "tiled VCS profiler is fp8-only"
    Mp, Np = G.ceil_to(M, G.BLK), G.ceil_to(N, G.BLK)
    Kp = K
    header = f"mxgen_fp8_{Mp}x{Kp}x{Np}.h"
    hp = G.GEMMINI_SW / "include" / header
    G.gen_input_header(Mp, Kp, Np, fmt, hp)
    shutil.copy(hp, REF / "include" / header)   # mirror header into the VCS repo

    body, nrw = _rewrite_readout(body)
    if nrw == 0:
        raise RuntimeError("could not find a per-row gemmini_mx_read_smem readout to rewrite")
    # Same LEGAL store config the fitting path applies (adapted to the tiled body):
    #  - the requantizer/mvout store-stride must be DIM*sizeof(elem_t), not the
    #    bf16-packed output stride. The tiled body uses BLK_OUTC*sizeof(out_t)
    #    (== the fitting body's OUT_COLS*sizeof(out_t)); leaving it makes the RTL
    #    requantizer run away (endless scale_factors writes -> fence never returns).
    body = body.replace("gemmini_config_st(BLK_OUTC * sizeof(out_t))",
                        "gemmini_config_st(DIM * sizeof(elem_t))")
    body = body.replace("gemmini_config_st(OUT_COLS * sizeof(out_t))",
                        "gemmini_config_st(DIM * sizeof(elem_t))")
    #  - mxquant LUT-update granularity tiles_K, not 1.
    body = re.sub(r"(gemmini_mxquant_config_mvout\([^;]*?,\s*)1\)", r"\1tiles_K)", body)
    #  - output smem region must start at 0 on RTL (spike tolerates 128). The tiled
    #    body hardcodes SPAD_DEST=128; with C=128 the loop_ws output/requantizer runs
    #    away (multi-million-cycle hang). RTL-validated kernels all use SPAD_DEST=0.
    body = re.sub(r"int\s+SPAD_DEST\s*=\s*128\s*;", "int SPAD_DEST = 0;", body)

    preamble = G._TILED_PREAMBLE.format(header=header, real_m=M, real_n=N)
    src = preamble + _RTL_MACROS + _MAIN.format(body=body)
    (REF / "bareMetalC" / f"{name}.c").write_text(src)
    return name


def build(name):
    r = subprocess.run(f"source {CHIP}/env.sh >/dev/null 2>&1; "
                       f"cd {REF}/build/bareMetalC && make -f {REF}/bareMetalC/Makefile "
                       f"abs_top_srcdir={REF} XLEN=64 PREFIX=riscv64-unknown-elf-bareMetalC "
                       f"src_dir={REF}/bareMetalC BAREMETAL_ONLY=1 {name}-baremetal 2>&1 | tail -20",
                       shell=True, executable="/bin/bash", capture_output=True, text=True)
    binp = REF / "build/bareMetalC" / f"{name}-baremetal"
    if not binp.exists():
        print(r.stdout)
    return binp if binp.exists() else None


def run_vcs(binp, timeout=2400):
    cmd = (f"{SIMV} +permissive +dramsim +dramsim_ini_dir={DRAM} +max-cycles=20000000 "
           f"+loadmem={binp} +permissive-off {binp} </dev/null 2>&1 | grep -iE 'MMCYCLES|CHKHW|VCS_TILED'")
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True,
                       cwd=str(CHIP / "sims/vcs"), timeout=timeout)
    cyc = re.search(r"MMCYCLES (\d+)", r.stdout)
    done = "VCS_TILED_DONE" in r.stdout
    return (int(cyc.group(1)) if cyc else None), done, r.stdout.strip()


if __name__ == "__main__":
    M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    bodyfile = next((a for a in sys.argv[4:] if a.endswith(".c")), None)
    if bodyfile:
        body = clean_code(pathlib.Path(bodyfile).read_text())
        name = f"vcstiled_{M}x{K}x{N}_opt"
    else:
        body = G._TILED_KERNEL
        name = f"vcstiled_{M}x{K}x{N}_base"
    gen_c(M, K, N, body, name)
    print(f"generated {name}.c")
    b = build(name)
    if not b:
        print(f"{name}: BUILD FAILED"); sys.exit(1)
    print(f"built {b}")
    cyc, done, out = run_vcs(b)
    print(out)
    print(f"{name}: MMCYCLES={cyc} DONE={done}")
