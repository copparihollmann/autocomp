"""Generate autocomp MX-Gemmini matmul problems from the Rakanic gemmini-rocc-tests
fork's hand-written reference kernels + their SHIPPED, independently-verified gold.

Why this exists: our self-capture pipeline (matmul_gen) bakes the spike output AS the
gold — correct-by-construction but self-consistent. The fork ships data headers whose
`C_out_bf16` is an INDEPENDENT fp32-reference (golden_model.py), and we verified those
shipped headers pass on spike AND on the RadianceGemminiOnlyConfig RTL (VCS) bit-for-bit.
So they are a TRUE correctness oracle. (Regenerating golden_model.py for arbitrary shapes
does NOT reproduce hardware-matching gold — only the shipped headers are trustworthy,
which is exactly why the original design used self-capture.)

Transform: each fork kernel `bareMetalC/<k>.c` is split at the uniform boundary
`gemmini_flush(0);` ... `int errors = 0;` into
  - harness  = preamble + decls + REPEAT loop with // SUBSTITUTE markers + gold/full_is_equal
               (prob.py injects flush+read_cycles timing and the full_is_equal(C_hw, gold) check)
  - baseline = void solution(void){ <compute region> }   (mvin/loop_ws_spad/mx_read_smem)
The gold array is the shipped header's C_out_bf16 packed 4-bf16/uint64 (all fork shapes M==N).
"""

import pathlib
import re

from autocomp.common import HARNESSES_DIR, SOLS_DIR
from autocomp.backend.gemmini.mx_pipeline.matmul_gen import (
    _parse_c_out_bf16, GEMMINI_SW,
)

FORK_BAREMETAL = pathlib.Path("/scratch/agustin/projects/gemmini-rocc-tests-ref/bareMetalC")
# Our chipyard's include/ holds the shipped headers autocomp actually compiles against;
# we parse gold from there so it matches the built binary exactly.
CHIPYARD_INCLUDE = GEMMINI_SW / "include"


def _packed_gold(M, header_path: pathlib.Path):
    """Shipped C_out_bf16 -> rows of M/4 uint64 words (cols 4j..4j+3, col 4j low)."""
    grid = _parse_c_out_bf16(header_path)
    rows = []
    for i in range(M):
        words = []
        for j in range(len(grid[i]) // 4):
            w = ((grid[i][4*j+3] << 48) | (grid[i][4*j+2] << 32)
                 | (grid[i][4*j+1] << 16) | grid[i][4*j])
            words.append(f"0x{w:016x}ULL")
        rows.append("  { " + ", ".join(words) + " }")
    return ",\n".join(rows)


def _split_kernel(src_text: str):
    """Return (preamble, head_decls, compute) split at flush / `int errors`."""
    pre, _, after_main = src_text.partition("int main()")
    if not after_main:
        raise RuntimeError("no int main() in fork kernel")
    # main body: drop the leading "{ ... " up to first flush
    body = after_main.split("{", 1)[1]
    head, _, after_flush = body.partition("gemmini_flush(0);")
    if not after_flush:
        raise RuntimeError("no gemmini_flush(0); in fork kernel")
    compute = after_flush.split("int errors", 1)[0]
    # strip memset(C_hw...) from head (it moves into the repeat loop)
    head = "\n".join(l for l in head.splitlines() if "memset(C_hw" not in l)
    return pre, head, compute


_HARNESS_TMPL = """{preamble}
#define OUTPUT_MATRIX_NAME C_hw
#ifndef fence
#define fence() gemmini_fence()
#endif

static const uint64_t gold[MATMUL_M][OUT_COLS] = {{
{gold}
}};

int full_is_equal(uint64_t x[MATMUL_M][OUT_COLS], const uint64_t y[MATMUL_M][OUT_COLS]) {{
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
{head}
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {{
    memset(C_hw, 0, sizeof(C_hw));
    // SUBSTITUTE HERE
    // SUBSTITUTE END
  }}
  printf("Correct result\\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}}
"""

_SOL_TMPL = """// AUTO-GENERATED baseline from Rakanic fork kernel {kernel} ({fmt}).
void solution(void) {{
{compute}}}
"""


def gen_fork_problem(kernel: str, header: str, fmt: str, prob_type: str, prob_id: int = 0):
    """kernel: fork .c basename (no ext). header: shipped data header basename.
    Writes harness + baseline sol for autocomp."""
    src = (FORK_BAREMETAL / f"{kernel}.c").read_text()
    hpath = CHIPYARD_INCLUDE / header
    txt = hpath.read_text()
    M = int(re.search(r"MATMUL_M\s+(\d+)", txt).group(1))
    pre, head, compute = _split_kernel(src)
    gold = _packed_gold(M, hpath)

    hdir = HARNESSES_DIR / prob_type
    sdir = SOLS_DIR / prob_type
    hdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (hdir / f"test{prob_id}.c").write_text(
        _HARNESS_TMPL.format(preamble=pre.rstrip(), head=head.rstrip(), gold=gold))
    (sdir / f"sol{prob_id}_exo_baseline.c").write_text(
        _SOL_TMPL.format(kernel=kernel, fmt=fmt, compute=compute))
    return prob_type, prob_id, header


# (kernel basename, shipped header, fmt, prob_type)
FORK_PROBLEMS = [
    ("matmul_tiled_fp8_64x64",      "matmul_fp8_64x64.h",      "fp8:e4m3", "fork-mm-fp8-64"),
    ("matmul_tiled_fp8_128x128",    "matmul_fp8_128x128.h",    "fp8:e4m3", "fork-mm-fp8-128"),
    ("matmul_tiled_fp8_128x128x256","matmul_fp8_128x128x256.h","fp8:e4m3", "fork-mm-fp8-256"),
    ("matmul_tiled_fp6_128x128x512","matmul_fp6_128x128x512.h","fp6:e3m2", "fork-mm-fp6-512"),
    ("matmul_tiled_fp4_128x128x512","matmul_fp4_128x128x512.h","fp4:e2m1", "fork-mm-fp4-512"),
]


def gen_all():
    out = []
    for kernel, header, fmt, ptype in FORK_PROBLEMS:
        out.append(gen_fork_problem(kernel, header, fmt, ptype))
    return out


if __name__ == "__main__":
    for r in gen_all():
        print("generated", r)
