#!/usr/bin/env python3
"""Render all muon autocomp harnesses from one template."""

from pathlib import Path

ROOT = Path("/scratch/agustin/projects/autocomp/harnesses/muon")

TEMPLATE = """// Autocomp harness: {desc}
//
// The candidate kernel is substituted between the SUBSTITUTE markers and must define:
//   void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
//                    uint32_t threads_per_threadblock, uint32_t threadblock_id);

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

// Single-byte console writes; the simulator echoes every store. Keep verdict to one
// conditional store: longer per-char sequences can be dropped by the Muon compiler.
// Verdict is returned as main()'s exit code; the simulator prints "Error: <code>".
// (Device console printing is unreliable with this toolchain - do not print.)

#ifndef NUM_WARPS
#define NUM_WARPS 4 // occupancy per core; threadblock spans MU_NUM_CORES cores
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

// Relative FP tolerance: accommodates reassociated accumulation (tiling/split-K)
// and the RTL's ~1-ULP-approximate divider.
#ifndef TOLERANCE_REL
#define TOLERANCE_REL {tol}f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS {tol_abs}f
#endif

#include "data"

{extra}

struct KernelArgs {{
{args}
}};

// SUBSTITUTE HERE
// SUBSTITUTE END

static KernelArgs kernel_args;

static inline float fabsf_(float x) {{ return x < 0.0f ? -x : x; }}
static inline bool close_enough(float c, float g) {{
  return fabsf_(c - g) <= TOLERANCE_REL * fabsf_(g) + TOLERANCE_ABS;
}}
static inline uint32_t hart_id() {{
  uint32_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id)::"memory");
  return id;
}}

int main() {{
  kernel_args = {{{init}}};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); // both cores fully done before verification

  // Verify + report once: single lane; non-zero harts spin (sim ends when hart 0 exits).
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) {{
    for (;;) {{}}
  }}

  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++) {{
    if (!close_enough({out}[i], gold_raw[i])) errors++;
  }}
  // Verdict via tohost: the simulator's ECALL reports rs1, so encode ECALL with
  // rs1 = code. 0 = pass, else (errors<<1)|1 -> prints "case=<errors>".
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}}
"""

MU_EXP = """// fp32 exp shared by all candidates; the golden output is computed with this same
// polynomial (see gen_data.py), so kernels must use mu_exp().
static inline float mu_exp(float x) {
  const float LOG2E = 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;
  float t = x * LOG2E;
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * LN2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
  union { uint32_t i; float f; } s;
  s.i = (uint32_t)((k + 127) << 23);
  return p * s.f;
}"""

MU_RSQRT = """// fp32 reciprocal-sqrt (no libm/sqrtf libcall on this toolchain): fast-inverse-sqrt seed
// + 3 Newton iters (~1e-6 rel). layer_norm golden uses exact 1/sqrt; tolerance covers the residual.
static inline float mu_rsqrt(float x) {
  union { float f; uint32_t i; } u; u.f = x;
  u.i = 0x5f3759dfu - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}"""

PROBLEMS = {
    0: dict(
        desc="fp32 matmul C[M,N] = A[M,K] @ B[K,N]",
        tol="1.0e-4",
        tol_abs="1.0e-5",
        extra="",
        args="  __global float *A, *B, *C;\n  uint32_t M, N, K;",
        init="A_raw, B_raw, C_raw, M, N, K",
        count="M * N",
        out="C_raw",
    ),
    1: dict(
        desc="fp32 conv2d patch-embed OUT[OC,OH,OW] = IN[C,H,W] * W[OC,C,16,16], stride 16",
        tol="1.0e-4",
        tol_abs="1.0e-5",
        extra="",
        args=(
            "  __global float *in, *w, *out;\n"
            "  uint32_t C, H, W, OC, KH, KW, stride, OH, OW;"
        ),
        init="in_raw, w_raw, out_raw, C, H, W, OC, KH, KW, STRIDE, OH, OW",
        count="OC * OH * OW",
        out="out_raw",
    ),
    2: dict(
        desc="single-head attention O = softmax(Q K^T / sqrt(d)) V (fp32, seq 64, d 64)",
        tol="2.0e-4",
        tol_abs="1.0e-5",
        extra="__global float scratch_raw[SEQ * SEQ];\n\n" + MU_EXP,
        args="  __global float *Q, *K, *V, *O, *scratch;\n  uint32_t seq, d;",
        init="q_raw, k_raw, v_raw, o_raw, scratch_raw, SEQ, HEAD_DIM",
        count="SEQ * HEAD_DIM",
        out="o_raw",
    ),
    3: dict(
        desc="single-head attention O = softmax(Q K^T / sqrt(d)) V (fp32, seq 96, d 64)",
        tol="2.0e-4",
        tol_abs="1.0e-5",
        extra="__global float scratch_raw[SEQ * SEQ];\n\n" + MU_EXP,
        args="  __global float *Q, *K, *V, *O, *scratch;\n  uint32_t seq, d;",
        init="q_raw, k_raw, v_raw, o_raw, scratch_raw, SEQ, HEAD_DIM",
        count="SEQ * HEAD_DIM",
        out="o_raw",
    ),
    4: dict(
        desc="SwiGLU activation C[m,n] = silu(A) * B (fp32)",
        tol="1.0e-3",
        tol_abs="2.0e-4",
        extra=MU_EXP,
        args="  __global float *A, *B, *C;\n  uint32_t M, N;",
        init="A_raw, B_raw, C_raw, M, N",
        count="M * N",
        out="C_raw",
    ),
    5: dict(
        desc="row softmax OUT[r,:] = softmax(IN[r,:]) (fp32)",
        tol="2.0e-4",
        tol_abs="1.0e-5",
        extra=MU_EXP,
        args="  __global float *in, *out;\n  uint32_t rows, cols;",
        init="x_raw, out_raw, ROWS, COLS",
        count="ROWS * COLS",
        out="out_raw",
    ),
    8: dict(
        desc="layer_norm OUT[r,:] = (IN[r,:]-mean)/sqrt(var+eps)*gamma+beta (fp32, 64x768)",
        tol="1.0e-3",
        tol_abs="2.0e-4",
        extra=MU_RSQRT,  # mu_rsqrt (no sqrtf libcall)
        args="  __global float *in, *gamma, *beta, *out;\n  uint32_t rows, cols;",
        init="x_raw, gamma_raw, beta_raw, out_raw, ROWS, COLS",
        count="ROWS * COLS",
        out="out_raw",
    ),
    9: dict(
        desc="GELU C[m,n] = x * sigmoid(1.702*x) (fp32 sigmoid-approx, 64x512)",
        tol="1.0e-3",
        tol_abs="2.0e-4",
        extra=MU_EXP,
        args="  __global float *A, *C;\n  uint32_t M, N;",
        init="A_raw, C_raw, M, N",
        count="M * N",
        out="C_raw",
    ),
    10: dict(
        desc="large-K matmul C[M,N]=A[M,K]@B[K,N] (fp32, 64x64x768; K>SMEM forces K-streaming)",
        tol="1.0e-3",  # K=768 fma-vs-numpy accumulation drift > 1e-4
        tol_abs="1.0e-4",
        extra="",
        args="  __global float *A, *B, *C;\n  uint32_t M, N, K;",
        init="A_raw, B_raw, C_raw, M, N, K",
        count="M * N",
        out="C_raw",
    ),
    11: dict(
        desc="single-head attention O=softmax(QK^T/sqrt(d))V (fp32, seq 96, head_dim 72 non-pow2)",
        tol="2.0e-4",
        tol_abs="1.0e-5",
        extra="__global float scratch_raw[SEQ * SEQ];\n\n" + MU_EXP,
        args="  __global float *Q, *K, *V, *O, *scratch;\n  uint32_t seq, d;",
        init="q_raw, k_raw, v_raw, o_raw, scratch_raw, SEQ, HEAD_DIM",
        count="SEQ * HEAD_DIM",
        out="o_raw",
    ),
    13: dict(
        desc="tall-skinny matmul C[M,N]=A[M,K]@B[K,N] (fp32, M=8 K=256 N=768; underutilization)",
        tol="1.0e-4",
        tol_abs="1.0e-5",
        extra="",
        args="  __global float *A, *B, *C;\n  uint32_t M, N, K;",
        init="A_raw, B_raw, C_raw, M, N, K",
        count="M * N",
        out="C_raw",
    ),
}

for n, p in PROBLEMS.items():
    rendered = TEMPLATE.format(**p)
    path = ROOT / f"test{n}" / f"test{n}.cpp"
    path.write_text(rendered)
    # also write the flat test{n}.c — Prob.tests (autocomp/search/prob.py) globs this for the
    # candidate substitution; muon_eval copies data/Makefile from the test{n}/ subdir.
    flat = ROOT / f"test{n}.c"
    flat.write_text(rendered)
    print(f"wrote {path} + {flat}")
