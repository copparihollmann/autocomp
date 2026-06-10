# Autocomp kernel contract for radiance Muon

Every generated kernel must define exactly:

```cpp
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
);
```

The harness launches it with `mu_schedule(kernel_body, &kernel_args, NUM_WARPS)`:
one threadblock spanning MU_NUM_CORES=2 cores x NUM_WARPS=4 warps x 16 lanes = 128
threads. `tid_in_threadblock` is 0..127; warp id = tid / 16; lane id = tid % 16.

The kernel receives a `KernelArgs*` (problem-specific struct defined in the harness)
with `__global float*` buffers and shape fields. Use only these buffers.

Rules:
- Pure fp32 SIMT code. RV32IM + Zfinx. No mxgemmini/tensor instructions.
- For exponentials use the harness-provided `mu_exp(float)`; never expf/exp.
  Sigmoid: `1/(1 + mu_exp(-x))`.
- Barriers: `mu_barrier(id, num_warps)` synchronizes all warps of the threadblock
  (8 warps total when NUM_WARPS=4 since two cores).
- Shared memory (128 KiB per cluster) is addressed at address 0; use
  `__local` pointers or store/load shared intrinsics (see mu_intrinsics.h).
- Do NOT print from kernels. Verification is done by the harness (do not modify).
- Do not redefine main, mu_exp, KernelArgs.

Optimization levers: per-thread ILP (process multiple elements per lane),
register tiling, smem staging for reuse, fewer barriers, coalesced sequential
lane accesses, loop unrolling, occupancy.
