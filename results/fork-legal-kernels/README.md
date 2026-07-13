# RTL-legal optimized MX matmul kernels (fork baselines)

These are **RTL-legal** optimizations of the Rakanic fork MX matmul baselines, found after
making the spike model faithful to the hardware mvin/mvout field widths (see
`MX_AUTOCOMP_STATUS.md` §4c/§4d). Unlike the earlier autocomp "winners" (which used `cols=64`
batched mvin and **fatal on VCS**), every kernel here passes on **both spike and VCS RTL**.

## The transformation
Single-port **batched ("block") mvin** of A and B at **batch=2** → `cols = 2*DIM = 32`, which fits
the hardware's **6-bit mvin cols field** (max 63; `cols=64` overflows → 0 → `$fatal`). batch=2 also
divides the tile counts (8/16/4) cleanly, beating batch=3 (which leaves remainders). For fp6/fp4 the
baselines also had a `gemmini_fence()` after **every** mvin; those are removed (one fence before the
compute suffices), which is most of the fp6/fp4 win.

Standard B scratchpad layout `b_base + (k*tiles_J + j)*DIM` is used throughout — the fp8-128
autocomp winner's *transposed* layout `(j*tiles_K+k)` is correct on spike but **wrong on VCS** (a
separate spike↔RTL `loop_ws_spad` B-addressing gap), so we use the VCS-validated layout.

## Results (spike read_cycles; VCS = correctness on RadianceGemminiOnlyConfig RTL)

| kernel              | baseline | optimized | speedup | VCS RTL        |
|---------------------|---------:|----------:|--------:|----------------|
| fp8 64×64×64        |      294 |       146 |  2.01×  | PASS ✓         |
| fp8 128×128×128     |      853 |       525 |  1.62×  | PASS ✓         |
| fp8 128×128×256     |     1607 |       902 |  1.78×  | PASS ✓         |
| fp6 128×128×512     |     1939 |       721 |  2.69×  | PASS ✓         |
| fp4 128×128×512     |     1658 |       695 |  2.39×  | PASS ✓         |

All speedups are honest: real Gemmini ops, independent (shipped-header golden_model) gold, and
RTL-validated. Contrast the prior spike-only "winners" (fp8 3.6–3.9×) that were RTL-illegal.

## Reproduce
Kernels live in `gemmini-rocc-tests-ref/bareMetalC/*_LEGAL.c` (registered in that Makefile).
- spike: `build_spike.sh` style build (RUNNER=spike), `spike --extension=gemmini build_spike/bareMetalC/<t>-baremetal`
- VCS:   build with `RUNNER=vcs` (no `-DSPIKE_SIM`) → `build/bareMetalC/<t>-baremetal`, then
  `make -C sims/vcs run-binary CONFIG=RadianceGemminiOnlyConfig LOADMEM=1 BINARY=<abs>`
- cycle measurement: `autocomp/measure_*.py` (runs the body through the fork prob's read_cycles harness).
