# Per-kernel bottleneck analysis + hand-vs-autocomp routing (Parts C+D)

RTL (Verilator, tapeout-330+trace-fix), single-pass, DRAIN=2000. Compute-window = the phase
the engine actually computes (MX P2 K-loop / SIMT kernel_body); U_kernel = end-to-end incl.
config/DMA/move-out. MX warm = per-tile steady (cold tile excluded). Bottleneck via the
decision tree; route per the cyclotron-blind-spot table (compute/issue-count -> autocomp can
see it; memory-hierarchy/latency/overlap/overhead/register -> hand + RTL arbiter).

| kernel | eng | U_compute | U_kernel | MX warm | IPC (issue%) | DRAM-BW | idle% | overlap | spills | **binding limit** | route |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| SwiGLU 64x512 | simt | TRUNC† | TRUNC† | — | 0.035 (2%) | — | 0% | 1.00x | 0 | **MEMORY-BOUND (L0d backpressure assertion at full shape; unbuffered per-tile l0d)** | Hand + RTL-gate |
| Softmax 64x67 | simt | 0.82% | 0.82% | — | 0.203 (10%) | 21% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| RoPE 512x128 | simt | 0.27% | 0.27% | — | 0.023 (1%) | 12% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| RMSNorm 64x512 | simt | 0.14% | 0.14% | — | 0.023 (1%) | 6% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| ResAdd 128x512 | simt | TRUNC† | TRUNC† | — | 0.048 (2%) | — | 0% | 1.00x | 0 | **MEMORY-BOUND (L0d backpressure assertion at full shape; unbuffered per-tile l0d)** | Hand + RTL-gate |
| GEMV 512x512 | simt | TRUNC† | TRUNC† | — | 0.071 (4%) | — | 0% | 1.00x | 0 | **MEMORY-BOUND (L0d backpressure assertion at full shape; unbuffered per-tile l0d)** | Hand + RTL-gate |
| decode QKt S128 D64 | simt | 0.67% | 0.67% | — | 0.128 (6%) | 22% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| GEMV-softmax S128 | simt | 0.03% | 0.03% | — | 0.148 (7%) | 1% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| decode PV S128 D64 | simt | 1.64% | 1.64% | — | 0.308 (15%) | 54% | 0% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| FUSED QKV+RoPE 64^3 | fused | 14.11% | 1.66% | — | 0.166 (8%) | — | 47% | 1.00x | 6 | **FIXED-OVERHEAD-bound** | Hand |
| MxGEMM fp4 64x64x128 | mx | 7.23% | 1.91% | — | — | 11% | 60% | 1.00x | 0 | **FIXED-OVERHEAD-bound** | Hand |
| MxGEMM fp6 128^3 | mx | 14.13% | 14.13% | — | — | 57% | 23% | 1.00x | 0 | **LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)** | Hand + RTL-gate |
| Requant fp8 64x64x128 | mx | 22.39% | 4.08% | — | — | 10% | 75% | 1.00x | 0 | **FIXED-OVERHEAD-bound** | Hand |

† TRUNC = trace truncated by the L0d backpressure assertion at full tinyllama shape (unbuffered per-tile l0d, makeLandingPads=false; cluster cache has it =true). Whole-kernel util is therefore not measurable at full shape; the captured IPC (low) + the assertion itself confirm MEMORY-BOUND. Same bottleneck class as the clean elementwise/GEMV kernels.

## Routing rationale

- **SwiGLU 64x512** → Hand + RTL-gate: saturates l0d response path; reduce in-flight traffic / coalesce; whether the tapeout l0d should carry landing pads is a DUT call for the team
- **Softmax 64x67** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **RoPE 512x128** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **RMSNorm 64x512** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **ResAdd 128x512** → Hand + RTL-gate: saturates l0d response path; reduce in-flight traffic / coalesce; whether the tapeout l0d should carry landing pads is a DUT call for the team
- **GEMV 512x512** → Hand + RTL-gate: saturates l0d response path; reduce in-flight traffic / coalesce; whether the tapeout l0d should carry landing pads is a DUT call for the team
- **decode QKt S128 D64** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **GEMV-softmax S128** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **decode PV S128 D64** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **FUSED QKV+RoPE 64^3** → Hand: hoist config, DMA-vs-SIMT move-out, amortize/fuse; restructuring invisible to issue-count
- **MxGEMM fp4 64x64x128** → Hand: hoist config, DMA-vs-SIMT move-out, amortize/fuse; restructuring invisible to issue-count
- **MxGEMM fp6 128^3** → Hand + RTL-gate: canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search
- **Requant fp8 64x64x128** → Hand: hoist config, DMA-vs-SIMT move-out, amortize/fuse; restructuring invisible to issue-count
