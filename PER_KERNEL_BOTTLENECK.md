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

## Part D — one routed optimization carried through + RTL-gated

**Diagnosis under test:** the MX kernels are FIXED-OVERHEAD-bound (config + operand-DMA + SIMT
move-out dominate; the systolic array itself is ~90% utilized *in its compute window*). Routed lever
(hand, not autocomp): **amortize the fixed overhead over more compute** (larger tiles / more K per
launch) and **shrink the move-out**.

**RTL A/B (Verilator, tapeout-330), fp8 128×128×K, TILE_K=128, sweeping K:**

| K | GK | compute-window util | whole-kernel util | move-out % of kernel |
|--:|--:|--:|--:|--:|
| 128 | 1 | 88.1% | 20.4% | 48.3% |
| 256 | 2 | 88.0% | 20.7% | 32.1% |
| 512 | 4 | 91.9% | **34.2%** | 26.7% |

**Result — diagnosis CONFIRMED:** compute-window util is flat (~88–92%: the array is already
saturated when it computes), while whole-kernel util rises **1.68× (20.4%→34.2%)** purely from
amortizing the fixed overhead, and the SIMT move-out fraction falls 48%→27%. This is the signature of
a FIXED-OVERHEAD bottleneck (improvement comes from *restructuring/amortization*, not from touching
the compute) — and it is exactly what the routing predicted, RTL-measured.

**Projected next step (same lever):** even at K=512 the move-out is still 27% + config/DMA. Moving the
epilogue off the SIMT path (DMA move-out, `SIMT_GMEM_MOVE_OUT=false`) or overlapping it with the next
tile's compute would reclaim most of that 27% → whole-kernel util toward ~47%. (Not cycle-A/B'd here:
the DMA path is untraced and inserts a bogus SIMT copy for trace visibility, so a clean trace-db cycle
comparison needs the drain-independent net-cycle oracle, not the phased tool.)

**Negative control (routing the other way):** the LATENCY-bound SIMT kernels run at IPC 0.02–0.31
(2–15% issue) and util <2% — ~90%+ of cycles are memory stalls. A compute-side / instruction-count
change (what autocomp optimizes, what cyclotron can rank) cannot move a kernel that is not
instruction-bound, so these are correctly routed to hand + RTL-gate (overlap/prefetch/double-buffer),
NOT to an autocomp $-search. The data (low IPC, below every compute ceiling) is the evidence.

## SIMT-core compute utilization (the MX analog for the Muon lanes)

MX's 92–96% is the *systolic array's* MAC utilization. For the SIMT cores the compute metric is
`essential_MACs / (32 MAC/cyc × cyc)` (2 cores × 16 lanes × 1 `fmadd.s`/lane/cyc; the 8 warps are
latency-hiding occupancy, not extra FLOPs). Measured on RTL (fmadd.s trace-execution counts):

| SIMT kernel (RTL) | cyc | IPC (issue%) | FMA-fraction | non-FMA per FMA | **SIMT compute util** |
|---|--:|--:|--:|--:|--:|
| decode GEMV (M=1, 64×128) | 47,259 | 0.102 (5%) | 10.7% | ~8.4 | 0.54% |
| **GEMM 64³ (naive, compute-bound)** | 297,726 | **0.737 (37%)** | **7.5%** | ~12.4 | **2.75%** |
| elementwise (RoPE/RMSNorm/softmax) | — | 0.02–0.20 | — | — | 0.14–1.6% |

**The key result:** even the compute-bound matmul, which issues at **37% of peak issue** (10× the
memory-bound kernels — the core is genuinely busy), reaches only **2.75% of SIMT FP peak**, because a
scalar SIMT ISA spends ~92% of issued instructions on non-FMA work (`lw.global` loads, address
arithmetic, loop control, `vx_split_n/vx_pred_n/vx_join` divergence). That is the architectural
ceiling.

**MX vs SIMT effective matmul throughput:** MX ≈ 256 × 0.92 ≈ **235 MAC/cyc**; SIMT ≈ 32 × 0.0275 ≈
**0.9 MAC/cyc** — a **~260× gap**. Quantitatively why GEMM/attention-matmul route to MX and the SIMT
cores own norms/activation/RoPE/residual/decode-GEMV.

**RTL notes:** a SMEM-tiled SIMT GEMM at 64³ trips the l0d backpressure assertion (same unbuffered-l0d
hazard); the naive version completes but is memory-bound (IPC 0.74 is issue-limited by the 12× non-FMA
overhead, not compute). Its RTL verify also fails (tohost=1417) — another drain-invariant
cyclotron-vs-RTL mismatch; cycles/util remain valid (data-independent control flow).

### SIMT compute — rigorous windup/steady/winddown + realistic roofline (correction)

Concern raised: is the SIMT number measured over the *steady* compute window (excluding windup/
winddown), like the MX P2 K-loop is? Checked on GEMM 64³:

| segment | cyc | % of kernel |
|---|--:|--:|
| windup (kernel start → 1st FMA) | 2,657 | 0.9% |
| **STEADY (1st → last FMA)** | 294,991 | **99.1%** |
| winddown (last FMA → end) | 78 | 0.0% |

Unlike MX (where config+DMA+move-out are ~66% of the kernel), a compute-heavy SIMT matmul is
**99% inner loop** — so windup/winddown are negligible and the steady compute util (**2.78%**) equals
the whole-kernel util (2.75%). The number was steady-state; now proven.

**Ideal roofline:** 2.78% of FP peak (64 flop/cyc).

**Realistic roofline A — instruction-mix / issue ceiling (the binding one):** with a 7.5% FMA-fraction,
the achievable ceiling even at *peak issue* is `2 instr/cyc × 0.075 × 16 lanes = 2.39 MAC/cyc = 7.5% of
FP peak`. Steady achieves 0.89 MAC/cyc = **37% of this realistic ceiling** — and 37% == the issue
utilization, i.e. the remaining gap to the mix-ceiling is pure memory-latency stall (63% of cycles
issue nothing). So the SIMT matmul is **memory-latency-bound underneath a 7.5%-of-peak instruction-mix
ceiling**. Tiling to hide the latency (approach the 7.5% ceiling) trips the l0d assertion on this SoC.

**Realistic roofline B — memory-BW:** best-case full-reuse AI = 10.7 flop/B < DRAM ridge 16 → cold-DRAM
ceiling 43 flop/cyc; the naive kernel re-reads B so real AI is lower. BW is not the *binding* limit
here (issue/mix + latency is), but it confirms SIMT matmul is nowhere near compute-bound.

**Bottom line unchanged, now rigorous:** steady SIMT matmul compute util = **2.78% of ideal FP peak**,
**37% of its realistic (instruction-mix) ceiling of 7.5%**. vs MX steady 92–96% of a peak 8× higher.

## MX ↔ SIMT interleaving (concurrent-engine utilization)

Two kinds of interleaving, measured separately (the whole-Radiance block's `overlap` = Σ engine-active
/ union: 1.0× = serialized, >1× = concurrent):

**1. Orchestration/prefetch overlap — happens, but nearly free.** During the anchor128 MX compute
window (P2 K-loop, 35,645 cyc) the SIMT cores issue only 3,321 instrs → **IPC 0.093 (4.7% of peak
issue)**: MMIO to drive the systolic array + `copy_gmem_to_smem_async` prefetch of the next tile's
operands, overlapping the array's compute (this is the latency-hiding that keeps MX at 92%). So the
systolic array and the async-DMA/orchestration path DO run concurrently.

**2. Compute–compute overlap — does NOT happen anywhere.** During MX matmul the SIMT compute engine is
**~95% idle** (only that 4.7% orchestration). No measured kernel runs SIMT FMAs concurrently with MX
MACs — `overlap = 1.00×` on every kernel, including the "fused" test33 (its RoPE runs strictly *after*
the matmul completes, serialized).

**The headroom this quantifies:** ~95% of the SIMT cores' issue capacity sits idle for the entire MX
matmul. Scheduling independent SIMT work there (the next op's RMSNorm/RoPE/residual/dequant, or a
second attention head) would be near-free — this is the concrete, measured basis for the "~2× on
fused layers" lever. The tool would show it as `overlap > 1×` and a higher combined throughput
efficiency; today that number is 1.00× everywhere.

**How the framework scores a genuinely interleaved kernel** (when one is built): per-engine active
fraction over the shared window, `any-engine busy %`, `overlap = Σ active / union` (>1× = real
concurrency), and combined throughput efficiency `(MX_flop + SIMT_flop) / ((MX_peak+SIMT_peak)×cyc)` —
so concurrent MX+SIMT work is credited to both engines in the same cycles.

## Tensor core (= MX-Gemmini) — ideal vs realistic utilization

Confirmed via RTL: the "tensor core" is **MX-Gemmini itself** (the only matrix engine, `GemminiTileLike`,
16×16 systolic; no separate Muon wmma/mma unit — `CyclotronTile.scala` has none, and `sgemm_tcore`
drives Gemmini). So profiling the tensor core = the MX profiling, with both rooflines made explicit:

**IDEAL roofline (% of peak MAC):** compute-window **91.9%** (K=512), 88% (K=128); whole-kernel 34%→2%
(overhead-diluted). Peak 256 MAC/cyc fp8, 512 fp6/fp4.

**REALISTIC roofline (compute-window):** the systolic array's operand feed is **~3.7 B/cyc from SMEM**
with reuse (raw wavefront 16 A + 16 B fp8 = 32 B/cyc), both ≤ SMEM capacity (64 B/cyc, 32 contended).
So the array is **NOT operand-feed-bound → COMPUTE-bound → realistic ≈ ideal ≈ 92%**. The remaining
8% to peak is **systolic fill/drain pipeline latency**, not memory.

**Key contrast with the SIMT cores:** the tensor core has **no per-MAC instruction overhead** (one
systolic array streams operands; there are no per-element loads/address/control/divergence
instructions). So its realistic ceiling ≈ its ideal peak, and it actually reaches ~92% — whereas the
SIMT lanes are capped at a ~7.5%-of-peak instruction-mix ceiling. That is the fundamental reason
matmul lives on the tensor core.

**Feed-ceiling caveat (fp6/fp4):** sub-byte formats run the 32-wide path at 512 MAC/cyc → raw wavefront
feed ≈ 64 B/cyc = the *uncontended* SMEM limit (2× the contended 32 B/cyc). So fp6/fp4 sit right at the
SMEM-feed ceiling; if both cores contend for SMEM the sub-byte tensor-core path can become
feed-bound. (The measured fp6/fp4 kernels are tiny/overhead-bound, so this ceiling isn't hit there —
but at scale it's the realistic limiter for sub-byte matmul, worth an RTL check.)
