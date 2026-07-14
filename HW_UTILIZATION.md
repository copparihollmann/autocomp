# Whole-Radiance HW utilization — methodology + RTL numbers

Measured on the **RTL** (Verilator, tapeout-330 + trace-fix) via `scripts/muon/hw_utilization.py`.
Cycles are **compute-only**: counted from the **first kernel instruction's emission** to the last,
using ELF-symbol PC ranges (`kernel_body` + `mxgemm*`) ∩ the trace-db `inst` table. Harness `main`
(setup + DRAIN spin + tohost) and `verify_body` are excluded.

## The rule that makes this correct for *whole* Radiance
The two compute engines are **not** interchangeable and must not be conflated:

- **MX-Gemmini** does matmul on a 16×16 systolic array. **Its work is NOT in the instruction trace**
  (only Muon instructions are traced). So MX utilization is analytical: `essential_MACs / (PEAK_MX ×
  cycles)`. The Muon warps only **orchestrate** (issue MMIO + SIMT move-out), so Muon IPC on an MX
  kernel is *orchestration*, deliberately near-zero — **not** a compute-throughput number.
- **Muon SIMT** does the elementwise / GEMV / SIMT-GEMM compute on 2 cores × 16 lanes. Here Muon IPC
  and the SIMT FP roofline **are** the compute metrics.

## Verified peaks (RadianceSingleClusterConfig)
| Unit | Peak | Basis |
|---|---|---|
| MX-Gemmini fp8 | **256 MAC/cyc** | 16×16 systolic, 16-wide fp8 tiles (mod.rs:105) |
| MX-Gemmini fp6/fp4 | **512 MAC/cyc** | 32-wide sub-byte tiles (mod.rs:106) — *2× fp8; the old tool wrongly used 256* |
| Muon SIMT | **32 MAC/cyc = 64 flop/cyc** | 2 cores × 16 lanes, 1 FMA/lane/cyc |
| Muon issue | **~2 instr/core-cyc** (peak) | 2 cores, single-issue/core (RS-scoreboarded; superscalar width unconfirmed → issue-slot % is indicative) |

## RTL results (compute-only)
| Kernel | Compute engine | actual cycles | roofline util | Muon IPC (note) |
|---|---|---|---|---|
| MX fp8 128×128×512 | MX systolic | 95,829 | **34.2%** (MACs/256) | 0.062 — orchestration only |
| MX fp8 64×64×64 | MX systolic | 52,015 | **1.97%** — overhead-bound (tiny matmul) | 0.024 — orchestration |
| SIMT GEMV 64×128 | Muon lanes | 47,259 | **0.54%** FP (memory-bound) | 0.102 (5% issue-slots) |
| FUSED QKV(MX)+RoPE(SIMT) 64³ | both | *(RTL run pending)* | MX-phase + SIMT-phase split | — |

**Reading these honestly:**
- MX util rises with problem size (1.97% @ 64³ → 34% @ 128²×512): small matmuls are dominated by
  fixed config/scale-staging/DMA/move-out; the systolic array is only "busy" for a fraction. Real
  workloads want large fused tiles to amortize that overhead.
- Muon IPC on MX kernels (0.02–0.06) is *correct and expected* — the warps aren't computing, they're
  driving the accelerator. Do **not** read it as "the chip is 3% utilized."
- SIMT GEMV FP util is ~0 because it's memory-bandwidth-bound; the FP roofline isn't the limiter —
  IPC/BW is. The tool flags this.

## What "actual compute cycles" includes / excludes
- **Includes**: config, scale-staging, operand DMA wait, the systolic compute, and the SIMT C
  move-out — i.e. the real end-to-end kernel time (with inherent memory/accelerator stalls, which are
  real and must not be excluded).
- **Excludes**: harness setup before the first kernel instruction, the DRAIN write-drain spin, and the
  verify pass. The `last` cycle is the last Muon kernel instruction (the move-out), which is *after*
  the accelerator finishes (move-out reads C from SMEM behind a fence) — so the accelerator's
  completion is captured.
