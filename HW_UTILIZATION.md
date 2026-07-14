# Whole-Radiance HW utilization — methodology + RTL numbers

Measured on the **RTL** (Verilator, tapeout-330 + trace-fix). The trustworthy tool is
**`scripts/muon/hw_util_phased.py`** (phase-aware, multi-scope, both engines + whole-Radiance;
supersedes the whole-span `hw_utilization.py`). Cycles come from the trace-db `inst` table over
ELF-symbol PC ranges; harness `main` (setup + DRAIN spin + tohost) and `verify_body` are excluded.

## Utilization is reported at MULTIPLE SCOPES × BOTH ENGINES + whole-Radiance
A single "utilization %" is meaningless without saying *of what window* and *for which engine*. The
tool fills only the cells that are physically real for the kernel (`—` otherwise):

**Scopes** (which window the useful work is divided by):
- **inner / steady** — warm per-K-tile, cold + edge tiles excluded (MX, needs GK≥4). The *reachable
  ceiling*: what the engine does when fed and warm.
- **compute-window** — the phase where the engine actually computes (MX = `main_matmul_k_loop` P2;
  SIMT = `kernel_body`). Excludes config/DMA/move-out. "How good is the compute itself."
- **whole-kernel** — first→last kernel instr, *including* config + operand-DMA + move-out.
  End-to-end; overhead-diluted; this is what a caller actually pays per launch.
- **fused-layer** — one kernel that drives *both* engines (test33): whole-kernel window, reported
  per engine. For a single op, layer == kernel.
- **whole-layer** (`--layer manifest.json`) — several ops that make up a transformer sub-layer, run
  serially on the one cluster: layer cycles = Σ kernel cycles; per-engine util over the cycles where
  that engine is the compute engine; + idle% (per-launch overhead) and combined throughput eff.

**Engines** (never conflated — the trace records only Muon instructions):
- **MX-Gemmini** systolic: `U = essential_MACs / (PEAK_MX[fmt] × window_cyc)` (analytical; MX work is
  NOT traced). Muon IPC on an MX kernel is *orchestration only*, not throughput.
- **Muon-SIMT**: `U = essential_MACs/(32·win)` or `FLOPs/(64·win)`, + IPC / issue-slot util.
- **Whole-Radiance** (both engines, whole-kernel window): per-engine **active fraction**, **any-engine
  busy %** (→ idle % = pure launch/config/barrier), **engine overlap** (`Σactive / union`; **1.0× =
  serialized**, >1.05× = genuine concurrency), and a **throughput efficiency** vs the summed peak of
  both engines. That last one is a *mixed-precision op-throughput* ratio (MX fp8 ops + SIMT fp32 ops);
  a low value = idle silicon (the other engine sitting unused), **not** numeric error.

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

## Realistic roofline (besides the ideal/compute roofline)
The ideal roofline (% of peak compute) makes memory-bound kernels look terrible when they're near
their *achievable* limit. The realistic roofline adds two more ceilings:
- **Memory-BW ceiling (diagonal)**: `min(compute_peak, BW × arithmetic_intensity)`. BW = 4 B/cyc
  (cold DRAM) or 64 B/cyc (SMEM-resident), from the RTL-fitted timing model.
- **Overhead/latency floor**: a kernel below BOTH the compute and the memory-BW ceilings is
  latency/overhead-bound (not saturating either resource).

Measured (RTL, compute-only):
| Kernel | AI (flop/B) | ideal (compute) | realistic (cold-DRAM) | DRAM-BW used | binding limit |
|---|---|---|---|---|---|
| MX fp8 128×128×512 | 102 | 34.2% | 42.7% | 42.7% of 4 B/cyc | **overhead/latency-bound** |
| SIMT GEMV 64×128 | 0.49 | 0.54% | 17.7% | 17.7% of 4 B/cyc | **overhead/latency-bound** |

**Diagnosis:** both kernels are below both ceilings → **latency/overhead-bound**, not compute- or
BW-bound. GEMV's real ceiling is ~2 flop/cyc (memory-bound, AI=0.49 « DRAM ridge 16); its "0.54% of
compute peak" is against an unreachable peak. The lever is hiding latency / cutting fixed overhead
(fusion, larger tiles, more in-flight memory requests, MX/SIMT overlap) — NOT adding FLOPs or BW.

Usage: `hw_utilization.py <elf> <trace> --engine mx|simt|fused --macs M*N*K --bytes <DRAM bytes> --fmt fp8|fp6|fp4`
