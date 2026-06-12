# Muon utilization — conservative & realistic (2026-06-11)

Tool: `scripts/muon/util_calc.py`. Designed to NEVER exaggerate:
- **Numerator** = essential problem FLOPs (matmul 2·M·N·K), not executed instructions → no credit for
  redundant/overhead work.
- **Denominator** = FULL hardware peak (1 FMA/lane every cycle). If the FP pipe can't sustain 1/cyc,
  true util is HIGHER → we under-claim.
- **Cycles** = TOTAL measured (launch + DRAM fill + compute + drain), not a cherry-picked compute region.

Peak (fp32): 2 cores × 16 lanes × 2 flop/FMA = **64 flop/cyc = 32 GFLOP/s @ 500 MHz**.

## Measured — TWO numbers (report both, they answer different questions)

**UTIL_total** includes the autocomp BENCHMARK verify loop (empty-kernel harness overhead ≈ **377k cyc**
for every matmul — data setup + launch + fp-compare-all-outputs). That loop is a benchmark artifact a
real deployment never runs, and it dominates these tiny-matmul totals → makes util look like ~1-5%.

**UTIL_net** = kernel-only (total − measured harness overhead = the kernel's own loads+compute+stores).
This is the fair "does the KERNEL utilize the machine" number.

| kernel | total cyc | overhead | net cyc | UTIL_total | **UTIL_net** |
|---|---|---|---|---|---|
| prob0 64³ naive  cyc@bw6 | 480,918 | 377,080 | 103,838 | 1.70% | 7.9% |
| prob0 64³ **smem** cyc@bw6 | 419,797 | 377,080 | 42,717 | 1.95% | **19.2%** |
| prob6 64×64×128 SMEM     | 487,016 | 377,136 | 109,880 | 3.36% | 14.9% |
| prob10 64×64×384         | 987,528 | 377,126 | 610,402 | 4.98% | 8.1% |

(RTL totals: prob0 naive 843,556 / smem 726,584 — RTL has its own host-launch overhead, not the
autocomp verify loop, so subtract an RTL empty-kernel run for RTL-net; cyclotron-net above uses the
matched autocomp overhead.)

**Answer to "do the kernels utilize well?":** not 1-5% — that was the verify loop. Kernel-only the
well-tiled SMEM 64³ reaches **~19% of SIMT FP peak**; naive ~8%. Still latency/occupancy-limited (64³
is small: net runs ~5× the compute roofline), but an order of magnitude better than the benchmark total.

## What it means for room-for-improvement

1. **Util is 1–5% — and that's honest.** These benchmark matmuls are tiny; the FP lanes sit idle most
   of the runtime.
2. **Low util ≠ big speedup available.** The roofline floor for 64³ is ~8,192 cyc; RTL runs 89× that.
   That gap is NOT compute- or memory-throughput-bound (both floors ≈ 8k) — it is **latency + fixed
   host-launch/DRAM-fill/drain overhead + low occupancy**. SMEM-tiling can't remove fixed overhead, so
   RTL confirms SMEM beats naive by only **1.16×** on 64³. Near the practical ceiling for this shape.
3. **The compass: util RISES with problem size** (1%→3.4%→4.9% as K 64→128→384; cyc/roofline 89×→30×→20×).
   Fixed overhead amortizes → on the larger/real shapes compute becomes the true bottleneck and
   tiling/reuse actually pays. **That is where the harvest headroom is**, not in micro-opting 64³.
4. **All shapes are compute-bound at the bw=6 roofline** (comp_floor ≥ mem_floor), increasingly so as K
   grows. So steady-state is compute-limited — but tiny problems never reach steady state.

Scope: utilization is measured against the **Muon SIMT FP peak only** (64 flop/cyc). The on-die
MX-Gemmini tensor core is intentionally **out of scope** — we do not use it and the denominator does
NOT include it. These are Muon-SIMT kernels and that is the machine we optimize.

Bottom line: report util on TOTAL cycles (1–5%, vs the SIMT FP peak). Use it as a ceiling gauge —
small problems are overhead-bound (little real room), large problems have real compute headroom
(where SMEM/reuse helps).
