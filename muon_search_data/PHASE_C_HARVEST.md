# Phase C — bw=6 harvest results (2026-06-11)

Parallel hybrid search (plan=gemini-3.5-flash, code=qwen3-coder-480b) at the RTL-validated bw=6 oracle,
on the high-headroom / now-RTL-faithful targets. Spend this run: ~$22 ($153.3 → $175.3). Under $188 guard.

## Honest results — like-for-like TOTAL cycles (run_problem.sh, the metric we report)

| prob | shape | baseline | this-run best | **total speedup** | note |
|---|---|---|---|---|---|
| 1 conv patch-embed | — | 102,215 | 96,861 | **1.055×** | prior saved best `muon_1_beam_iters8` is better at **1.29×** (bw6); this run did NOT beat it |
| 6 matmul | 64×64×128 | 487,016 | 487,016 | **1.00×** | best == baseline; nothing found |
| 10 matmul | 64×64×384 | 998,938 | 987,528 | **1.01×** | noise-level |

**The harvest found no new wins.** That is itself the finding, not a failure:

- Where the baseline is **naive**, the bw=6 memory win is real and RTL-confirmed: prob0 naive→SMEM =
  **1.16× total on RTL** (cyclotron@bw6 matches within 1.3%; see `rtl_confirmation/`).
- Where the baseline is **already SMEM-staged** (prob6, prob10) the win is already baked in, so there is
  nothing left to harvest at the SIMT level.
- conv (prob1) headroom was already captured by the earlier search (`muon_1_beam_iters8`, 1.29×); the
  fresh run only reached 1.055×.

This is consistent with `UTILIZATION.md`: SMEM kernels already do what's achievable at the SIMT level
(util 3–5%, vs the Muon SIMT FP peak); the remaining gap is fixed launch/fill/drain overhead + latency.
Further LLM search across already-tiled baselines is low-yield.

## Metric gotcha (important)
The search's "Best candidate score: N" = **NET** cycles (`muon_eval.py:132`,
`latency − harness_overhead`), and the value logged at `search.py:437 __init__` is the **seed** score
(before iteration 1). `run_problem.sh` reports **TOTAL**. Never compare a search score against a
run_problem number — that mixes net-vs-total and fabricates fake speedups (it briefly showed a bogus
"4.43×" for prob6). Always compare baseline-vs-best BOTH through run_problem (total).

## Recommendation
Run the bw=6 harvest only from NAIVE baselines (or on larger/fused shapes); don't re-optimize SMEM
seeds. Remaining SIMT-level headroom is in larger/fused problems and cutting fixed launch/drain
overhead. (The MX-Gemmini tensor core is out of scope — we do not use it.)
