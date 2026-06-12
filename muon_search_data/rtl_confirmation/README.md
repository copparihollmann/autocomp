# RTL confirmation of the bw=6 oracle fix (2026-06-11)

The session-long blocker was that cyclotron (the autocomp search oracle) under-ranked
memory-hierarchy wins. The fix: model finite global-memory bandwidth in
`chipyard/generators/radiance/cyclotron/config/timing/gmem.toml`,
`[gmem.levels.data] bytes_per_cycle 64 -> 6` (data level only; tag/mshr stay 64). Commit aed1f7e.

This dir holds the evidence that bw=6 is faithful to real RTL.

## Phase A — RTL confirms bw=6 (matmul naive vs SMEM)

Built a trace-free `RadianceSingleClusterFastConfig` (exact mirror of RadianceSingleClusterConfig
but `trace=false, profiler=false`) so RTL gates run WITHOUT the cyclotron trace-sqlite writes that
made the trace=true sim take ~90 min. Trace-free: spam=0, $finish reached, EXIT=0, ~25-28 min/kernel.

| metric | naive (sol0_baseline) | smem (sol0_smem_manual) | total ratio |
|---|---|---|---|
| RTL (FastConfig, $finish_ps/2000 @500MHz) | 843,556 cyc | 726,584 cyc | **1.161×** |
| cyclotron @ bw=6 (same .cpp) | 480,918 | 419,797 | **1.146×** |
| cyclotron @ bw=64 (old oracle) | 390,953 | 381,807 | 1.02× (under-predicts) |

**bw=6 reproduces the RTL total-cycle ratio within 1.3%; bw=64 under-predicted it.** Oracle fix
RTL-validated. (The earlier "2.42×" was a net/overhead-subtracted compute-region ratio; end-to-end
TOTAL for this shape is ~1.16×, the rest fixed host-launch/DRAM-fill overhead. Report TOTAL cycles.)

## Phase B — per-op faithfulness (cyclotron, bw6 vs bw64)

See `faithfulness_bw.csv` / `faithfulness_bw.sh`:
- conv (prob1): 1.29× @bw6 vs 1.34× @bw64 — compute win SURVIVES (faithful)
- layer_norm (prob8): ~1.0× both — no win to mask
- gelu (prob9): 1.00× both — best == baseline (no opt found)
- softmax (prob5): best produces no cycles at BOTH bws — kernel broken, bw-independent
- attn (calib): smem-attn TIMEOUT at BOTH bw6 and bw64 — pre-existing pathology, NOT a bw6 artifact

No ranking flips between bw6 and bw64 → bw=6 doesn't spuriously mask compute wins. Faithful.

## How to reproduce
- Build RTL: `cd sims/vcs && bash -lc 'source ../../env.sh && make CONFIG=RadianceSingleClusterFastConfig default'`
  (MUST source env.sh — system JDK-21 breaks the sbt metabuild; conda env pins JDK-20).
- Gate: `simv-...-RadianceSingleClusterFastConfig +permissive +max-cycles=10000000 +loadmem=K.soc.elf +permissive-off K.soc.elf` (no +trace-db). cycles = $finish_ps / 2000.
- Faithfulness: `scripts/muon/faithfulness_bw.sh`.
