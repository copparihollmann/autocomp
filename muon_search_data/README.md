# Muon autocomp search data (for analysis / reuse later)

Durable, git-tracked snapshot of the autocomp-on-Muon search runs — the decisions, transforms,
winning kernels, and cost. Lives in git "with the rest" (branch `muon-fidelity-session`, alongside
`MUON_SEARCH_TUNING.md`, `MUON_SETUP.md`, `MUON_RESULTS.md`). Re-snapshotted after each run.

## What's here (compact, committed)
- `ledgers/transform_ledger.jsonl` — **the decisions dataset**: every optimization transform the
  search applied, per problem/iteration, with its measured effect (cycles before/after). ~990 entries.
  This is the primary "data for later" — what worked, what didn't, per kernel.
- `HEURISTICS.md` — auto-derived heuristics summary (from `refresh_results.sh`).
- `best_kernels/muon_<prob>_<hash>.cpp` — the winning kernel per problem per run (the search outputs).
- `ledgers/muon-spend.jsonl`, `muon-spend_total.json` — per-call + cumulative cost ledger.

## Full candidate corpus (bulk, on /scratch — `output/` is gitignored)
Every candidate kernel + plan + per-iteration metrics + eval verdicts/diagnostics + per-run cost
ledgers are under `autocomp/output/` and the archived runs:
- `output/_archived_0009/` (first iters8 run, Sonnet)
- `output/_arch2/`, `output/_arch3/` (intermediate)
- `output/_arch_gemini_*/` (pre-Gemini-sweep archives)
- `output/built:muon_muon_<p>_beam_iters8_cyclotron/` (live/most-recent run)
These persist on /scratch (~157M). To preserve off-scratch, tar them:
`tar czf muon_corpus_<date>.tgz autocomp/output/_arch* autocomp/output/built:muon_*`.

## Provenance / config
Search config + fidelity envelope: `../MUON_SEARCH_TUNING.md`. Models: Gemini 3.1 Pro Preview
(provider gcp; dodges Bedrock daily-token throttle) at iters8/beam3; earlier runs used Sonnet 4.6.
Eval: cyclotron `--timing` (timed run authoritative for correctness). Spend cap $200 (mode=stop).

## Key caveats for using this as data
- cyclotron is **issue-bound** and **under-rewards SMEM-tiling** (see MUON_SEARCH_TUNING §0) — speedups
  here are cyclotron-cycle ratios; matmul/attention SMEM gains are under-represented vs RTL.
- Some early speedups (pre eval-fix) were timing-false-passes; the current timed-authoritative eval
  is trustworthy (e.g. attn64 corrected 1.11x→1.00x).
