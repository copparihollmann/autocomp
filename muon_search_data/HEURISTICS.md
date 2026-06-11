# Muon autocomp heuristics (auto-generated)

Source: `output/transform_ledger.jsonl` (990 rows total, 413 muon).
Regenerate with `scripts/muon/refresh_results.sh`. Curated insights live in MUON_RESULTS.md.

## Outcome distribution (muon)
```
compile_error=343  regressed=36  improved=24  incorrect=5  correct_no_gain=5
```

## Transforms that improved (by speedup)
```
1.24x  P0  SMEM Staging of A and B with Transposed B Layout + Single Ac
1.24x  P0  Shared-Memory Tiling with Warp-Level K-Splitting and Single 
1.13x  P0  SMEM-Tiled MatMul with Interleaved A/B Loading and Single-Ac
1.06x  P4  11 — Reorder multiplications to issue x  B[i] in parallel wi
1.06x  P4  Widen global loads to 64-bit pairs
1.06x  P4  to apply: Strategy 10 — Use mu_fnexp(_Float16) to compute ex
1.05x  P1  Strategy 8
1.05x  P1  Strategy 9
1.05x  P1  Strategy 9 — Use store_shared_from_global for Phase 0c weigh
1.04x  P1  Strategy 8
1.04x  P1  Software Pipelining the GEMM Inner Loop with Prefetch-Ahead 
1.04x  P1  12
1.03x  P6  Loop Reordering and Restructuring (Strategy 2)
1.02x  P2  Cache Reused Data in SMEM (Strategy 3) + ILP via Batching (S
1.02x  P2  Warp-Parallel Row Softmax with SMEM Reduction
1.02x  P2  Fuse QK^T, softmax, and PV into a single tiled Flash Attenti
1.01x  P2  unknown
1.01x  P7  unknown
1.01x  P6  unknown
1.00x  P6  unknown
```

## Negatives — what fails (compile_error / incorrect)
```
185x  compile_error unknown
  3x  compile_error Software Pipelining
  2x  compile_error Cache reused data in local memory instead of reloading 
  2x  compile_error Reduce Data Movement (SMEM Tiling)
  2x  compile_error Precompute global index arithmetic outside inner loops
  2x  compile_error 10 — Replace integer division/modulo with incremental c
  2x  compile_error Use ILP by loading multiple elements per thread before 
  2x  compile_error Strategy 10
  2x  compile_error Loop Unrolling (Strategy 2)
  2x  incorrect     unknown
  1x  compile_error Cache reused data in shared memory (SMEM tiling)
  1x  compile_error Strategy 8
  1x  compile_error Stage Global Loads into Shared Memory (SMEM Tiling)
  1x  compile_error Warp-Specialized A-Row / B-Column SMEM Tiling with Sing
  1x  compile_error Warp-Specialized SMEM Tiling with Transposed B Layout
  1x  compile_error Shared-Memory Tiling with SMEM Staging for A and B Tile
  1x  compile_error SMEM Tiling of A and B with Single Accumulator per Thre
  1x  compile_error Stage global loads into shared memory before compute
  1x  compile_error cache reused data in local (shared) memory instead of r
  1x  compile_error Broadcast A[row][k] to all lanes of a warp
```
