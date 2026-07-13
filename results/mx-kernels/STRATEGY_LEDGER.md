# MX autocomp — transformation → effect ledger

| kernel | baseline | best | speedup | winning transforms |
|---|---|---|---|---|
| fork-mm-fp8-128 | 853 | 217 | 3.93× | Eliminate Redundant Double `gemmini_fence()`; (unparsed) |
| fork-mm-fp8-256 | 1607 | 442 | 3.64× | Batch consecutive B-tile mvin calls using `gemmini_block_mvin`; Restructure the B mvin loop to j-outer/k-inner order**; (unparsed); **Strategy 5 — Choose tile layout to avoid scratchpad bank conflicts**; Pack A and B Tiles Contiguously with Correct Layout; Use `mvin2` port for B tile loads to parallelize A and B loads.**; Use `mvin2` port for B tile loads while using `mvin` port for A tile loads with separate c; Increase `lut_update_granularity` to Reduce Requantizer Refresh Overhead; use `mvin2`/`mvin3` ports to parallelize A and B tile loads into scratchpad simultaneously; Batch A and B mvins using MAX_BLOCK_LEN=4 tiles per command.**; Increase `lut_update_granularity` to `tiles_K`**; Interleave A and B mvin commands across both DMA ports**; Interleave per-tile A and B mvins within a unified loop; try new parameter values**: The `SPAD_DEST = 128` is a fixed choice. Looking at the B mvin; Load B tiles using `gemmini_extended_mvin2` (the dedicated second DMA port) while A tiles  |

## Transformation effectiveness (across all runs)

| transformation | times tried | appeared in an improving iter |
|---|---|---|
| (other) | 24 | 7 |
| dual-port mvin (overlap A/B DMA) | 8 | 6 |
| batch mvins (block_mvin) | 3 | 2 |
| raise lut_update_granularity | 2 | 2 |
| remove premature fence | 4 | 1 |
| tile layout / avoid bank conflict | 1 | 1 |
| contiguous tile packing | 1 | 1 |
| tune tile/param values | 2 | 1 |
| SPAD_DEST / output placement | 1 | 1 |
| B-tile reuse / I-band split | 3 | 0 |
| loop reorder / fuse | 2 | 0 |
