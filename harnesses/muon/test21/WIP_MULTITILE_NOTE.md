# RESOLVED (2026-07-10) — prob21 abandoned, superseded by prob22

The original premise of this note ("need a large OUTPUT tile, blocked by SPAD_DEST=256
C/A overlap") was a DEAD END. The real `mxgemm<C>` driver never does multiple output
tiles: it computes a single TILE_M×TILE_N tile and loops K internally. So an
accelerator-bound anchor does NOT need a large output tile — it needs large K.

The actual multi-tile failure was a **co-model bug**, not a layout constraint: cyclotron's
`mx_loop_ws_spad` always read e8m0 scales from `SF_MEM_A/B` base, but the kernel
double-buffers scales per K-tile (odd tiles at `SF_MEM_* + 0x800`, selected via
`scale_act_sel`/`scale_wgt_sel` = rs1[60]/rs1[61] of `gemmini_mxquant_config_mvout`).
Every odd K-tile therefore read stale even-tile scales. Fixed in
`cyclotron/src/muon/mxgemmini/{state,mod}.rs` (capture the select bits, add the buffer
offset at read time). Regression test `scale_double_buffer_select_is_honored`.

**Anchor is now prob22** = 64×64×512 (TILE_K=64, 8 K-tiles): same proven, RTL-passing
64×64 SMEM layout as prob20 (no overlap), 8× accelerator compute. `calib_mx.sh 22` shows
KCMP and KDMA are both identifiable. K-sweep (prob20/23/22/24 = K 64/128/512/1024) all
PASS bit-exact vs golden.

This prob21 dir (test21.c.WIP etc.) can be deleted; kept only as a breadcrumb.
