# COHERENCE — the invariant, and how it is checked

Four independent models of the same hardware must agree. If they drift, results are silently
wrong, and every speedup number becomes fiction. This file states the invariant and names the
automated check for each link.

```
        RTL (radiance tapeout-330 + gemmini tapeout-260329)     <- the silicon
                 |
                 |  [4] RTL gate
                 v
   cyclotron MX co-model  ==  mx_golden  ==  upstream golden models  ==  spike
        (fast sim)          (our oracle)      (lib/mxgemmini/*.py)     (libgemmini)
                 ^               ^                    ^
                 |    [1] unit   |   [3] xcheck       |
                 +---------------+--------------------+
                          [2] end-to-end kernels
```

**Invariant:** for any MX kernel, `cyclotron == mx_golden == upstream golden == spike == RTL`,
**bit-exactly**. There is no FP tolerance: the idealized numpy/torch matmul disagrees with the
silicon on ~90% of elements, so a tolerance loose enough to admit it also admits broken kernels.

## The checks

| # | Link | Command | Current status |
|---|---|---|---|
| 1 | co-model internals ↔ spike | `cd cyclotron && cargo test --release mxgemmini` | **35/35 pass** |
| 2 | kernel ↔ golden (end-to-end) | `autocomp/scripts/muon/run_problem_mx.sh <N>` | prob20/22/23/24 **PASS** |
| 2b | **negative control** (the gate must be able to FAIL) | run the same kernel *without* `CYCLOTRON_MXGEMMINI=1` | must report `case=4095` |
| 3 | our golden ↔ upstream golden | `autocomp/scripts/muon/xcheck_upstream_golden.py <harness>` | **4096/4096 bit-exact**, K=64/128/512 |
| 4 | everything ↔ silicon | `autocomp/scripts/muon/vcs_gate_mx.sh <N>` | ⏳ pending tapeout-330 simv (VCS license) |

Check 2b is not optional. A gate that cannot fail proves nothing — always confirm the negative
control reports errors before trusting a PASS.

## Measurement (what a "cycle" means here)

- `rtl_kernel_cycles.py` — NET kernel cycles from the run's trace-db. The `DRAIN_ITERS` spin
  dominates `$finish` (~92% of cycles, with ~100k variance), so the raw simv cycle count is
  **useless for perf**. The drain loop's first cycle = kernel completion; this is provably
  drain-independent (identical for `DRAIN=2000` and `DRAIN=200000`).
- `kernel_utilization.py` — kernel-only cycles / IPC / ideal-vs-actual, from ELF symbol PC ranges
  (`kernel_body` ∪ `mxgemm*`, excluding `main`/`verify_body`) intersected with the trace-db.
  Works on **both** cyclotron (`--gen-trace true`) and RTL — same schema, directly comparable.
- Trace `cycle` is the Muon **core** clock = 2× the tile clock the vcs gate divides `$finish_ps` by.
- **Trace cycles require the TracerBlackBox free-running counter (radiance 6462826+a4656ba).**
  Stock tapeout-330 predates it, so its `cyclotron_trace` DPI passes no cycle → the co-model reads
  the next arg (`trace_valid`, 0/1) as the cycle and every field shifts by one → the whole trace-db
  is garbage (`cycle∈{0,1}`, `rtl_kernel_cycles.py`→0). This hits **Verilator AND VCS** alike (same
  Cyclotron.cc+Rust). The tapeout-330 build here backports those 2 trace-only commits (DUT
  unchanged). Correction to an earlier claim: "cycle is a property of the RTL" is true of the
  *silicon* but the trace-db field only exists once that counter is elaborated in — so it is NOT
  automatically present just because two simulators run the same RTL.

**Known bias (measured on tapeout-330, symbol-based kernel_body span, core cyc):**

| K | tiles | RTL | cyclotron | ratio |
|---|---|---|---|---|
| 64   | 1  | 52015 | 15622 | 3.33× |
| 128  | 2  | 53567 | 17243 | 3.11× |
| 512  | 8  | 64136 | 36393 | 1.76× |
| 1024 | 16 | 78803 | 61943 | 1.27× |

cyclotron under-models the fixed SIMT/setup overhead (low intercept ~13k vs RTL ~50k) while its
per-tile slope runs steep — so it *under*-predicts total kernel cycles, worst at low K (3.3×),
converging toward 1 at high K. The 1.76× at K=512 is stable across the +35 simv and real tapeout-330
silicon. **KCMP/KDMA cannot fix this** — these 64×64 baselines are SIMT-bound in the co-model, so the
accelerator coefficients are unidentifiable (see VERSIONS.lock). **Rank on RTL-gated numbers; treat
cyclotron speedups as directional.**

## Traps that have already cost real time

1. **`SPAD_DEST` must be computed, never hardcoded.** C lives in the scratchpad beside the
   *double-buffered* A/B tiles, so its address depends on the operand footprint. The old
   `SPAD_DEST = 256` was correct only for 64×64 with TILE_K≤64; every larger tile wrote C on top
   of the A operand — silently, with no error. `GemmConfig::SPAD_DEST()` now computes it and
   `static_assert`s feasibility (128×128 needs TILE_K ≤ 128; 128×128 tk256 and 256×256 tk256 are
   genuinely impossible).
2. **`BANK_ROWS` must be 2048** (128 KiB SMEM). A wrong pin puts the B operand out of range —
   silently. Guarded by `static_assert`.
3. **Scale factors are double-buffered per K-tile** — odd tiles at `SF_MEM_* + 0x800`, selected by
   `rs1[60]`/`rs1[61]` of `gemmini_mxquant_config_mvout`. Ignoring the select bits makes every odd
   K-tile read stale scales (this was a real co-model bug).
4. **A missing data header means a kernel has never been built** — and therefore never tested.
   Treat any kernel without a committed `mxgemm.data.*.h` as unverified.
5. **RTL "failures" are often the write-drain race**, not numerics — `-DDRAIN_ITERS` (via
   `EXTRA_MU_CFLAGS`, which `common.mk` must actually consume). Scattered/early mismatches = a real
   bug; a contiguous wrong tail = drain.

## Golden model provenance

`autocomp/scripts/muon/mx_golden` implements the **hardware** accumulation semantics: a 16-deep
systolic array with the `acc_e`/`acc_m` per-column precision schedule, product truncation to
`PROD_MANT_BITS=7`, and one e8m0 scale per 32-element K group. Validated bit-exact against **real
spike libgemmini** and, independently, against the **upstream** `lib/mxgemmini` golden models.
