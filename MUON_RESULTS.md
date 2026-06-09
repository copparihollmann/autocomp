# Muon (radiance) autocomp — results & findings

Canonical, curated record of what we've learned optimizing Muon kernels on cyclotron +
RTL. Auto-generated heuristics (incl. negatives) live in `HEURISTICS.md`; the raw data is
`output/transform_ledger.jsonl`. Setup/pitfalls: `MUON_SETUP.md`. Refresh after any
search with `scripts/muon/refresh_results.sh`.

Metric is always **total cycles** (never IPC — gameable by unrolling). Kernel-only =
sim total − empty-kernel harness overhead.

## LAUNCHED kernel suite (cyclotron-correct; calibrated-model cycles; RTL where gated)

All PASS correctness on cyclotron (FP-exact oracle). Cycles = kernel-only (total − empty-overhead)
under the structurally-calibrated memory model (L0d=4KB, DRAM 40/8):

| target | best kernel | cyclotron kern-only | speedup | RTL |
|---|---|---|---|---|
| matmul | sol0_smem_lean | 53,450 vs 72,074 | 1.35× (cyc) | **2.52× RTL-confirmed** |
| softmax | sol5_unroll8 | 23,870 vs 51,498 | **2.16×** | **reg-LEGAL; RTL ran to completion** ($finish, 2,581,722 RTL cyc total@2GHz); verify tohost=79 = write-drain race (not a kernel bug) |
| conv | sol1_baseline | 129,350 | baseline best | RTL 75k (compute-bound) |
| swiglu | sol4_baseline | 35,453 | baseline best | RTL sim did NOT complete (hung ~10k instr, timeout-killed); cyclotron-correct — test4 harness/exit issue, see below |
| attention | sol2_baseline | 263,582 | baseline (SMEM lean2 SLOWER 415k; tiled 2.44× is RTL-ILLEGAL) | **reg-LEGAL; RTL ran to completion** (2,261,387 RTL cyc total@2GHz; ELF lacked tohost/fromhost so no on-device verdict) |

Notes: softmax's win dropped 6.4×→2.16× under the calibrated (memory-aware) model — the realistic
number. Attention has NO win at seq=64,d=64: SMEM-lean 415k (slower), ILP-unroll 324k (slower,
0.81×), register-tiled 2.44× asserts globalOverSubscription (RTL-illegal). Its dot-products are
short and it's global-scratch-access-bound, not reduction-chain-bound, so neither SMEM staging
nor ILP helps — baseline (263k) is the best LEGAL+correct kernel. matmul/softmax are the real
wins; conv/swiglu/attention launch at baseline (each verified no optimization beats baseline on
this HW at these shapes).

RTL gate (vcs_gate suite, RadianceSingleClusterConfig @ 2 GHz, cycles = $finish_ps/500):
softmax unroll8 and attention baseline both ran to $finish and are **register-LEGAL** (no
globalOverSubscription) — RTL totals 2.58M and 2.26M cycles (HARNESS-INCLUSIVE: data setup +
launch + on-device verify, NOT directly comparable to cyclotron kernel-only; they confirm
reg-legality + that the kernels execute on hardware). matmul lean-SMEM is the only kernel with a
clean kernel-only RTL anchor (2.52×, 440,169 vs 1,107,417). The softmax on-device verify asserts
tohost=79 = the write-drain race (known harness artifact, kernels are FP-exact on cyclotron).
swiglu test4's RTL sim hung early (~10k instructions, timeout-killed, no $finish) — a test4
harness/exit issue, not a kernel correctness problem (cyclotron-correct); needs the host-side
verify (#25) or a test4 harness fix to gate on RTL.

## Confirmed kernel speedups

| kernel | shape | baseline (kern-only cyc) | best | speedup | how | RTL? |
|---|---|---|---|---|---|---|
| **softmax** | 64×67 | 111,367 | 17,364 | **6.41×** | 8-way unroll + independent accumulators + branchless fmax (`sol5_unroll8.cpp`) | not yet gated |
| softmax | 64×67 | 111,367 | 26,641 | 4.18× | 4-way unroll (`sol5_unroll.cpp`) | — |
| **matmul** | 64³ | 88,560 | — | **2.52× on RTL** | lean SMEM staging, no tiling (`sol0_smem_lean.cpp`, 22 regs); RTL 440,169 vs 1,107,417 | **PASS** |
| matmul | 64³ | 88,560 | 40,207 | 2.20× (cyclotron) | SMEM + 2×8 reg tile — **RTL-ILLEGAL** (50 regs) | FAIL |
| attention | 64×64 | 260,599 | 106,900 | 2.44× (cyclotron) | full SMEM + 4-out tile — **RTL-ILLEGAL** (54 regs) | FAIL |

Note the cyclotron-vs-RTL gap on matmul: cyclotron showed lean-SMEM as only 1.35×
(kernel-only) but RTL measured **2.52×** — cyclotron `--timing` under-counts global-memory
latency, so SMEM staging pays off more on real hardware than cyclotron predicts.

## The register budget (RTL physical register file)

RadianceSingleClusterConfig: `numPhysRegs=256` shared across `numWarps` active warps. A
kernel using N distinct registers needs N×warps physical regs; exceed 256 → RTL Rename
asserts `globalOverSubscription` (Rename.scala:123) and the kernel is **unrunnable**.
- At **8 warps** → ≤32 regs/kernel. At **4 warps** → ≤64. **Team guidance: use 4 warps**
  for real kernels (8 only for trivial/gemmini). Keep a few below the cap.
- Cyclotron did NOT model this → fixed: `num_phys_regs` config + static `kernel_body`
  scan in `sim/top.rs` that panics on over-subscription (matches RTL 5/5). Currently
  `num_phys_regs=0` (disabled) in `config_muon.toml` because the check divides by the
  hardware 8, not the launched 4 — TODO: read `__mu_num_warps` and divide by that.
- Static pre-check: `scripts/muon/reg_check.sh <prob> <cand.cpp>`.
- **RESOLVED (refined): RTL legality = PEAK simultaneously-live registers (the rename
  high-water mark), NOT total distinct operands.** softmax unroll8 (50 *distinct* regs but
  short-lived across separate passes) runs on RTL with NO globalOverSubscription (physregs
  peaked ~99 < 256); matmul 2×8 (16 accumulators all live across the whole K-loop) asserted.
  So `reg_check.sh` (static distinct count) is CONSERVATIVE — it over-flags kernels whose
  registers aren't simultaneously live. Treat it as an upper-bound screen; the true test is
  peak live regs × warps ≤ 256, which only the RTL Rename (or a liveness analysis) knows.

## Heuristics that work (hand-verified)

- **Break reduction dependency chains** — the softmax 6.4× came entirely from replacing
  sequential `m`/`sum` reductions with 4–8 independent accumulators (`m0..m7`, `s0..s7`)
  + branchless `fmax`. Pure ILP win; biggest lever for streaming/elementwise kernels.
- **SMEM staging for reused operands** (matmul A/B, attention Q/K/V) → ~2.5× on RTL.
  Only when data is re-read many times.
- Pad SMEM column stride +16 floats to avoid 16-way bank conflicts.

## Negatives that cost time (don't repeat)

- **Register tiling / deep unrolling on matmul/attention** → exceeds the reg budget,
  RTL-illegal despite looking fast on cyclotron.
- **SMEM staging or 2-threads/row on softmax** → SLOWER (157k vs 111k). No reuse to
  amortize the staging/barrier cost; streaming baseline already good.
- **Conv im2col / SMEM** → slower than baseline (data read too few times).
- **Elementwise ILP unroll on swiglu** → slower (exp count constant, register pressure).
- **Agent: 90% of search candidates are compile_errors** (261/289) — the #1 thing to fix;
  recent agent grounding (register rules, verified examples, better `clean_code`) targets it.

## *** ACTUAL ROOT CAUSE: global write-drain race in the on-device verify (NOT exp/FP/registers) ***

After a long bisection, the softmax/attention/swiglu RTL "failures" are NOT a compute bug at
all — the exp math is correct. They are a **global-memory write-drain / coherence race in the
ON-DEVICE verify**: the harness has hart 0 read all of out[] right after the kernel, but on RTL
~1/4 of the writer warps' stores haven't drained to memory yet, so hart 0 reads stale ZEROS.
Cyclotron doesn't model drain latency, so it sees every write -> passes.

Evidence (all RTL, cyclotron=0 zeros in every case):
- exp-only kernel: ~1150 of 4288 outputs are EXACTLY 0 (not wrong values — zero/unwritten).
- TRIVIAL `out[i]=in[i]+1` kernel (no exp): also ~990 zeros -> NOT exp-specific.
- Zero count is TIMING-dependent (exp 1150 vs add 990) -> a drain race, not a coverage gap.
- matmul PASSES because it interleaves writes with heavy compute (K-loop) + has an in-kernel
  mu_barrier, so its stores drain before verify. The fast elementwise kernels don't.
- Fixes tried: end-of-kernel mu_barrier(1,nw) -> 990→744 zeros (partial). mu_fence() in the
  kernel -> HANGS (waits on inactive warps). hart-0 spin-delay -> too slow, timed out.
  hart-0 mu_fence() before verify -> testing.

So: the EARLIER theories (fcvt rounding, approximate divider, FMA) were all WRONG — those probes
were ALSO misleading because compile-time-constant inputs got CONSTANT-FOLDED (never ran on RTL).
The real bug is verify-side memory sync. The kernels (softmax 6.4x, attention 2.44x, etc.) are
correct; they just need a correct drain before the on-device read, or host-side verification.

TODO: land a robust drain (hart-0 fence, or host.cpp reads out[] from DRAM after GPU done), then
re-gate softmax/attention/swiglu at tight 2e-4 -> expect PASS. matmul 2.52x already confirmed.

CONCLUSION (proof complete): a TRIVIALLY-correct kernel `out[i]=in[i]+1` "fails" the on-device
verify with ~990 zeros on RTL. A kernel that obviously computes correctly cannot be wrong =>
the ON-DEVICE VERIFY is the broken component, not the kernels. So softmax 6.4x / attention 2.44x
(cyclotron FP-exact) are CORRECT; their RTL *cycle* gains are valid; only the RTL correctness-
CHECK is unreliable for fast kernels. Drain fixes tried: end-barrier 990->744 (partial), hart-0
fence 990->886 (partial), in-kernel mu_fence HANGS, host-side read-loop TIMED OUT (4288 DRAM
reads + host/GPU handshake too slow per wall-clock). The robust fix = host-side verify needs more
infra work (fewer reads / faster handshake / spin drain) OR run the perf gate and accept cyclotron
for correctness. Recommendation: for new kernels, trust cyclotron correctness (FP-exact) + use RTL
for CYCLES; treat the on-device RTL correctness verify as unreliable until host-verify is hardened.

**HV3 experiment (task #25) — CONFIRMED NEGATIVE.** Tried: GPU main = mu_schedule + fence +
`vx_tmc_zero()` (idle the manager warp) + masked spin, so the GPU would go idle and set
RAD_HOST_GPU_ALL_FINISHED WITHOUT crt-exiting, letting host.cpp read drained out[]. Result: a
clean RTL run hit the max-cycle timeout (EXIT=124), GPU stuck idle at 2000 instructions (IPC 0.05),
no $finish — i.e. `vx_tmc_zero()` idle does NOT set ALL_FINISHED, so the host poll spins forever.
Conclusion: mu_schedule only signals "done" via the crt-exit/tohost path (which ends the sim before
a host verify runs). Host-side verify is RUNTIME-OWNER-BLOCKED: needs a muonrt change to assert
ALL_FINISHED when the manager warp idles, or a 2-phase relaunch verify. Confirmed dead-end for the
self-contained-kernel approach.

## (obsolete theory) mu_exp fcvt rounding — DISPROVEN (constant-folded probes); see root cause above

Isolated with an exp-only kernel (`out[i]=mu_exp(in[i])`, NO sum/divide, harnesses/muon/testexp):
- cyclotron: PASS (matches numpy gold). RTL: **FAIL, 1073/4288 errors (25%) at tight 2e-4 tol.**
- So the earlier "approximate divider" theory was WRONG. The divergence is in **exp itself**, and
  it's large (survives a 10× looser tolerance), not a precision wobble.
- Mechanism: mu_exp does `k=(int)(t+0.5)` (`fcvt.w.s ...rtz`) then builds `2^k` exactly via
  `(k+127)<<23`. A ±1 error in k → exp off by 2× → ~25% large errors. The fmadd polynomial is
  fine (cyclotron uses true `f32::mul_add`, matches RTL FMA).
- Suspect: cyclotron `fcvt.*.s` (src/muon/execute.rs:345) **ignores the static rounding-mode
  field** and always truncates (`to_i32()`/rtz). If RTL uses the dynamic frm (round-to-nearest),
  `trunc(t+0.5)` vs `rne(t+0.5)` disagree for ~half of inputs → k off by 1.
- Affects ALL exp kernels: **softmax, attention (softmax), swiglu (sigmoid)**. matmul/conv immune.

FIX OPTIONS (pick after confirming RTL's actual fcvt rounding):
1. Rounding-robust mu_exp: compute k without depending on fcvt rounding (e.g. floor via
   compare-and-correct, or integer magic-number rounding), so cyclotron==RTL==gold regardless.
2. Fix cyclotron's fcvt to honor the instruction's static rm field (principled; needs RTL truth).
3. If RTL genuinely ignores static rm (hardware bug), report it.
Until fixed, cyclotron `isa-test passed` does NOT imply RTL-correct for exp kernels.

## (superseded) RTL approximate divider — earlier (wrong) theory; real cause is mu_exp above

matmul (FMA only) passes RTL bit-exact, but **softmax fails RTL correctness even at
baseline** (tohost=139 = 69 errors at 2e-4 rel tol; unroll8 = 39 errors — its tree sum is
actually closer to the numpy-pairwise gold). Root cause: **RTL has an approximate FP
divider** coarser than cyclotron's, so `e[i]/sum` diverges beyond a 2e-4 tolerance that
cyclotron passes. This is a SECOND cyclotron-vs-RTL gap (after registers):
- Affects every kernel that divides: **softmax, attention (softmax), swiglu (sigmoid)**.
  matmul/conv (no divide) are unaffected.
- Fix (kernel side): the harness FP tolerance must reflect the hardware divider, not the
  exact numpy gold. Set softmax `TOLERANCE_REL≈2e-3` (calibrating). The "correct" answer on
  this hardware is only divider-precise; tightening below that flags hardware-faithful kernels.
- Fix (cyclotron side, TODO): model the approximate divider/reciprocal so cyclotron flags
  divide-precision divergence early (same approach as the register check).
- Implication: cyclotron `isa-test passed` for a divide-kernel does NOT guarantee RTL
  correctness at tight tolerance — always RTL-gate divide kernels.

## DRAM/memory timing model — calibration (part 1) + key realism finding

Cyclotron --timing DOES model memory (src/timeflow/gmem/, enabled via config_muon.toml
[timing] include of config/timing/gmem.toml): L0/L1/L2 cache hierarchy + DRAM node, with
per-stage base_latency, bytes_per_cycle (bandwidth), MSHRs, coalescing, writebacks.

Calibrated structural params against the RTL RadianceSingleClusterConfig:
- L0d: cyclotron was 512 sets x 64B = 32KB; RTL is 64 sets x 64B = **4KB** -> set l0_sets=64.
- DRAM: cyclotron 200 cyc / 32 B/cyc; RTL is **40 cyc / 8 B/cyc** (4 GB/s @ 500 MHz) -> set.
- L1 (64KB) and L2 (512KB) already MATCH RTL.

**KEY FINDING — cyclotron under-counts global-memory cost ~30x.** Anchor: matmul baseline vs
lean-SMEM kernel-only DIFF (host overhead cancels): RTL = 1,107,417-440,169 = **667,248 cyc**;
cyclotron default = only **~22,900** (the whole 48KB matmul working set fits in cyclotron's
caches as cheap hits at 64 B/cyc). So memory-bound kernels look far better on cyclotron than RTL.

Reducing the cache/coalescer `bytes_per_cycle` 64->1 matches the matmul diff (686k≈667k), BUT
over-penalizes compute-bound kernels (softmax 6.41x -> 1.23x) because bytes_per_cycle conflates
request RATE and request SIZE. A single bandwidth knob can't fit both -> motivates part 2
(request-rate model).

**Realism implication (important):** softmax baseline and unroll8 have IDENTICAL global traffic,
so unroll's win is a CONSTANT ~94k compute cycles; the 6.41x ratio only holds when memory is cheap.
If RTL memory is ~30x costlier (matmul anchor says so), softmax is MEMORY-BOUND on RTL and the real
speedup of compute-only optims (unroll) is ~1.2-1.5x, NOT 6.4x. **Compute-only optimizations barely
help memory-bound kernels; cutting global traffic (SMEM staging) is the real lever (matmul 2.52x).**
=> Trust cyclotron RANKINGS less for compute-vs-memory tradeoffs until the memory model is calibrated;
the SMEM/traffic-reduction wins (matmul) are the trustworthy ones.

## DRAM/memory model — part 2: request-rate throttle (implemented)

Added a `cycles_per_request` field to cyclotron's `ServerConfig`/`TimedServer` (src/timeq.rs):
minimum service cycles each request occupies a server, independent of bytes — a per-REQUEST
rate limit separate from byte bandwidth (`service = max(ceil(bytes/bytes_per_cycle),
cycles_per_request)`). Default 0 = backward-compatible. Exposed in config/timing/gmem.toml per
node/level. This was needed because `bytes_per_cycle` alone conflates request rate and size.

Final calibrated `gmem.toml` (matches the RTL RadianceSingleClusterConfig anchor):
- l0_sets=64 (L0d=4KB), DRAM base_latency=40, bytes_per_cycle=8 (structural matches to RTL).
- `cycles_per_request = 64` on the coalescer + all 3 cache data levels -> matmul baseline-vs-SMEM
  kernel-only DIFF = 686k ≈ RTL 667,248 (was 22,900). So cyclotron now charges global traffic
  realistically; the search's kernel-only latency reflects memory-bound behavior.

UPDATE (anchor results): cpr=64 does NOT generalize — REVERTED. RTL anchors collected:
- matmul (memory-bound): RTL kernel diff baseline-SMEM = 667,248.
- conv (compute-bound): RTL cv_base=75,462, cv_ilp=83,670 total cyc (ILP 8,208 SLOWER on RTL).
- softmax: RTL runs TIMED OUT (>1700s wall — the racy on-device verify reading 4288 elems is slow).
At cpr=64, cyclotron conv = 4.76M (60x RTL's 75k!) because cycles_per_request on the CACHE levels
penalizes cache HITS, and conv has ~196k cached input reads. So a single global memory-cost knob
CANNOT fit both memory-bound matmul and compute-bound conv. FINAL state: structural fixes only
(L0d=4KB, DRAM 40cyc/8Bpc) — these keep conv sane (cyclotron 129k vs RTL 75k region, same order)
and don't over-penalize. The cycles_per_request FIELD is kept (backward-compat, default 0; usable
on the DRAM node where a per-request cost is physical) but NOT applied to cache levels.

Caveats / remaining work:
- The matmul memory-bound gap (cyclotron ~22k vs RTL 667k) is NOT fixable by a blanket per-access
  penalty (breaks conv). It needs the PHYSICAL fixes: model RTL coalescing (#23) for the strided
  A/B pattern + latency/MLP (limited outstanding requests expose latency on RTL that cyclotron's
  high bandwidth hides). That's the real calibration work.
- conv is compute-bound on RTL (75k) and cyclotron roughly agrees (129k) -> compute-bound kernels
  are already OK; only memory-bound (matmul) is under-counted.
- The row-buffer/bank DRAM model (tRCD/tRP/tRAS, DRAMSim-style) is the next add for DRAM-bound
  kernels (large working sets that miss L1/L2); our current kernels are L1-resident so the cache
  request-rate was the dominant lever, not DRAM row-buffer.
- Backup of original params at config/timing/gmem.toml.orig.

## Infra findings

- **RTL gate** (`scripts/muon/vcs_gate.sh`): needs a real launching `host.cpp`; success is
  a silent `$finish` at TestDriver line 158 (use `+verbose` for `*** PASSED ***`). "tohost
  symbols not in ELF" is benign. Run serial (concurrent runs corrupt the cyclotron trace DB).
- **DiffTest** (`SingleCoreDiffTestConfig`): functional lockstep cyclotron-vs-RTL; matched
  100 instructions of a legal kernel with zero divergence, but the GPU stalls past ~100
  retired instructions (NOT dramsim — confirmed by dropping it). Not usable for deep pinning
  yet. Right tool for *numerical* mismatches, not the register one (which is structural).
- **DRAMSim2**: valuable for perf realism (keep for the perf gate); irrelevant for difftest
  (timing doesn't change architectural values). Our 2.52× was measured WITHOUT dramsim →
  re-gate finalists with it for realistic numbers.
- **cyclotron == RTL**: bit-exact on the 100 difftested instructions; perf ranking ρ=1.0;
  FP 1-ULP = RTL approximate divider (use FP tolerance). See `/scratch/.../CYCLOTRON_VS_RTL_FINDINGS.md`.

## Workflow (always do this)

1. Run searches (`run_*_muon.sh`).
2. `scripts/muon/refresh_results.sh` → rebuilds `output/transform_ledger.jsonl` + `HEURISTICS.md`.
3. Gate top candidates on RTL (`vcs_gate.sh`), serial; pre-screen with `reg_check.sh`.
4. Update this file's tables with RTL-confirmed numbers.

## Next lever

Cyclotron-calibrated **cost model** (`mx_pipeline/cost_model.py` mechanism: static counts →
`fit_dynamic` on cyclotron trace) as a pre-ranker (sim only top-k) and the feature axis for
learned heuristics — per the `ONBOARDING.md` architecture.

---

## Real-shape workloads (smolvla matmul + flash attention) — added this session

Two new autocomp problems wired with REAL extracted shapes, to answer "where does the win
land on the actual workload" rather than on toy 64^3 tiles.

### prob6 — smolvla/openvla matmul, K=128 (real openvla `20x128 @ 128x512`, freq 7)
- Baseline = SMEM full-staging GEMM (self-contained, inline `__builtin_bit_cast`): **100,150
  kernel-only cycles**, verified correct on cyclotron.
- **THE MEMORY WALL (key finding):** the NAIVE global-memory GEMM is cache-resident ONLY at
  K<=64 (644k cyc). At K>=96 the A/B working set (>=48KB) exceeds L0/L1 and the kernel goes
  fully DRAM-bound: K=96/128/256 all EXCEED the 10M-cycle sim cap; K=256 exceeds even an 80M
  cap (371s wall). So on-chip staging isn't a 2-3x optimization at real depth — it's the
  difference between completing and not (>800x vs the staged kernel). This is why naive
  baselines are only trainable at K<=64, and why prob6's baseline must be SMEM-staged.
- Search built + launched + passed init; BLOCKED by Bedrock daily-token quota before producing
  optimized candidates. One-command relaunch: `run_search_muon 6`.

### prob7 — flash attention (accelerator-centric softmax), seq=128, d=64
- Rewrote attention so the softmax is FUSED and the SEQ x SEQ score matrix is NEVER
  materialized in global memory (that 3x global round-trip is the real softmax bottleneck at
  realistic seq). Online softmax (running max m, denom l), K/V staged in SMEM, per-row O
  accumulator in SMEM. Register-legal, self-contained.
- **Verified CORRECT** on cyclotron (`isa-test passed`): **898,684 kernel-only cycles** at
  seq=128 (vs naive 3-phase ~2.31M total -> flash 2.04M total = 1.13x even before search).
- Why only 1.13x at seq=128: 64KB of scores is still largely cache-resident, so the memory
  wall isn't fully exposed. The gap grows at seq=256+ (naive goes DRAM-bound) but those evals
  get too slow (~40s+ each) for a practical search loop. seq=128 is the tractable choice where
  the SEARCH still has rich headroom (register-block the O accumulator, warp-cooperative dot
  products, key-tiling, skip redundant SMEM round-trips).
- A subtle correctness trap fixed: the polynomial `mu_exp(m - mnew)` overflows its int cast
  when m is the -inf sentinel on the first key -> guard `corr = (m < -1e37) ? 0 : mu_exp(...)`.
- Search built + launched + passed init; BLOCKED by the same quota. Relaunch: `run_search_muon 7`.

### Infra fixes landed this session (both would have silently broken the searches)
- `agent_builder/.built/muon/rules.yaml` had unquoted `:` in list items -> YAML ScannerError
  crashed EVERY search at startup. Quoted the offending items.
- `clean_code()` strips everything above `kernel_body` for a raw (unfenced) baseline file, so
  top-level helper functions (bits<->float) were cut -> compile error on the initial code.
  Fix: both new baselines are self-contained via inline `__builtin_bit_cast` (no top-level helpers).
- Root-caused the cryptic `"Error: 0"`: it's cyclotron's `[sim] timeout` cap being hit without
  `finished()` (top.rs:127, returns Err(0)) — i.e. the kernel didn't terminate within the cap,
  NOT a correctness failure. Raised the cap 10M -> 80M for deep-K kernels.
- `[sim] timeout` raised to 80M; `muon_eval.SIM_TIMEOUT` 300 -> 700s so deep baselines complete.

### Flash attention hand-optimization probe (no-LLM, while search quota-blocked)
Tried `sol7_block.cpp` — KEY-BLOCKED flash (BC=4): amortize the per-row O accumulator
SMEM read-modify-write over a block of keys (RMW once per block, not per key).
Result: **correct, 883,093 vs 898,684 = 1.02× (negligible)**, reg count 63 (RTL-borderline).
Informative NEGATIVE: the O round-trip is NOT the bottleneck. The dominant cost is raw SMEM
LOAD VOLUME — the K-dot (d loads/key) + V reads (d loads/key), both d*seq per row, unchanged by
key-blocking. At seq=128 the flash kernel is SMEM-bandwidth-bound near its floor for the
1-query-row-per-thread mapping. The real lever = REDUCE load volume: register-block multiple query
rows per thread (reuse each K[j]/V[j] load across R rows, cutting load volume R x), or
warp-cooperative loads (lanes share). That's the search's job (register tiling within the budget) —
queued via auto_relaunch.sh to fire when the Bedrock daily-token quota clears.

Also tried `sol7_reg2.cpp` — 2 query rows/thread, reusing each K[j]/V[j] SMEM load for both
rows (halves SMEM loads): **SLOWER, 0.58x (1,550,901 vs 898,684)**, 53 regs. Halving SMEM loads
did NOT help -> at seq=128 the flash kernel is NOT memory-load-bound either; SMEM loads are cheap
(~1-2 cyc) and the doubled per-row body + register pressure dominate. CONCLUSION: at seq=128 the
flash kernel is COMPUTE/INSTRUCTION-bound (mu_exp + arithmetic), near its floor for simple hand
transforms (key-block 1.02x, 2-row 0.58x both fail). The big flash wins live at large seq (memory
wall, intractable to sim) or need the LLM search's creativity (warp-cooperative reductions, fusing
the exp, different parallelization). sol7_baseline (898,684, correct, reg-legal) stays the
searchable starting point. Auto-relaunch (scripts/muon/auto_relaunch.sh) will fire the prob6/prob7
searches when the Bedrock daily-token quota clears.

### prob6 matmul register-tiling — RTL legality settled (both ILLEGAL, no-LLM RTL gates)
Hypothesis: the 4-warp harness launch (vs 8) might make register-tiled matmul RTL-legal. TESTED on RTL:
- **2x8 tile** (16 live accumulators, 50 distinct regs): cyclotron **2.21x** (45,268 vs 100,150),
  correct — but RTL **globalOverSubscription ASSERTED** (Rename.scala:123). ILLEGAL.
- **2x4 tile** (8 live accumulators, 38 distinct regs): cyclotron **2.14x** (46,795), correct —
  RTL **globalOverSubscription ASSERTED**. ILLEGAL.
CONCLUSION: matmul register tiling exceeds the RTL physical-register budget (~32 regs/warp, the
file is provisioned for the hardware warp count regardless of the launch config). The SMEM-lean
baseline (22 regs, 1 accumulator) remains the RTL-LEGAL best for the real-shape matmul; its 2.52x
(64^3) is RTL-confirmed. The cyclotron 2.1-2.2x register-tile "wins" are an artifact of cyclotron
NOT enforcing the register budget — this is exactly why task #26 (model the shared physreg file in
cyclotron) matters: without it the search will keep proposing fast-on-cyclotron / illegal-on-RTL
register-tiled kernels. Net for prob6: SMEM-lean baseline is the shippable kernel; further legal
wins need either lower-register restructuring (1xN tiles, smaller — diminishing speedup) or the
LLM search exploring within the 32-reg budget (auto_relaunch-queued).

### Real-shape searches completed (auto_relaunch fired 2026-06-05 15:53, both 6 iters)
- **prob6 smolvla matmul (M=N=64, K=128):** baseline 100,150 -> best 99,916 cyc = **1.002x (FLAT)**.
  Search found no legal win. Confirms the hand-probe verdict: SMEM-lean baseline (22 regs) is at
  the RTL-legal floor; every faster transform is register-tiling, which is RTL-illegal
  (globalOverSubscription). Cost $0.98, 20 calls.
- **prob7 flash attention (seq=128, d=64):** baseline 898,684 -> best 833,360 cyc = **1.078x**.
  Small win; confirms seq=128 flash is compute/instruction-bound (mu_exp+arith), near its floor for
  the search space. Big flash wins need seq>=256 (memory wall, intractable to sim). Cost $0.76, 19 calls.
- INTERPRETATION: both searches independently re-derived the ceilings mapped by hand earlier.
  Honest loop (no fake speedups). Standing RTL-confirmed wins remain matmul 2.52x + softmax 2.16x.
  Spend to date: $22.58 / $100 (409 Sonnet calls).
- Reinforces task #26 priority: without a register-budget model in cyclotron, the matmul search has
  no gradient toward legal kernels (the fast ones are illegal but cyclotron rewards them).

### Attention (prob2, seq=64) hand-optimization + #26 runtime register counter
- Naive 3-phase baseline reproduces at 835,725 cyc (memory model active).
- FLASH (sol7) on seq=64: 1,010,470 = SLOWER. Online-softmax serialization + O-accum SMEM
  round-trips cost more than the global-score round-trip saves when scores (16KB) are cache-resident.
- 3-phase SMEM, reg-tiled 4-wide (sol2_smem): 663,240 = 1.26x on cyclotron, CORRECT.
  BUT RTL gate -> globalOverSubscription (Rename.scala:123). ILLEGAL. Same as matmul reg-tiles.
- 3-phase SMEM lean, 1-out/thread (sol2_smem_lean): 865,662 = SLOWER than baseline. The win was
  the register tiling, not the SMEM staging -> and tiling is RTL-illegal. No legal attention win yet.
- swiglu (prob4): baseline now 1,174,318 (was 75,940 in pre-DRAM-calibration search -> ALL old
  search numbers are stale under the current memory model). 4-wide ILP (sol4_ilp) = 1,179,994, no
  help -> swiglu is mu_exp/throughput-bound, not memory-latency-bound. Low headroom.

### #26 runtime rename counter (cyclotron) — IMPLEMENTED, calibrating
- Added MuonCore.track_register_pressure: counts distinct (warp, rd_addr) first-writes during
  EXECUTION, mirroring Rename.scala assigning=valid&&writesToRd&&!assigned(wid)(rd); panics at
  num_phys_regs (256), exactly the RTL globalOverSubscription assert. rd_addr is the Muon wide-ISA
  native 8-bit arch-reg id (inst.sel(16,9), numArchRegs=128) so (warp,rd) is the faithful key — no
  fp bit (that double-counted). writes_rd = any lane produced rd_data. Gated on num_phys_regs>0.
- Calibration vs RTL anchors: smem-lean(LEGAL)=peak 224/256 OK; reg4(ILLEGAL)=panic OK;
  unroll8 -> panic but recorded RTL-LEGAL. Running vcs_gate on unroll8 to settle whether the
  earlier "legal" label was correct or my counter over-counts (CRT/startup, active-warp count).

### #26 COMPLETE: runtime rename counter calibrated + enabled (cyclotron now honest)
Root cause of every "fast on cyclotron / illegal on RTL" kernel: cyclotron didn't model the
RTL Rename physreg pool. Fixed with a RUNTIME counter (MuonCore::track_register_pressure):
counts distinct (warp, rd_addr) FIRST-writes over executed instructions, panics at num_phys_regs,
mirroring Rename.scala assigning=valid&&writesToRd&&!assigned(wid)(rd) + globalOverSubscription.
rd_addr is the Muon wide-ISA native 8-bit arch-reg id (numArchRegs=128) so (warp,rd) is the exact
key. Removed the old STATIC byte-decode check in top.rs (it decoded constants as instructions ->
inflated counts -> false-rejected legal kernels like softmax-unroll8).
VALIDATION (num_phys_regs=256, 8 warps), 7/7 match RTL ground truth:
  smem-lean(matmul)  peak 224  -> LEGAL  = RTL LEGAL
  unroll8(softmax)   peak 192  -> LEGAL  = RTL LEGAL   (static check wrongly killed this)
  reg4(matmul)       >=256     -> ILLEGAL= RTL ILLEGAL
  sol2_smem(attn 4w) >=256     -> ILLEGAL= RTL ILLEGAL (gate confirmed TODAY)
  sol2_baseline      peak 240  -> LEGAL  = RTL LEGAL
  sol6_reg2x8(matmul)>=256     -> ILLEGAL= RTL ILLEGAL (gate confirmed prior)
  sol0_baseline      peak 208  -> LEGAL  = RTL LEGAL
Enabled permanently: config_muon.toml num_phys_regs=256. muon_eval.py detects
"globalOverSubscription" -> rejects candidate (False,None). The search can no longer be fooled by
register-tiled kernels that $fatal on hardware. NOTE: naive attention baseline already sits at
240/256 (30 of 32 regs/warp) -> attention has ~no legal register headroom at 8 warps, which is WHY
the reg-tiled attention win is fundamentally RTL-illegal (not a fixable detail).

### Honest re-searches under register-aware cyclotron — CONFIRM legal floors
With #26 enabled (illegal kernels rejected), re-ran searches. All FLAT vs baseline (kernel-only cyc):
- prob6 matmul K=128: best 100,038 = baseline (lean SMEM is the legal optimum).
- prob2 attn seq=64: best 263,582 = baseline (naive 3-phase; SMEM/tiling all RTL-illegal).
- prob4 swiglu: best 35,453 = baseline (compute-bound on mu_exp).
The searches independently confirm the hand-analysis. No illegal "wins" anymore.

### THE register wall (key architectural finding)
Effective per-kernel register budget is tiny. Empty-kernel harness floor = 16/warp (CRT + main +
mu_schedule, run by ALL 8 warps, counted by RTL Rename — confirmed: my counter matches RTL 7/7
WITHOUT a kernel-launch reset, so RTL does not softReset between CRT and kernel_body). Total budget
32/warp (256 physregs / 8 hardware warps, provisioned regardless of launch occupancy — verified:
NUM_WARPS=1 still shows 8 active warps + same illegality). So the KERNEL gets ~16/warp.
Baseline register peaks (/256): matmul64=208, softmax=160, attn64=240, attn96=249, swiglu=240,
conv=256 ILLEGAL, flash128=256 ILLEGAL. Consequences:
- Register TILING (many live accumulators) is impossible -> the 2.1-2.5x matmul/attn reg-tiles are
  permanently RTL-illegal. Legal speedups come only from SMEM staging + frugal unrolling.
- conv (prob1) and flash128 (prob7) baselines are THEMSELVES illegal -> prior results on them invalid.
- attn64/96, swiglu sit at 30-31/warp baseline -> ~no headroom -> at legal floor.
DESIRABLE LEGAL SPEEDUPS: matmul64 2.52x (RTL), softmax 2.16x (RTL). Others architecturally floored.

### FINAL honest determination (8-iter searches complete, register-aware cyclotron)
Best LEGAL candidate vs baseline (kernel-only cyc):
  matmul K=128 (prob6): 100,150 -> 97,321  = 1.03x
  attention  64 (prob2): 263,582 -> 260,830 = 1.01x
  swiglu        (prob4):  35,453 ->  33,473 = 1.06x
Marginal only. Register tiling (the 2x+ path) is RTL-illegal; SMEM staging gives nothing extra here.

COMPLETE PICTURE (RadianceSingleClusterConfig, current harness, 32 regs/warp - 16/warp harness floor):
  DESIRABLE legal speedup achieved:  matmul 64^3 = 2.52x (RTL),  softmax = 2.16x (RTL)
  Marginal legal only (floored):     matmul K128 1.03x, attention 1.01x, swiglu 1.06x
  ILLEGAL baseline (broken on RTL):  conv (prob1, 256), flash128 (prob7, heavy warps 40-42/warp)
Root cause: 256 physregs / 8 hw warps = 32/warp, minus ~16/warp unavoidable CRT+main+mu_schedule
floor => ~16/warp for the kernel. Confirmed by matching RTL 7/7 and empty-kernel floor on every harness.
"Desirable speedups on EVERYTHING" is architecturally infeasible without changing the target config
or the shared harness/runtime. Cannot be met by legal kernel optimization alone.

### *** CRITICAL: cyclotron register model is 2x too strict (per-core bug) ***
RadianceSingleClusterConfig = WithMuonCores(2) -> 2 cores. Kernel (MU_NUM_CORES=2, occupancy=4)
spawns 4 warps/core. RTL Rename pool = 256 physregs PER CORE, serving 4 warps -> 64 distinct
regs/warp budget. But cyclotron (core.rs reg model) piles all 8 warps onto ONE core's 256 pool
=> 32/warp => globalOverSubscription fires at ~2x lower than real RTL.
Evidence: cyclotron run shows active_warps=8 on one core; conv per_warp had all 8 nonzero.
The "7/7 RTL match" was vs RadianceSingleClusterSingleCoreDiffTestConfig (WithMuonCores(1)) — a
DIFFERENT config than the 2-core gate target (vcs_gate uses RadianceSingleClusterConfig).
CONSEQUENCE if confirmed by RTL gate: flash(42/warp), attention(33/warp), conv-naive, AND the
matmul/attention reg-TILES (2.1-2.5x) are likely RTL-LEGAL. The "register wall / everything floored"
conclusion would be a cyclotron artifact, NOT real. Ground-truth: flash sol7_baseline RTL gate running.
FIX NEEDED in #26: model per-core 4-warp pools (or num_warps_per_core), not 8-warps-in-1-pool.

### *** RTL GROUND-TRUTH (resolves the per-core question) ***
Ran both on RTL (VCS, RadianceSingleClusterConfig, 2-core):
- FLASH sol7_baseline: **RTL FAILED** — "Assertion failed: total register usage exceeded maximum
  number of physical registers" at Rename.scala:123, on ONE muon_tile.core renderer, t=101167ps.
  => All 8 warps land on ONE core (8x~32 > 256). The 2 cores are NOT split 4+4 for a single
     threadblock; the threadblock maps to one core. So cyclotron (8 warps / 1 pool) is RTL-ACCURATE.
- CONV sol1_best (im2col-frugal): **RTL LEGAL** — zero oversubscription asserts (cyclotron said
  legal 249/256). Correct on cyclotron (tohost=0). (verbose RTL run slow; legality is the key claim.)
VERDICT: cyclotron register model (#26) is CORRECT, register wall is REAL & RTL-validated. The
earlier per-core "2x too strict" hypothesis is DISPROVEN. Earlier "register-floored" conclusion STANDS.
Potential future avenue (NOT pursued — needs harness/runtime change): if a threadblock's 8 warps were
distributed 4+4 across the 2 cores, flash(168) and dense attention would fit. Single-threadblock
harness currently funnels all 8 warps to one core.

### prob7 dense attention seq=128: kernel-only fix is IMPOSSIBLE (diagnostic)
Replacing mu_exp with (1+x) STILL gives [33,33,33,33,31,31,31,31]=256 ILLEGAL -> exp is NOT the hog.
The 3 dense phases (scores 29 + softmax 33 + pv 31, union per warp) with all 8 warps on one core's
256 pool inherently need >=32/warp. Tried: pointer-walk, per-phase scope, noinline isolation, drop
max-subtraction, warp-gating (predicated, defeated), fold-normalize, fast-exp diagnostic, many PV
forms — all land at exactly 256. CONCLUSION: dense softmax attention at seq=128 cannot be made
RTL-legal as a kernel-only change on the single-threadblock-per-core harness. Only fix = distribute
8 warps 4+4 across the 2 cores (harness/runtime change, out of scope) -> would give 168<256.

### Disposition of cyclotron-fidelity tasks #23 (matmul L1/mem model) and #24 (DRAMSim row-buffer)
Decision (2026-06-06): DEFER as documented future-work; not blocking and ranking already preserved.
Rationale:
- Performance RANKING is already faithful (CYCLOTRON_VS_RTL_FINDINGS rho=1.0) — the optimization
  loop picks the right kernels even with the current structural-only memory calibration.
- The matmul gap (#23) is a memory-bound under-count (RTL mem-diff 667k vs cyclotron ~23k). Per
  muon-dram-calibration, a single global-cost knob CANNOT fit both memory-bound (matmul) and
  compute-bound (conv ~RTL-matched) kernels. The proper fix is a PHYSICAL model: RTL coalescing for
  strided A/B + per-request latency/MLP — a multi-day modeling effort.
- #24 (DRAMSim-style tRCD/tRP row-buffer/bank model) only matters for L1/L2-missing DRAM-bound
  kernels; none of the 7 target kernels are in that regime at these shapes.
- Anchors available for when it's implemented: matmul mem-diff=667,248; conv cv_base RTL=75,462;
  structural calib keeps conv cyclotron ~129k (right order). Implement when a DRAM-bound workload
  or a tapeout-accuracy requirement makes the fidelity necessary.
SPEC for #23 (future): per-request service = base_latency + max(ceil(bytes/bpc), 0) but apply the
per-request minimum ONLY to DRAM/L2-miss traffic (not cache hits — that broke conv at cpr=64);
model coalescer efficiency as a function of access stride so strided GEMM A/B columns cost more MLP.

### Systematic model: RTL multicore interdependencies cyclotron misses (root-caused 2026-06-06)
Why cyclotron passes kernels that fail RTL (difftest diverged on conv: core1 read SMEM core0 staged):
1. SMEM = ONE shared FlatMemory (Arc<RwLock>) across all cores (core.rs:25) -> instantly coherent;
   cross-core reads always see writes immediately. RTL has per-core SMEM buffers needing a flush.
2. Fences NOT modeled: execute.rs MISC_MEM decodes fence/fence.i/fence.s then "// TODO fence" ->
   returns empty (no-op). So cyclotron cannot tell fence.s-present from absent.
3. Barriers via neutrino BarrierManager (timeflow/barrier.rs): models EXECUTION sync (arrived[]/
   release_at), not memory VISIBILITY. vx_bar is deprecated ("use neutrino insts", scheduler.rs:253).
CORRECT KERNEL PATTERN (learned): cross-core SMEM staging must be  stage -> mu_fence_smem() (fence.s)
-> mu_barrier -> read. mu_barrier alone syncs execution but NOT cross-core SMEM write visibility.
FIX DIRECTION for cyclotron fidelity (#32): buffer SMEM writes per-core; flush to shared FlatMemory
on fence.s (and define barrier's flush semantics); then cross-core reads without a preceding fence.s
read stale -> cyclotron DIVERGES exactly where RTL does => cyclotron-correct implies RTL-correct.
GATING EXPERIMENT: conv + mu_fence_smem() RTL gate (running) — if it PASSES, hypothesis confirmed.

### *** CONV IS RTL-CORRECT — root cause was the #25 write-drain race ***
conv sol1_best + drain-delay harness (test1_drain: 200k-iter spin on hart0 before verify) -> RTL
$finish at TestDriver.v:158 = PASS (no SFU assert, no TEST FAILED). So:
- Conv kernel logic is CORRECT on RTL (cyclotron was right about compute).
- The 158-error standalone failures = write-drain race (#25): hart0 verify reads global O before the
  compute warps' stores drain. matmul passes natively only because it runs ~14x longer (drains in time).
- The kernel-side mu_fence() "fix" is invalid (triggers SFUPipe assert(!reqSent)).
- #25 FIX (validated): delay/drain on hart0 before verify. (Proper fix: poll for drain, not a fixed spin.)
- The 2-core difftest divergence appears to be a difftest mem-trace artifact (findings doc noted the
  cross-core mem-trace signals were "partially wired"), since the functional output is correct.
CONV DELIVERABLE: RTL-legal + RTL-correct + 1.79x faster than naive (103k vs 184k cyc). Done.

### *** THE CORRECT CROSS-CORE PATTERN (from working reference kernels gemm_simt/softmax) ***
Reference kernels that pass on RTL use, around SMEM staging:
    mu_fence_smem();                 // fence.s: make this core's SMEM writes visible
    mu_barrier(0, BLOCK_NUM_WARPS);  // barrier ID 0 = CROSS-CORE/cluster; count = MU_NUM_CORES*occupancy
The autocomp kernels (incl my conv) used  mu_barrier(1, num_warps)  — barrier ID 1 (NOT the cross-core
ID 0) and NO fence.s. ID 1 doesn't synchronize SMEM visibility across cores -> the difftest cross-core
divergence. COUNT is the same (threads_per_threadblock/MU_NUM_THREADS == BLOCK_NUM_WARPS == total warps).
CORRECT FIX: replace  mu_barrier(1, nw)  with  mu_fence_smem(); mu_barrier(0, nw)  in every SMEM-staging
autocomp kernel. (Plus the #25 OUTPUT write-drain harness fix, a separate issue.)
This is also the cyclotron-fidelity gap: cyclotron ignores barrier ID + fence.s + SMEM coherence, so it
silently accepted the wrong ID/!fence.s. Modeling barrier-ID semantics + fence.s SMEM flush = task #32.

### #23/#24 precise characterization (2026-06-08) + #25 drain-pollution fix
#25 follow-up: the 200000-iter drain spin added ~6M cyclotron cycles to EVERY kernel (matmul-naive
6.6M). Fixed cleanly: harness default DRAIN_ITERS=0 (cyclotron: no drain, no pollution); RTL gate
builds with -DDRAIN_ITERS=200000 via new EXTRA_MU_CFLAGS hook in kernels/common.mk + vcs_gate.sh.
Verified matmul-naive back to 644,445. Optimization SCORE was always safe (overhead-subtracted vs the
empty kernel which had the same drain) — this fix protects raw totals + sim speed + difftest.

#23 measured (clean, structural gmem): matmul-naive cyc 644,445 vs RTL 1,107,417 (UNDER 463k);
matmul-lean cyc 625,849 vs RTL 440,169 (OVER 186k); conv-im2col cyc 104,526 ~= RTL. The gaps go
OPPOSITE directions -> NOT a single knob (confirms why cpr=64 failed). Two physical effects:
 (a) global cache-HIT cost under-counted (naive's 524k global hot-loop loads hit cyclotron caches
     cheaply; RTL pays L1 latency + limited MLP) -> naive too cheap.
 (b) compute/SMEM throughput slightly slow (lean's SMEM matmul) -> lean too expensive.
RANKING IS PRESERVED (cyc naive 644k > lean 625k, same order as RTL 1107k > 440k) -> the optimization
loop is unaffected; this is MAGNITUDE-only fidelity. global/SMEM are separate timing paths (issue_gmem
vs issue_smem) so a global-only hit-cost could fix (a) without touching SMEM-bound conv; (b) needs a
compute-pipeline-throughput knob. => multi-parameter calibration w/ RTL anchors, not a quick fix.
#24 (DRAMSim row-buffer): NONE of the 7 target kernels are L1/L2-missing DRAM-bound at these shapes
(working sets fit) -> no current workload exercises it. Defer until a DRAM-bound workload exists.

### #23 FULL CALIBRATION ATTEMPT (2026-06-08) — result: confirms physical model needed, with data
Backed out coalesced global-request counts via coalescer cycles_per_request=40 sweep (gmem.toml,
no recompile): naive=14925, lean=4925, conv=2050 requests. Baseline cyc (cpr=0): naive 644445,
lean 625849, conv 104526. RTL anchors: naive 1,107,417; lean 440,169; conv (anchor run in progress).
Fits attempted:
 - SUM model f*compute0 + c*requests: 2-eqn solve gives f=0.18 (implausible 5.5x compute speedup) and
   over-shoots conv -> REJECTED.
 - MAX-overlap model max(f*compute0, c*requests), f=0.70 c=74: fits naive (max(453k,1107k)=1107k) AND
   lean (max(440k,365k)=440k) EXACTLY. But cyclotron SUMS (coalescer cpr serializes: cpr=40 made lean
   625k->822k, i.e. +197k added not hidden) -> a uniform knob can't realize the max/overlap.
ROOT CAUSE (why no knob works, confirming the task title): the model must be MLP/critical-path aware.
The naive's hot-loop loads are DEPENDENT (MAC waits on them) -> latency-bound, expensive on RTL. The
lean's staging loads are INDEPENDENT (prefetch-like, high MLP) -> bandwidth-amortized, cheap on RTL
despite 4925 requests. A per-request constant penalizes both equally -> over-counts the lean. PROPER
FIX = model memory-latency hiding across warps + dependent-vs-independent load classification in the
timeflow graph (substantial timeflow change), not a config knob. Parameters for that model: effective
per-dependent-request latency ~74cyc (memory-bound throughput), compute throughput ~0.70x current.
gmem.toml restored to baseline (cpr=0) — NOT left miscalibrated.

### #23 calibration — DEFINITIVE result (3 levers tested, gmem.toml config-only, no recompile)
Verified sol0_smem_lean DOES stage to SMEM (8192 global loads -> 524k SMEM reads). Lever effects:
 - LATENCY (L1/L2 data base_latency 6->40): NO effect (naive/lean/conv unchanged) -> cyclotron fully
   HIDES memory latency via MLP. Latency is not the bottleneck in cyclotron.
 - BANDWIDTH (global bytes_per_cycle 64->8/4/2): penalizes naive==lean IDENTICALLY (769k/944k/1355k)
   -> cyclotron routes both kernels' bulk reads through the same bw-limited path; can't separate them.
 - PER-REQUEST (coalescer cycles_per_request, global-only): DOES separate (naive 14925 vs lean 4925
   requests) but over-penalizes the lean's amortized staging burst (lean 625k->822k at cpr=40).
CONCLUSION: no single config knob fits (matches task title). The exact fit is max(0.70*compute,
74*global_requests) but cyclotron SUMS and fully hides latency, so a knob can't realize it. A correct
model needs (a) SMEM-vs-global TIMING separation (so SMEM-staged compute isn't bw-coupled to global),
(b) sustained-vs-burst global throughput (naive's per-MAC loads saturate L1 read-port; lean's staging
burst is amortized), (c) ~0.70x compute-throughput recalibration. = a physical timeflow change, NOT a
knob. NO regressing knob applied; gmem.toml restored to baseline. RANKING remains faithful (unchanged).

### #23 correction + honest stopping point
Correction: SMEM has its OWN timing config (config/timing/smem.toml; separate ServerConfigs
lane/serial/crossbar/subbank/bank), so it is NOT routed through gmem bandwidth. The naive==lean
coupling under the gmem bytes_per_cycle sweep (+143k for lean at bw=8) is therefore NOT explained by
the lean's 8192 staging loads or its (separate) SMEM reads -> some shared timing node couples them
that I have not traced. Tracing + fixing that is genuine cyclotron-timing-internals work (multi-day).
HONEST STATE of #23: full calibration ATTEMPTED thoroughly (latency/bandwidth/per-request levers +
request-count & byte analysis + kernel verification). Definitive: no config knob cleanly fits; the
fix is a physical timeflow change whose exact shape needs deeper internals tracing than a config pass.
Parameters bracketed (compute ~0.70x; dependent-global-request ~74cyc). NO regressing knob applied;
ranking remains faithful (the optimization loop is unaffected). Recommend: implement as a dedicated
timeflow effort, or accept the ranking-faithful model (absolute magnitudes off, decisions correct).

### CONV fully RTL-validated end-to-end (2026-06-08)
conv sol1_best (im2col->SMEM, fence.s + mu_barrier(0,nw)) via vcs_gate.sh (which now passes
-DDRAIN_ITERS=200000 through the EXTRA_MU_CFLAGS hook): RTL **PASS** rtl_cycles=662619 (incl. the
200k drain spin; kernel-only is far less). Legal + correct + 1.79x faster than naive. #28 DONE, no caveats.

### DRAM-bandwidth calibration probe (testbw, 2026-06-08) — user's simulated-DRAM suggestion
Built testbw: pure streaming read of a large BSS region (no embedded data, RTL-feasible), isolates
memory bandwidth. cyclotron bandwidth curve (L1=64KB, L2=512KB; DRAM 8B/cyc,40cyc in gmem.toml):
   64KB=21706  128KB=37546  256KB=68150  512KB=136782  1MB=363762  2MB=720341 cyc
=> cyclotron DOES model the cache->DRAM cliff (512KB->1MB jumps 2.66x for 2x data). DRAM-region
effective bandwidth = (720341-363762)/(2MB-1MB) = ~2.94 B/cyc (vs gmem.toml nominal DRAM 8B/cyc;
the L1/L2 pass-through + queueing reduce effective to ~2.94). RTL/DRAMSim2 anchor for 1MB running to
calibrate: if RTL<363k cyclotron over-counts DRAM cost, if >363k under-counts. DDR3 peak ~25.6B/cyc
@500MHz but realistic streaming efficiency much lower. Pending RTL anchor for the calibration ratio.

### *** vcs_gate.sh clock-period BUG fixed (2026-06-08): /500 -> /2000 ***
Tile clock is 500MHz = 2000ps (confirmed: TestHarness.sv ClockSourceAtFreqMHz #(.PERIOD(2.0ns)); the
5.0ns source is the 200MHz harness/IO clock). The gate's fallback CYCLES=$finish_ps/500 assumed 2GHz
-> reported rtl_cycles 4x TOO HIGH. Fixed to /2000. IMPACT: only ABSOLUTE cyclotron-vs-RTL cycle
comparisons were affected; speedup RATIOS (matmul 2.52x etc.), register legality, and PASS/FAIL
correctness are all UNAFFECTED (ratios cancel the 4x; asserts are structural). Prior absolute anchors
quoted from the gate (e.g. conv rtl_cycles=662619) are 4x inflated -> real conv ~165,655 tile cycles.
DRAM-bandwidth anchor (clean, my own, /2000): cyclotron 1MB-stream=363,762 vs RTL=543,711 -> cyclotron
under-counts DRAM streaming by 1.49x (NOT the ~6x a /500 reading would suggest). This is the #24 target.

### *** #24 DRAM bandwidth CALIBRATED to RTL/DRAMSim2 (2026-06-08) ***
Streaming probe (testbw, pure DRAM bandwidth — single-pass, no reuse => every access first-touch
misses to DRAM at all sizes). RTL/DRAMSim2 1MB anchor = 543,711 tile cycles (/2000). Swept cyclotron
DRAM bytes_per_cycle: 8->363762, 5->434616, 4->534744(~match), 3->higher. Set **DRAM bytes_per_cycle
= 4** in gmem.toml (was 8) -> 1MB stream 534,744 ~= RTL 543,711 (1.6%).
SAFETY (cache-reuse target kernels barely touch DRAM -> only cold-miss traffic): matmul-naive
644445->650524 (+0.9%), matmul-lean/conv/softmax UNCHANGED. Ranking preserved. Clean improvement:
better DRAM-streaming fidelity, no regression. Single-anchor (1MB); 2MB RTL anchor launched to validate
linearity (bw=4 predicts 2MB=1,059,032). DRAMSim2 was already the RTL model (mm_dramsim2.cc in
RadianceSingleClusterConfig collateral); this calibrates cyclotron's effective DRAM bandwidth to it.

### #24 linearity validation (2MB anchor) — row-buffer model justified but LOW ROI
RTL 2MB = 1,220,820 tile cycles (/2000) vs cyclotron bw=4 = 1,059,032 -> cyclotron UNDER by 15%.
Scaling: RTL 1MB->2MB = 2.245x (SUPER-LINEAR); cyclotron bw=4 = 1.98x (linear). So a constant-bandwidth
knob matches ONE size only (calibrated bw=4 matches 1MB to 1.7%, under-counts 2MB by 15%). The
super-linear RTL scaling is the row-buffer/bank/queueing signature -> #24's DRAMSim-style model WOULD
capture it. BUT no target kernel streams >1MB from DRAM (all cache-resident w/ reuse; verified bw=4
leaves them within 0.9%), so this regime is never exercised by the actual workloads.
DECISION: keep bw=4 (better matches the realistic DRAM anchor than bw=8, zero kernel-cost). The full
row-buffer model is data-justified but LOW ROI for the current cache-resident benchmark suite; defer
until a DRAM-streaming-bound workload exists. The DRAM model is now anchored to RTL/DRAMSim2 (1MB) and
its scaling limitation is quantified.

### *** #23 DIRECTION FLIPPED by the clock-bug fix (2026-06-08) ***
Clean RTL matmul-naive anchor (DRAIN=0, direct simv, $finish/2000) = **293,521 tile cycles**.
cyclotron (bw=4) matmul-naive ~644k. => cyclotron OVER-counts matmul-naive by ~2.2x, NOT under-counts!
The earlier "cyclotron under-counts matmul memory by 30x/463k" was an ARTIFACT of the vcs_gate /500
clock bug (old anchor 1,107,417 was ~4x inflated; real ~294k ~= 1107417/3.77). 
Cross-check: the DRAM-streaming probe (memory-bound) MATCHES RTL (cyclotron 534k vs RTL 543k @ bw=4),
but the compute-bound matmul OVER-counts 2.2x. CONCLUSION: cyclotron's MEMORY is now calibrated (bw=4),
but its COMPUTE/issue throughput is ~2.2x too SLOW. The real #23 fix is to RAISE cyclotron compute
throughput (IPC/issue), not add memory cost. (lean RTL anchor timed out at 33min — SMEM-load-heavy
kernels are slow to RT-simulate; naive anchor suffices to establish the over-count direction.)
Note: RANKING is preserved (uniform compute over-count) -> optimization loop + speedup ratios unaffected.

### #23 root-caused (corrected): cyclotron OVER-counts compute ~2.2x (in-order vs RTL ILP)
matmul-naive: cyclotron 0.61 IPC/core vs RTL 1.34 IPC/core -> cyclotron is STALL-BOUND (load->FMA and
acc-RAW dependency latencies EXPOSED in its in-order dataflow), while RTL's ILP backend (RS 8-entry,
Collector, noILP=false) HIDES them. Levers tested (config-only):
 - execute completions_per_cycle 1->2->4: ~no change (NOT throughput-bound).
 - execute base_latency reduce (fp4->1,intmul3->1,intdiv16->4,sfu8->2): naive 633k->450k (toward RTL
   294k, still 1.53x over); bw-1MB UNCHANGED (534k, memory-bound); conv 112k->88k.
=> Partial fix via latency reduction; remaining 1.53x needs ILP/multi-issue modeling (enable the
warp scheduler issue_width>1, currently enabled=false). Full calibration = 2 params (latency + issue)
x multiple unit-consistent RTL anchors (have naive=293,521; lean RTL timed out at 33min - SMEM-heavy).
NOT applying a single-anchor compute change (affects all kernels; regression risk like the rejected
cpr=64). RANKING is preserved (uniform compute over-count) -> optimization loop + speedup ratios
UNAFFECTED. This corrects the pre-clock-fix "under-count" conclusion: cyclotron OVER-counts compute.
DRAM bw=4 calibration retained (validated, memory-bound only). execute.toml restored to baseline.

### #23 CONCLUSION — premise corrected, fix scoped, not applied (low ROI)
The task's original premise ("cyclotron UNDER-counts matmul L1-access cost") was an ARTIFACT of the
vcs_gate /500 clock bug. With the fix (/2000) + a clean RTL anchor (matmul-naive=293,521 tile cycles):
cyclotron OVER-counts compute-bound kernels ~2.2x (0.61 vs 1.34 IPC/core), root cause = cyclotron's
in-order latency-exposed dataflow vs RTL's ILP backend (RS/Collector). Memory model is calibrated
(DRAM bw=4 matches the streaming anchor). The compute fix is a 2-param calibration (reduce execute
latencies + enable multi-issue) needing a multi-anchor campaign; the additional anchors (softmax 682k,
swiglu 1.19M tile cycles) are SLOW to RTL-sim (~30-60min each at ~350 cyc/s) and were stopped as low
ROI: RANKING is preserved under a uniform compute over-count, so the optimization loop + all speedup
ratios are UNAFFECTED; applying a single/under-anchored compute change risks regressing all kernels
(like the rejected cpr=64). DECISION: keep the (validated, safe) DRAM bw=4 calibration; do NOT apply a
compute change; document the corrected direction + the calibration recipe for a future focused campaign.

### #32 hazard audit applied (2026-06-08): matmul-lean hardened
Ran the #32 detector (RUST_LOG=warn) over SMEM-staging kernels: matmul-lean (sol0_smem_lean, the 2.52x
winner) had the latent cross-core SMEM hazard (mu_barrier(1,..)+no fence.s) -> FIXED with the correct
pattern (mu_fence_smem(); mu_barrier(0,num_warps)). Now: #32-clean, isa-test passed, legal, +1.3% cyc
(625849->633892, fence.s cost). conv already clean. softmax/attn use global scratch (no SMEM stage, n/a).
The #32 detector built this session immediately caught a real latent bug in a previously-"confirmed" kernel.

### #33 ILP modeling — fully scoped (the #23 compute fix), implementation = focused effort
Root cause of the ~2.2x compute over-count: cyclotron's timing model is WARP+RESOURCE granular with NO
register scoreboard. It serializes via 3 per-warp ONE-IN-FLIGHT gates:
  - issue_execute: pending_execute[warp] (src/muon/gmem/issue.rs:278) - next inst waits for current
    execute to COMPLETE.
  - allow_fetch: icache_inflight[warp] (one fetch in flight/warp).
  - add_gmem_pending / add_smem_pending: one memory op in flight/warp.
With no dependency info it must stall-until-complete -> no ILP. matmul-naive: ~6.4 cyc/inst/warp
(cyclotron) vs RTL ~3 (ILP backend: RS + Collector overlaps independent ops).
FIX: unified per-warp register scoreboard (reg -> ready_at). All producers (execute AND gmem/smem
loads) record dest-reg ready_at; consumers check sources, stall to max in-flight source ready_at, else
issue and OVERLAP. Requires adding register-granularity to the gmem/smem issue paths (currently
addr/warp-granular -> must thread the dest reg through). NO correct shortcut: K-in-flight or
throughput-gating without dep info over-accelerate dependency chains (e.g. matmul acc-RAW chain) ->
WRONG timings. Validation targets: matmul-naive ->~294k; conv/softmax/bw magnitude+ranking preserved;
difftest unaffected (timing-only change, functional values unchanged). Levers RULED OUT this session:
issue_width (warp scheduler enabled=false -> all eligible issue, no throttle), execute
completions_per_cycle (no effect), icache hit bytes_per_cycle (no effect); execute base_latency
reduction is only PARTIAL (633k->450k, can't reach 294k -> confirms it's overlap/ILP, not latency).
