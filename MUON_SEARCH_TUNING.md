# Autocomp-on-Muon: search tuning, produced files, and the cyclotron fidelity envelope

Durable reference for *why* the Muon autocomp search is configured the way it is, what files the
pipeline produces, and — most importantly — **what cyclotron can and cannot rank**, so future runs
don't chase wins the model is blind to. Companion to `MUON_SETUP.md` (how to run), `MUON_RESULTS.md`
(findings ledger), `scripts/muon/HEURISTICS.md` (auto-generated transform ledger), and
`chipyard/generators/radiance/cyclotron/MODELING_CHANGES.md` (the timing-model PR).

Last validated: 2026-06-10.

---

## Oracle fix (2026-06-10) — global-memory bandwidth model (Option 3)

**Problem.** cyclotron under-ranked SMEM-tiling (matmul SMEM = 1.21× net vs RTL's 2.5×) because global
loads hit the L0/L1 data nodes at **64 B/cyc** — effectively free — so the cache-resident naive matmul's
3.27 MB of global traffic never became the bottleneck (issue-bound, latency hidden by 8 warps).

**Fix (config-only, no Rust change).** Lower the global-cache **data-node bandwidth** to model real
HW's finite global-memory bandwidth: `scripts/muon/config/timing/gmem.toml`, all three
`[gmem.levels.data] bytes_per_cycle: 64 → 6`. This makes the naive matmul **global-bandwidth-bound**
(net cycles scale with traffic) while SMEM-tiled stays **compute-bound** (flat) — exactly RTL's behavior.

**Calibration (matmul naive/SMEM net ratio vs the global-BW knob):**
| bw (B/cyc) | naive_net | smem_net | ratio |
|---|---|---|---|
| 64 (old) | ~63k | ~52k | 1.21× |
| 16 | 57,082 | 42,888 | 1.33× |
| 8 | 82,038 | 42,623 | 1.92× |
| **6** | **103,838** | **42,717** | **2.43×** ✓ matches RTL ~2.5× |
| 4 | 142,501 | 42,596 | 3.35× |

**bw=6** picked to match the RTL-validated 2.5× matmul anchor. All baselines still PASS at bw=6
(timing-only change — correctness/difftest unaffected; no eval timeouts): matmul/conv/softmax/gelu fast,
attn64 931k / layer_norm 1.3M (memory-bound now = realistic). **Caveat:** calibrating to RTL bandwidth
is *faithful* — it reveals memory-class wins (SMEM tiling) but, on now-memory-bound kernels
(attn/layer_norm), may mask pure-compute tweaks that don't reduce traffic. That's correct iff it mirrors
RTL. (Pending: a clean RTL naive-vs-SMEM gate to confirm the exact target; re-running the search at bw=6
to confirm the cheap search now *discovers* SMEM-tiling on its own.)

---

## Expanded problem suite (2026-06-10) — model2MLIR shape/op coverage

The original 8 problems (0-7) were tiny tiles that didn't represent the dominant shapes/ops the
10 model2MLIR models actually run (contractions ~43% of ops but tall-skinny + large-K; layer_norm
759 ops + gelu 135 ops entirely missing; non-pow2 head dims; long seq). Added 5 runnable problems
(all baselines verified PASS + RTL-legal under `--timing`):

| # | op / shape | source | baseline cyc | autocomp-productive? |
|---|---|---|---|---|
| 8 | layer_norm 64×768 (+γ/β); mu_rsqrt (no sqrtf libcall) | smolvla ln 1×1024×768 | 4.95M | ✅ reduction (faithful) |
| 9 | gelu 64×512, sigmoid-approx `x*sigmoid(1.702x)` via mu_exp | smolvla/pi05 gelu | 2.75M | ✅ elementwise (faithful) |
| 10 | large-K matmul 64×64×**384** (A+B=192KB>SMEM → K-streaming) | smolvla 1024×768@768×768 | 688K | ⚠️ under-ranks SMEM |
| 11 | attention seq96 **head_dim=72** (non-pow2) | pi05 16×256×72 | 1.18M | ⚠️ |
| 13 | tall-skinny matmul **M=8** K=256 N=768 (underutilization) | tiny_llama 8×2048 | 3.49M | ⚠️ |

Mechanics: `scripts/muon/gen_harnesses.py` `PROBLEMS` dict → renders `test{N}.cpp`; per-problem
`gen_data.py` (FP-matched golden), `sol{N}_baseline.cpp`, `context{N}.md`. `mu_rsqrt` and `mu_exp`
live in the harness `extra` (no libm).

**Findings while building (both real, documented):**
- **prob12 (long-seq attention, seq>128) is register-INFEASIBLE.** seq=192/256 baselines (naive
  3-phase AND row-parallel, even with `#pragma GCC unroll 1`) all land at exactly **256 distinct
  regs** = the 8-warp pool limit → `globalOverSubscription`. seq=128 (prob7) is the ceiling. The
  register-distinct-count grows with loop trip count (more executed bodies → more distinct rd's);
  whether this over-counts vs RTL (which renames cyclically) is unverified. **prob7 already covers
  the flash/SMEM-cliff regime**, so prob12 was dropped from the active set.
- **prob10 at K=768 hit a cyclotron functional bug** — exactly one contiguous 128-output block
  (2 rows) is flat-wrong (fails even at 10% tolerance; golden f32-vs-f64 max_rel 3e-4, so golden is
  fine). Only at large K (K=256/384 pass; K=768 fails) → smells like a functional issue on the
  ~49K-element global arrays. Filed; prob10 uses K=384 (still forces K-streaming).

`gen_harnesses.py PROBLEMS` now = {0,1,2,3,4,5,8,9,10,11,13} (6/7 hand-added, 12 dropped).

---

## 0. TL;DR — the fidelity envelope (read this first)

cyclotron is an **issue-bound / instruction-count timing model**. It ranks faithfully:
- compute / FP-throughput optimizations,
- register tiling (and rejects register-oversubscription exactly as RTL `Rename.scala:123`),
- instruction-count reductions, loop restructuring, barrier/phase changes.

It is **blind to memory-hierarchy reuse** (the classic "stage to SMEM to cut global loads" win):
- A global load costs ≈ the same issue slot as a SMEM op, so trading many global loads for SMEM
  reuse is **neutral-to-negative** in the model, while it's a **~2.5× win on RTL**.
- **Evidence (2026-06-09):** matmul 64³ naive vs `sol0_smem_lean` — cyclotron 390,953 vs 396,232
  (**0.99×**, SMEM looks *slower*); RTL 842,854(naive, failed-but-ran) vs 255,719, and a clean
  drain=0 naive = 281,189 → SMEM ~2.5× faster (matches prior RTL-validated 2.52×).
- Root cause (profiled): both kernels are memory-queue/issue bound at ~390k; the phased SMEM kernel
  can't keep cyclotron's latency-bound pipeline (`max_inflight=15`) full during staging bursts, and
  carries *more* total memory instructions (19k global + 37k SMEM > 51k global). No single config
  knob fixes this without distorting absolutes ~3× and breaking conv/attn (the documented
  memory-bound-vs-compute-bound tension, MODELING_CHANGES §3). A proper fix = structural
  memory-bandwidth modeling (separate project).

**Consequence:** run autocomp for **conv + attention** (faithful) and **compute/register-class** matmul
changes. For the matmul SMEM-tiling win, don't search — use the known `sol0_smem_lean` and confirm with
an RTL gate. Real wins found by search (cyclotron-ranked, plausible): conv 1.23×, attn64 1.11×,
attn96 1.09×, attn128 1.80×.

---

## 1. Search configuration

Set in `autocomp/search/run_search_muon.py`, overridable via env (read at lines ~59-62):

| knob | value | env | rationale |
|------|-------|-----|-----------|
| iterations | 12 | `MUON_ITERS` | more depth; affordable since two-phase eval makes failures cheap |
| beam_size | 3 | `MUON_BEAM` | wider beam |
| plans/iter | 3 | `MUON_PLANS` | 3 plans × 3 impls = 9 candidates/iter |
| impls/plan | 3 | `MUON_CODES` | |
| models (plan+code) | Sonnet 4.6 | — | `code_models=None` → same model both phases (user choice; quality over token-cap dodging) |
| reimplement_failed | **True** | — | failed candidates retried with the eval diagnostic (Lever 3) |
| dropout_menu_options | 0.25 | — | |

`MUON_ITERS`/`MUON_BEAM` are exported in `muon.env`. Run: `bash run_all_muon.sh` or
`.venv/bin/python -m autocomp.search.run_search_muon <prob>` (problems 0=matmul, 1=conv, 2=attn64,
3=attn96, 7=attn128).

### Discovery levers applied (2026-06-09)
- **Lever 2 — agent win-list** (`agent_builder/.built/muon/`, read at instantiation, *no rebuild*):
  `optimization_menu.yaml` already carried the Muon win-list (register tiling, +16 SMEM-stride
  padding, separate load/compute phases, …); added one lesson — *stage only reused operands, never
  once-read data* (the conv im2col finding). `rules.yaml` already has the 256-physreg/8-warp cap,
  128-thread mapping, mu_barrier/mu_fence rules.
- **Lever 3 — informative failures** (`backend/muon/muon_eval.py`): every failure mode now returns a
  specific diagnostic (compile error / `INCORRECT: N lanes` / `RTL-ILLEGAL oversubscription` /
  `TOO SLOW`) stored on `stats["stderr"]` → `candidate.stderr` (search.py:545) → consumed by
  `reimplement_failed` (search.py:1409). Turns dead-ends into corrected retries.
- **Two-phase eval** (`muon_eval._run_one`): a functional-only run (no `--timing`, ~1s) is a **fast
  pre-filter** (catches compile errors, `globalOverSubscription`, and functionally-wrong kernels in
  ~1s instead of up to 180s); the timed run then runs **only for survivors** AND is **authoritative
  for correctness** — it re-checks `isa-test passed`. This is required: some failures are
  TIMING-DEPENDENT (cross-core SMEM race from `mu_barrier(1,…)` per-core instead of `(0,…)`
  cross-core; host-verify/write-drain) and pass functionally (instantly-coherent SMEM) but fail under
  timing. (A bug where correctness was taken only from the functional phase let such kernels false-pass
  — fixed; the timed verdict is final.)
- **Lever 1 — seeding: deprioritized.** Pre-flight showed hand-tuned seeds aren't *cyclotron*-faster
  than baselines (the SMEM blind spot), so seeding adds no value when cyclotron is the search oracle.

### Eval caps (`muon_eval.py` + `scripts/muon/config_muon.toml`)
- `FUNC_TIMEOUT=60` (functional gate wall), `SIM_TIMEOUT=180` (timed wall).
- cyclotron `[sim] timeout = 10_000_000` cycles (≈3× slowest legit baseline attn128 3.59M; self-
  terminates runaways). Lowered from 80M (which let pathological candidates burn ~12 min each).

---

## 2. Baselines of record (what the search starts from)

`load_initial_code` (search.py) reads `sols/muon/sol{N}_baseline.cpp`. Two were swapped because the
naïve/flash versions are **RTL-illegal** (register oversubscription), which our own register model now
rejects:
- **prob1 conv** → `sol1_best.cpp` (im2col→SMEM matmul, RTL-legal, 98,559 cyc). Naïve kept as
  `sol1_naive_illegal.cpp`.
- **prob7 attn128** → `sol7_rowparallel.cpp` (row-parallel, sup=224<256 legal, 3.59M cyc). Flash kept
  as `sol7_flash_illegal.cpp`.
- prob0/2/3 baselines unchanged (legal).

---

## 3. Files the pipeline produces

**Per-run output dir** `output/built:muon_muon_<prob>_beam_iters<N>_cyclotron/`:
- `candidates-iter-<i>/` — saved candidates (the **resume cache**: a present iter is reloaded, not
  recomputed — archive/delete the dir to force a fresh run).
- `metrics-iter-<i>.json`, `eval-results-iter-<i>/code_*_result.txt` (per-candidate verdict + the
  new failure diagnostic), `reimplemented-code-iter-<i>/`.
- `best_candidate_so_far.cpp`, `cost_live.json`, wandb run.

**Agent artifact** `autocomp/agent_builder/.built/muon/` (read at instantiation, edit-then-rerun, no
rebuild needed): `optimization_menu.yaml` (34 strategies), `rules.yaml` (general/planning/coding),
`isa_docs.md`, `architecture.md`, `code_examples.md`. Rebuild only to re-synthesize from
`agent_sources/muon/` via `run_agent_builder --agent-name muon`.

**Heuristics ledger**: `scripts/muon/refresh_results.sh` → `output/transform_ledger.jsonl` +
`HEURISTICS.md` (which transforms helped, auto-derived from runs).

---

## 4. Eval pipeline (correctness + latency)

`backend/muon/muon_eval.py`:
1. compile `kernel.radiance.elf` (LLVM_MUON);
2. **functional gate** (cyclotron, no `--timing`): `isa-test passed` (tohost ecall) /
   `case=N` (errors) / `globalOverSubscription` panic (RTL register legality, mirrors
   `Rename.scala:123`);
3. **timed run** (`--timing`) for `finished after N cycles`; latency = cycles − harness overhead
   (empty-kernel baseline, subtracted per-problem).
- Golden is baked into each `harnesses/muon/test{N}/data` (gen_data.py); FP compare with tolerance
  (RTL fdiv ~1 ULP; gold accumulates sequentially to match device order). Verdict ONLY via tohost
  ecall rs1 — device prints are unreliable.

---

## 5. RTL gate (final confirmation)

`scripts/muon/vcs_gate.sh <prob> <candidate.cpp>` → builds `kernel.soc.elf` (host.cpp launches the
GPU) → VCS `RadianceSingleClusterConfig` → PASS + `rtl_cycles = $finish_ps / 2000` (tile clock
500 MHz = 2000 ps; the `/500` 4× bug is fixed). Gate the per-target winner; for matmul, gate
`sol0_smem_lean` directly (the search can't surface it).
- **Gate gotchas (learned 2026-06-09):** drop `+verbose` (it makes a ~400k-cycle sim exceed the
  30-min timeout); the embedded cyclotron difftest co-sim is the real wall-time bottleneck (~30-40
  min); `tohost symbols not in ELF` and `cyclotron trace insert failed (non-fatal)` are harmless;
  `DRAIN_ITERS` (write-drain race fix) is baked into gate elfs but NOT cyclotron numbers, so absolute
  RTL-vs-cyclotron isn't directly comparable unless built drain=0.

---

## 6. Known limits & gotchas
- **cyclotron SMEM-tiling blind spot** — §0. Don't trust matmul SMEM rankings from cyclotron.
- **Bedrock daily-token cap** ≠ the $100 spend cap. The daily TPD cap (not $100) throttled the last
  attn128 run; the search self-recovers via exponential backoff (slower). Sequential per-problem loop
  so a throttle stall on one problem doesn't block others.
- **Resume cache** — a present `candidates-iter-*` dir is reloaded at $0; archive/delete to force fresh.
- **16-way SMEM bank conflict** — column stride a multiple of 16 floats serializes all lanes (looks
  like a hang under `--timing`); pad +16. **Cross-core SMEM** — `mu_fence_smem()` then
  `mu_barrier(0, total_warps)` (ID 0 = cross-core; ID 1 = per-core). **8-warp output map** — derive
  from `threads_per_threadblock`, never a literal warp count.
- **cyclotron absolute compute over-count ~1.34×** vs RTL (issue-bound model); ranking preserved for
  the faithful classes above.

## RTL confirmation of bw=6 (Phase A, 2026-06-11)

Built trace-free `RadianceSingleClusterFastConfig` (mirror of RadianceSingleClusterConfig,
`trace=false, profiler=false`) so RTL gates run WITHOUT the cyclotron trace-sqlite spam that made
the trace=true sim take ~90 min. Trace-free wall-time: ~25-28 min/kernel (700-840k RTL cycles),
spam=0, $finish reached, EXIT=0. Tractable spot-check tool; cyclotron@bw6 stays the everyday oracle.

Gated matmul naive (sol0_baseline) vs SMEM (sol0_smem_manual), same soc.elf, drain-matched:
  RTL:             naive=843556 cyc  smem=726584 cyc  total_ratio=1.161  ($finish_ps/2000, 500MHz)
  cyclotron@bw6:   naive=480918      smem=419797      total_ratio=1.146   (same .cpp kernels)
  cyclotron@bw64:  naive=390953      smem=381807      total_ratio=1.02    (under-predicts RTL!)

=> bw=6 reproduces the RTL total-cycle SMEM-vs-naive ratio to within 1.3% (1.146 vs 1.161),
   whereas bw=64 under-predicted it (1.02x). The oracle fix is RTL-validated on TOTAL cycles
   (the preferred metric; see memory "total cycles only").
   Caveat on the headline "2.42x": that was a NET (fixed-overhead-subtracted) compute-region ratio.
   End-to-end TOTAL speedup for this matmul shape is ~1.16x (rest is fixed host-launch/DRAM-fill
   overhead SMEM-tiling can't remove). Both framings are consistent; report TOTAL.

## Per-op faithfulness of bw=6 (Phase B, cyclotron, 2026-06-11)

faithfulness_bw.sh: baseline vs discovered-best at bw6 AND bw64 (total cycles, run_problem.sh).
  prob1 conv_patchembed: bw6 1.29x  | bw64 1.34x   -> compute win SURVIVES bw6 (faithful)
  prob5 softmax:         best fails to produce cycles at BOTH bws -> kernel broken, bw-independent
  prob8 layer_norm:      bw6 1.04x  | bw64 1.01x    -> ~no win either bw (nothing to mask)
  prob9 gelu:            bw6 1.00x  | bw64 1.00x    -> best == baseline (no opt found)
  matmul (prob0):        RTL-confirmed above
  attn (calib):          smem-attn TIMEOUT at BOTH bw6 and bw64 -> pre-existing pathology, NOT a bw6 artifact
=> No ranking flips between bw6 and bw64. bw=6 does not spuriously mask any detectable compute win;
   every divergence is either a surviving win (conv) or a bw-independent failure. bw=6 is faithful.
