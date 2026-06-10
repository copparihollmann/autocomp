# Autocomp-on-Muon: search tuning, produced files, and the cyclotron fidelity envelope

Durable reference for *why* the Muon autocomp search is configured the way it is, what files the
pipeline produces, and — most importantly — **what cyclotron can and cannot rank**, so future runs
don't chase wins the model is blind to. Companion to `MUON_SETUP.md` (how to run), `MUON_RESULTS.md`
(findings ledger), `scripts/muon/HEURISTICS.md` (auto-generated transform ledger), and
`chipyard/generators/radiance/cyclotron/MODELING_CHANGES.md` (the timing-model PR).

Last validated: 2026-06-09.

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
