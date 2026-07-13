# Onboarding: Autocomp as a steerable experiment engine (any target)

You're standing up **autocomp** (ucb-bar LLM kernel optimizer) for a hardware target and
want more than kernels: **labeled transformation trajectories + a calibrated instruction
cost model → compiler heuristics.** This guide is the portable playbook from a working
MX-Gemmini build; reuse the target-agnostic parts, rebuild the target-specific ones.

## The architecture (target-agnostic)
Autocomp's value isn't the final kernels — it's that it explores a transformation space and
emits data mining can't: the **full distribution including losers** (the negative examples
heuristics need). Convergence point = a **static instruction cost model** as shared currency:
1. bridges sim event-counts → calibrated cycles,
2. becomes autocomp's reward (rank statically, sim only top-k — Ansor/MetaSchedule pruning),
3. is the feature axis heuristics rank transformations over.

Closed loop: `mined motifs → typed action menu → generate variants → cost-model scores
(sim survivors only) → trajectory (state,action,Δcost,accept?) → learned heuristic → policy`.

**Sequence:** (1) cost model first — calibrate vs the sim, validate it reproduces a measured
win; (2) trajectory ingest — gives the negatives immediately; (3) cost-as-reward shim;
(4) heuristic learner, gated by cross-target generalization.

## Reuse directly (portable; just point at your output dir / swap features)
- **`autocomp/common/cost.py`** — Bedrock token→USD tracking, lifetime spend ledger, **hard
  spend cap** (`AUTOCOMP_SPEND_LIMIT_USD` + `=stop`). Backend/model-agnostic. Watch live:
  `python -m autocomp.common.cost --total`. (We added this; per-call ledger is authoritative —
  `run_metrics.json` double-counts resumed runs.)
- **`autocomp/backend/gemmini/mx_pipeline/transform_log.py`** — mines every run's
  `generated-plans-iter-N/` + `eval-results-iter-N/` + `candidates-iter-N/` (the SAME layout
  for all autocomp backends) → `transform_ledger.jsonl`: one row per attempt with
  `(prob, action/strategy, parent_latency, latency, speedup, outcome ∈ improved/correct_no_gain/
  regressed/incorrect/compile_error)`. **The incorrect+compile_error rows are your negatives.**
  Port: adjust the eval-result JSON keys + strategy regex for your backend.
- **`mx_pipeline/profile.py`** — `// PROF <name>` markers → `read_cycles()` phase split. Port:
  your cycle-read intrinsic.
- **`mx_pipeline/cost_model.py`** — feature extractor + lstsq fit (`fit` = static counts;
  `fit_dynamic` = real sim trace counts). MECHANISM is portable; swap the instruction regex
  set + the trace source. Result for matmul: static R²=0.57 → **dynamic R²=0.79, ±44 cyc**.
- **The run-env pattern** (`/scratch/agustin/projects/chipyard-mx/autocomp-env.sh`): source
  toolchain env → source creds `.env` → export region + spend log + cap → activate venv.

## Rebuild for your target (the target-specific parts)
1. **Eval backend** — autocomp ships `trn/tpu/cuda/saturn/xnnpack/metal/gemmini`. If yours is
   one, you're mostly done; else write one like `backend/gemmini/gemmini_eval.py`
   (build → run sim → parse {compiled, correct, latency}). Point its chipyard/sim path constant
   at your checkout.
2. **Cost-model features** — your instruction set + costs. Gemmini: mvin ∝ rows·⌈cols/dim⌉ +
   DMA (overlappable if double-buffered); compute ≈ dim (systolic, pipelined); config
   serializing; **fence = full drain (expensive — why batching won)**. RVV: per-instr
   throughput × trip count with VL/tail. Get dynamic counts from your sim's trace.
3. **Trace source** — for `fit_dynamic`, find how your sim emits executed-instruction counts
   (spike: `--log-commits` + the model's `dprintf`s).
4. **Agent** — `python -m autocomp.agent_builder.run_agent_builder --agent-name <t>
   --source-dir <curated docs>` → `built:<t>`. Curate ISA/programming-model docs; EXCLUDE giant
   generated-data files. Override the default Opus model id to one you can actually invoke.
5. **Correctness oracle + kernel templates/golden** — per op.

## Operational setup (Bedrock)
- **Auth**: a Bedrock API **bearer token** (`AWS_BEARER_TOKEN_BEDROCK`) authenticates, but
  autocomp routed Claude through `anthropic.AnthropicBedrock` (SigV4, ignores bearer). FIX
  (`llm_utils.py`): when a bearer token is set without SigV4 keys, route Claude through the
  **boto3 Converse path** (it auto-detects the bearer token). Needs recent boto3.
- **Model id**: verify with `bedrock.list_inference_profiles` — the working Sonnet is
  `us.anthropic.claude-sonnet-4-6` (geographic profile; **no `-v1:0` suffix**; bare FM rejects
  on-demand). `us.` geographic = STANDARD price (no surcharge → multiplier 1.0). Opus had **no
  invocable profile** in our account (needs console model-access + a us-west-2 profile).
- **Venv isolation**: install autocomp in its OWN venv — its deps need pydantic≥2; chipyard's
  `hammer-vlsi` needs pydantic<2 (they can't coexist).

## Hard-won gotchas (the negatives — avoid these time-sinks)
- **Never trust autocomp's native score** — re-cost every variant in your own currency.
- **Baseline-as-gold trap**: if gold = the baseline's own output, a buggy kernel "passes"
  (self-consistent garbage). VERIFY the baseline against a TRUE reference (e.g. CPU fp32 matmul
  of dequantized inputs) at least once per shape. We shipped a tiled GEMM that read OOB for N≠K
  and only caught it later.
- **Don't run the accelerator kernel twice in one program** — accelerator state (scale mem,
  smem) leaks between runs → different results. Capture gold in a SINGLE run; bake as constants.
- **Sim is layout-sensitive**: spike MX results depend on fence count, buffer addresses, even
  binary content. For big kernels, dump output + compare against a stored gold file in Python
  (a `// GOLD_FILE:` marker the runner checks) instead of baking gold into the binary.
- **Reference generators may not be hardware-faithful** — match the authoritative recipe
  exactly or use hardware-captured gold. (golden_model.py only matched with the exact
  `generate_test_data.sh` flags; e4m3 max-finite is **240** (IEEE emax=7), not OCP 448.)
- **Baremetal has no libm** — provide libm-free `exp`/`frexp` etc.
- **Serial eval**: candidates share one test file/binary → parallel eval clobbers them. Run
  candidates sequentially (or give each a unique name) + per-candidate timeout for hangs.
- **Cost model**: static source counts are weak (a mvin in a loop counts once); calibrate from
  DYNAMIC executed counts. Report the error band; don't act on a margin thinner than it.

## Start here
1. `source <env>.sh` (toolchain + creds + spend cap + venv); `python -m autocomp.common.cost --total`.
2. Smoke a 1-iteration search on a tiny verified problem; confirm Bedrock + eval + golden work.
3. Run `transform_log.py` after any searches → ledger (incl. negatives) from day one.
4. Build the cost model: `cost_model.fit` (static), then `fit_dynamic` (sim trace); validate it
   reproduces a measured win.
5. Wire `cost_model.predict()` into the eval backend as a pre-ranker (sim top-k); then learner.
