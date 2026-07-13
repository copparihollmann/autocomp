# Informed autocomp search — ready to run (NOT yet run; ~$3–8 Bedrock)

## Why this is "informed"
- **Seed = the optimal hand-derived kernel.** `attention_gen.py` now emits the **float-free integer
  I-BERT softmax** as the baseline, so `load_initial_code()` seeds the search with it — the LLM refines
  a good kernel instead of rediscovering it. (Same for flash/conv/fp6/fp4 once Phases 3–4 land.)
- **Cost is honest (no-FPU).** Run with `MX_NOFPU_COST=1`: `gemmini_eval._count_host_float_ops` penalizes
  any host float (the host Rocket has no FPU → float traps on RTL, but spike hides it). Default penalty
  `MX_FLOAT_PENALTY=1e8` per occurrence is *disqualifying* — a candidate that reintroduces float (e.g.
  swapping integer `iexp` back to `exp_f`) is rejected, even though it looks cheap on spike's FPU.
  Verified: integer kernel (0 float) keeps latency 1.10M; a float candidate (3 float ops) → +3e8.
- **Legality is auto-gated.** The faithful spike (tasks #34–37) rejects RTL-illegal cols=64 batched mvin
  and transposed-B layout, so illegal candidates never enter the beam.

## Invocation
```bash
cd /scratch/agustin/projects/autocomp
source /scratch/agustin/projects/chipyard-mx/autocomp-env.sh   # env + venv + $200 spend cap
export MX_NOFPU_COST=1                                          # honest no-FPU cost
python -m autocomp.search.run_mx_pipeline \
    --only attention,flash,attn-fp4,attn-fp6 \
    --iterations 3 --num-plans 2 --beam 2
```
Est. ~75–110 Sonnet calls → **$3–8** (well inside the ~$130 remaining of the $200 cap; `--reserve-usd 5`
auto-stops). Spike eval dominates wall-clock, not $.

## What it should search for (attention)
integer-exp polynomial degree / fixed-point `S_q`; integer encode fusion; QK/PV mvin batching (legal
batch≤3) + dual-port `mvin2` overlap. NOTE: host-softmax↔matmul-mvin **overlap** can't be credited by
spike's additive model — route those candidates to the VCS gate.

## Validation gate per winner
faithful spike (`Correct result` + legality) → `verify_truth` fp32 attention ref (within-5%) → VCS on the
1–2 best survivors (real no-FPU cycles; mandatory for overlap candidates).
