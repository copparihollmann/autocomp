"""Auto-iterate driver: generate MX-Gemmini problems (matmul / conv / attention)
and run autocomp over each, sequentially, under a budget.

This is the "prepare everything, then let it iterate" endpoint:
  - generates problems via mx_pipeline generators (golden = hardware-captured),
  - validates each baseline on spike before any LLM spend,
  - runs beam search (1 model, low candidate count) per problem,
  - hard-stops at the project budget cap (AUTOCOMP_SPEND_LIMIT_USD).

Run inside the autocomp env:
    python -m autocomp.search.run_mx_pipeline --generate-only      # $0
    python -m autocomp.search.run_mx_pipeline                      # spends
"""

import argparse
import pathlib
import random

from autocomp.common import logger, SOLS_DIR, cost
from autocomp.search.search import (
    create_backend_and_agents, load_initial_code, BeamSearchStrategy,
)
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig

from autocomp.backend.gemmini.mx_pipeline import matmul_gen, conv_gen, attention_gen, flash_attention_gen, fork_problem_gen

PROBLEMS = [
    # (family, prob_type, prob_id, generate-callable)
    ("matmul",    "gemmini-mx-matmul",    0, lambda: matmul_gen.gen_matmul_problem(64, 64, 64,   prob_id=0)),
    ("matmul",    "gemmini-mx-matmul",    1, lambda: matmul_gen.gen_matmul_problem(32, 32, 32,   prob_id=1)),
    ("matmul",    "gemmini-mx-matmul",    2, lambda: matmul_gen.gen_matmul_problem(64, 128, 128, prob_id=2)),
    ("matmul",    "gemmini-mx-matmul",    3, lambda: matmul_gen.gen_matmul_problem(128, 128, 256, prob_id=3)),
    ("conv",      "gemmini-mx-conv",      0, lambda: conv_gen.gen_conv_problem(2, 64, 8, 32, prob_id=0)),
    ("conv",      "gemmini-mx-conv",      1, lambda: conv_gen.gen_conv_problem(2, 32, 4, 32, prob_id=1)),
    ("attention", "gemmini-mx-attention", 0, lambda: attention_gen.gen_attention_problem(128, 64, prob_id=0)),
    # Outer-tiled (scratchpad-exceeding) shapes; gen captures gold from hardware.
    ("matmul",    "gemmini-mx-matmul-tiled", 0, lambda: matmul_gen.gen_matmul_problem_tiled(256, 256, 512,  prob_id=0)),
    ("matmul",    "gemmini-mx-matmul-tiled", 1, lambda: matmul_gen.gen_matmul_problem_tiled(1024, 768, 768, prob_id=1)),
    ("attention", "gemmini-mx-flash-attn",   0, lambda: flash_attention_gen.gen_flash_attention_problem(128, 64, prob_id=0)),
    # fp6 / fp4 microscaling (sub-byte). Distinct prob_types so the fp8 set is untouched.
    ("matmul",    "gemmini-mx-matmul-fp4", 0, lambda: matmul_gen.gen_matmul_problem(128, 128, 256, prob_type="gemmini-mx-matmul-fp4", prob_id=0, fmt="fp4:e2m1")),
    ("matmul",    "gemmini-mx-matmul-fp6", 0, lambda: matmul_gen.gen_matmul_problem(128, 128, 256, prob_type="gemmini-mx-matmul-fp6", prob_id=0, fmt="fp6:e3m2")),
    ("attention", "gemmini-mx-attn-fp4",   0, lambda: attention_gen.gen_attention_problem(128, 64, prob_type="gemmini-mx-attn-fp4", prob_id=0, fmt="fp4:e2m1")),
    ("attention", "gemmini-mx-attn-fp6",   0, lambda: attention_gen.gen_attention_problem(128, 64, prob_type="gemmini-mx-attn-fp6", prob_id=0, fmt="fp6:e3m2")),
    ("attention", "gemmini-mx-flash-fp4",  0, lambda: flash_attention_gen.gen_flash_attention_problem(128, 64, prob_type="gemmini-mx-flash-fp4", prob_id=0, fmt="fp4:e2m1")),
    ("attention", "gemmini-mx-flash-fp6",  0, lambda: flash_attention_gen.gen_flash_attention_problem(128, 64, prob_type="gemmini-mx-flash-fp6", prob_id=0, fmt="fp6:e3m2")),
    ("conv",      "gemmini-mx-conv-fp4",   0, lambda: conv_gen.gen_conv_problem(2, 32, 4, 32, prob_type="gemmini-mx-conv-fp4", prob_id=0, fmt="fp4:e2m1")),
    ("conv",      "gemmini-mx-conv-fp6",   0, lambda: conv_gen.gen_conv_problem(2, 32, 4, 32, prob_type="gemmini-mx-conv-fp6", prob_id=0, fmt="fp6:e3m2")),
    ("matmul",    "gemmini-mx-tiled-fp4",  0, lambda: matmul_gen.gen_matmul_problem_tiled(128, 128, 128, prob_type="gemmini-mx-tiled-fp4", prob_id=0, fmt="fp4:e2m1")),
    ("matmul",    "gemmini-mx-tiled-fp6",  0, lambda: matmul_gen.gen_matmul_problem_tiled(128, 128, 128, prob_type="gemmini-mx-tiled-fp6", prob_id=0, fmt="fp6:e3m2")),
    # Real smolvla GEMM shapes (from model2MLIR/workloads/smolvla/smolvla.mlir).
    ("matmul",    "smolvla-proj",  0, lambda: matmul_gen.gen_matmul_problem_tiled(1024, 768, 768,  prob_type="smolvla-proj", prob_id=0)),   # action-expert proj (60x in model)
    ("matmul",    "smolvla-mlp",   0, lambda: matmul_gen.gen_matmul_problem_tiled(1024, 768, 3072, prob_type="smolvla-mlp",  prob_id=0)),   # action-expert MLP up
    ("matmul",    "smolvla-act",   0, lambda: matmul_gen.gen_matmul_problem_tiled(128, 320, 320,   prob_type="smolvla-act",  prob_id=0)),   # action sub-block (small tiled -> searchable)
    # IRREGULAR-M real model shapes (model2MLIR audit): 61.6% of real matmuls have M%16!=0.
    # These were ungeneratable before the padded-loop_ws (pad_I/pad_J) generator. Sizes are
    # TRACTABLE proxies (the exact model dims like 1152^3 work but cost minutes/spike-eval -- too
    # slow for the search loop); each preserves the source model's IRREGULAR-M (the new coverage).
    # K%32==0 here (the K%32=16 cases 720/4304 are the separate Phase-4 path). Tagged by source.
    ("matmul",    "real-groot-patch",0, lambda: matmul_gen.gen_matmul_problem_tiled(41, 512, 512,  prob_type="real-groot-patch",prob_id=0)),  # groot patch grid, M=41 (M%16=9)
    ("matmul",    "real-smolvla-vis",0, lambda: matmul_gen.gen_matmul_problem_tiled(113, 256, 256, prob_type="real-smolvla-vis",prob_id=0)),  # smolvla vision seq, M=113 (M%16=1)
    ("matmul",    "real-smolvla-ffn",0, lambda: matmul_gen.gen_matmul_problem_tiled(50, 256, 512,  prob_type="real-smolvla-ffn",prob_id=0)),  # smolvla FFN, M=50 (M%16=2), large N
    ("matmul",    "real-llm-attn",   0, lambda: matmul_gen.gen_matmul_problem_tiled(8, 512, 512,   prob_type="real-llm-attn",   prob_id=0)),  # tiny_llama/molmoact attn, M=8 (M%16=8)
    ("matmul",    "real-pi05-ffn",   0, lambda: matmul_gen.gen_matmul_problem_tiled(256, 256, 512, prob_type="real-pi05-ffn",   prob_id=0)),  # pi05 FFN regime, clean M (real)
    ("matmul",    "real-bitvla",     0, lambda: matmul_gen.gen_matmul_problem(32, 256, 256,        prob_type="real-bitvla",     prob_id=0)),  # bitvla, M=32 (fits -> fitting path)
    # GEMV / M=1 autoregressive decode (213 instances in the models). M=1 underutilizes the 16x16
    # array, so the win is DMA-minimizing B-reuse -- a distinct regime for autocomp to search.
    ("matmul",    "real-gemv-1024",  0, lambda: matmul_gen.gen_matmul_problem_tiled(1, 1024, 1024, prob_type="real-gemv-1024",  prob_id=0)),  # xr0/tiny_llama decode, M=1 (B too big for fitting -> tiled)
    ("matmul",    "real-gemv-3072",  0, lambda: matmul_gen.gen_matmul_problem_tiled(1, 1024, 3072, prob_type="real-gemv-3072",  prob_id=0)),  # pi05 decode up-proj, M=1
    # Causal attention (autoregressive LLM/VLA). Masks future cols -> prob 0 in the integer softmax.
    ("attention", "real-attn-causal",0, lambda: attention_gen.gen_attention_problem(128, 64, prob_type="real-attn-causal", prob_id=0, causal=True)),
    # Multi-head / GQA / MQA attention (distinct per-head data). real models are 12-18 heads; these
    # are tractable proxies preserving the H-loop cost structure + GQA KV-reuse indexing for autocomp.
    ("attention", "real-mha",        0, lambda: attention_gen.gen_mha_attention_problem(64, 64, 4,        prob_type="real-mha",       prob_id=0)),  # 4-head MHA
    ("attention", "real-gqa",        0, lambda: attention_gen.gen_mha_attention_problem(64, 64, 4, H_kv=2, prob_type="real-gqa",       prob_id=0)),  # GQA (4 Q heads, 2 KV)
    ("attention", "real-mha-causal", 0, lambda: attention_gen.gen_mha_attention_problem(64, 64, 4, causal=True, prob_type="real-mha-causal", prob_id=0)),  # causal 4-head
    # Rakanic-fork reference matmuls re-based on the fork's SHIPPED, independently
    # verified gold (spike + VCS bit-for-bit). prob_type prefix "fork-mm-" so --only fork-mm.
    ("matmul",    "fork-mm-fp8-256", 0, lambda: fork_problem_gen.gen_fork_problem("matmul_tiled_fp8_128x128x256","matmul_fp8_128x128x256.h","fp8:e4m3","fork-mm-fp8-256",0)),
    ("matmul",    "fork-mm-fp8-128", 0, lambda: fork_problem_gen.gen_fork_problem("matmul_tiled_fp8_128x128","matmul_fp8_128x128.h","fp8:e4m3","fork-mm-fp8-128",0)),
    ("matmul",    "fork-mm-fp8-64",  0, lambda: fork_problem_gen.gen_fork_problem("matmul_tiled_fp8_64x64","matmul_fp8_64x64.h","fp8:e4m3","fork-mm-fp8-64",0)),
    ("matmul",    "fork-mm-fp6-512", 0, lambda: fork_problem_gen.gen_fork_problem("matmul_tiled_fp6_128x128x512","matmul_fp6_128x128x512.h","fp6:e3m2","fork-mm-fp6-512",0)),
    ("matmul",    "fork-mm-fp4-512", 0, lambda: fork_problem_gen.gen_fork_problem("matmul_tiled_fp4_128x128x512","matmul_fp4_128x128x512.h","fp4:e2m1","fork-mm-fp4-512",0)),
]


def _prior_winner_seeds(prob_type: str, prob_id: int) -> list[tuple[str, str]]:
    """Seed a re-run from the most recent prior winner for this problem (if any), so the search
    optimizes ON TOP of the best-known kernel instead of rediscovering it from the baseline."""
    import glob, os
    cands = glob.glob(f"output/mxpipe_{prob_type}_{prob_id}_iters*/best_candidate_so_far.c")
    if not cands:
        return []
    best = pathlib.Path(max(cands, key=os.path.getmtime))  # most-recently-written winner
    return [(best.read_text(), f"prior autocomp winner ({best.parent.name})")]


def optimize_problem(prob_type: str, prob_id: int, iterations: int, num_plans: int, beam: int,
                     plan_model: str = "aws::us.anthropic.claude-sonnet-4-6",
                     code_model: str = "", dropout_menu: float = 0.25):
    agent_name = "built:gemmini-mx"
    models = [plan_model]
    code_models = [code_model] if code_model else None  # None -> reuse plan model for implementation
    hw = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
    random.seed(1111)

    output_dir = pathlib.Path("output") / f"mxpipe_{prob_type}_{prob_id}_iters{iterations}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prob = Prob(prob_type, prob_id)
    initial_code = load_initial_code("gemmini", prob)
    extra_seeds = _prior_winner_seeds(prob_type, prob_id)
    eval_backend, agent, code_agent = create_backend_and_agents(
        "gemmini", agent_name, hw, prob, models, code_models,
        menu_strategy="one-shot", fine_grained_isa=True, example_rate=0.25,
        cache_dir=output_dir,
    )
    optimizer = BeamSearchStrategy(
        output_dir=output_dir, eval_backend=eval_backend, agent=agent,
        orig_code=initial_code, prob=prob, metric="latency", simulator="spike",
        give_score_feedback=1, give_util_feedback=0, give_hw_feedback=1,
        include_ancestors=False, plan_icl_examples=False, code_icl_examples=False,
        dropout_menu_options=dropout_menu, prevent_duplicate_level=0,
        translate_iters=0, translate_perf_threshold=15,
        translate_drop_original=True, translate_score=True,
        code_agent=code_agent, early_stop_iters=0, early_stop_threshold=1.0,
        continue_from="", use_edits=False,
        num_analyses=0, num_plan_candidates=num_plans, num_code_candidates=1,
        beam_size=beam, num_pairs_to_combine=0, num_gen_per_combine=0,
        trigger_exhaustive_threshold=1, trigger_exhaustive_iters=20,
        start_exhaustive_iters=0, reimplement_failed=True, skip_planning=False,
        extra_seed_codes=extra_seeds,
    )
    optimizer.optimize(iterations)
    best = output_dir / "best_candidate_so_far.c"
    return best if best.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-only", action="store_true", help="generate+validate problems, no LLM spend")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--num-plans", type=int, default=2)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--reserve-usd", type=float, default=5.0,
                    help="skip starting a new problem if remaining budget below this")
    ap.add_argument("--only", default="",
                    help="comma-separated substring filters; only matching prob_types run")
    ap.add_argument("--plan-model", default="aws::us.anthropic.claude-sonnet-4-6",
                    help="model for plan generation (pass a Haiku id to cut cost, e.g. aws::us.anthropic.claude-haiku-4-5-20251001)")
    ap.add_argument("--code-model", default="",
                    help="model for code implementation; empty = reuse --plan-model")
    ap.add_argument("--dropout-menu", type=float, default=0.25,
                    help="fraction of optimization-menu options randomly dropped per prompt (lower = keep the curated menu in-context)")
    args = ap.parse_args()

    logger.info("MX pipeline: generating problems (hardware-captured gold)...")
    filters = [f for f in args.only.split(",") if f]
    problems = [p for p in PROBLEMS if not filters or any(f in p[1] for f in filters)]
    logger.info("Selected %d/%d problems (filters: %s)", len(problems), len(PROBLEMS), filters or "none")
    generated = []
    for fam, ptype, pid, gen in problems:
        try:
            gen()
            generated.append((fam, ptype, pid))
            logger.info("  generated %s %s:%d", fam, ptype, pid)
        except Exception as e:
            logger.warning("  generation FAILED for %s %s:%d — %s", fam, ptype, pid, str(e)[:200])

    if args.generate_only:
        logger.info("Generate-only: %d/%d problems ready, no LLM spend.", len(generated), len(PROBLEMS))
        return

    lim = cost.spend_limit()
    for fam, ptype, pid in generated:
        total = cost.project_total().get("total_usd", 0.0)
        if lim is not None and total + args.reserve_usd >= lim:
            logger.warning("Stopping: project spend $%.2f near limit $%.2f", total, lim)
            break
        logger.info("=== Optimizing %s %s:%d (project $%.2f) ===", fam, ptype, pid, total)
        try:
            optimize_problem(ptype, pid, args.iterations, args.num_plans, args.beam,
                             plan_model=args.plan_model, code_model=args.code_model,
                             dropout_menu=args.dropout_menu)
        except cost.BudgetExceeded:
            logger.warning("Budget cap hit during %s:%d — stopping.", ptype, pid)
            break
        except Exception as e:
            logger.error("Optimization failed for %s:%d — %s", ptype, pid, str(e)[:300])

    total = cost.project_total().get("total_usd", 0.0)
    logger.info("MX pipeline done. Project lifetime spend: $%.4f", total)


if __name__ == "__main__":
    main()
