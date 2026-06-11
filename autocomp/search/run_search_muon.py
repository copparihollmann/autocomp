"""Radiance Muon kernel optimization runner (cyclotron simulator).

    source muon.env && .venv/bin/python -m autocomp.search.run_search_muon [prob_id]

Problems: 0=matmul, 1=conv2d patch-embed, 2=attention(64), 3=attention(96/flash),
4=swiglu, 5=softmax. Verdict via baseline-as-gold harnesses + tohost; latency =
cyclotron --timing total cycles. $100 hard budget enforced by autocomp.common.cost
(AUTOCOMP_SPEND_LIMIT_USD=100, mode=stop).
"""
import sys
import pathlib
import random

from autocomp.common import logger
from autocomp.search.search import (
    create_backend_and_agents,
    load_initial_code,
    BeamSearchStrategy,
    ExhaustiveSearchStrategy,
)
from autocomp.search.prob import Prob
from autocomp.hw_config.muon_config import MuonHardwareConfig
from autocomp.hw_config import (
    CudaHardwareConfig,
    GemminiHardwareConfig,
    MetalHardwareConfig,
    SaturnHardwareConfig,
    TrnHardwareConfig,
    TpuHardwareConfig,
)


def main():
    # ------------------------------------------------------------------
    # Target & environment
    # ------------------------------------------------------------------
    backend_name = "muon"
    agent_name = "built:muon"
    simulator = "cyclotron"
    hw_config = MuonHardwareConfig()

    prob_type = "muon"
    prob_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    models = [
        # HYBRID (proven on the MX-Gemmini target): plan on cheap off-AWS Flash, code on the SOTA
        # coder. Flash proposes optimization directions; Qwen-480B writes the kernels.
        "gcp::gemini-3.5-flash",
        # "gemini-3.1-pro-preview",                 # Pro: stronger planner but ~6-10x pricier (thinking)
        # "aws::us.anthropic.claude-sonnet-4-6",    # Sonnet 4.6 (Bedrock; daily-token throttled)
    ]
    # Code-gen on Qwen3-Coder-480B (Bedrock, separate quota => unthrottled). SOTA coder; drove
    # 1.4-2.07x on MX-Gemmini where cheap models found nothing. (Muon caveat: cyclotron under-ranks
    # SMEM, so its wins show only where the oracle is faithful — conv/attention compute, reg-tiling.)
    code_models = ["aws::qwen.qwen3-coder-480b-a35b-v1:0"]

    # ------------------------------------------------------------------
    # Search (minimal smoke config)
    # ------------------------------------------------------------------
    search_strategy = "beam"
    metric = "latency"
    import os
    iterations = int(os.getenv("MUON_ITERS", 8))
    num_plan_candidates = int(os.getenv("MUON_PLANS", 3))
    num_code_candidates = int(os.getenv("MUON_CODES", 3))
    beam_size = int(os.getenv("MUON_BEAM", 2))
    dropout_menu_options = 0.25
    early_stop_iters = 0            # 0 = disabled
    early_stop_threshold = 1.0
    skip_planning = False
    continue_from = ""

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    use_edits = False
    reimplement_failed = True  # retry failed candidates using the muon_eval diagnostic
                               # (carried via candidate.stderr) instead of dropping them

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    translate_iters = 0
    translate_perf_threshold = 15
    translate_drop_original = True
    translate_score = True

    # ------------------------------------------------------------------
    # Built-agent options
    # ------------------------------------------------------------------
    menu_strategy = "one-shot"      # None (static menu) or "one-shot"
    fine_grained_isa = True
    example_rate = 0.25

    # ------------------------------------------------------------------
    # Advanced / rarely changed
    # ------------------------------------------------------------------
    give_score_feedback = 1
    give_util_feedback = 0
    give_hw_feedback = 0
    include_ancestors = False
    plan_icl_examples = False
    code_icl_examples = False
    num_analyses = 0
    num_pairs_to_combine = 0
    num_gen_per_combine = 0
    trigger_exhaustive_threshold = 1
    trigger_exhaustive_iters = 20
    start_exhaustive_iters = 0
    prevent_duplicate_level = 0
    random.seed(1111)

    # ------------------------------------------------------------------
    # Sanitize model names for filesystem
    # ------------------------------------------------------------------
    models = [m.replace("/", "_") for m in models]
    if code_models is not None:
        code_models = [m.replace("/", "_") for m in code_models]

    # ------------------------------------------------------------------
    # Build output directory & start logging
    # ------------------------------------------------------------------
    clean_agent_name = pathlib.Path(agent_name).name if "/" in agent_name else agent_name
    output_str = f"{clean_agent_name}_{prob_type}_{prob_id}_{search_strategy}_iters{iterations}"
    output_str += os.getenv("MUON_TAG", "")
    if simulator is not None:
        output_str += f"_{simulator}"
    output_dir = pathlib.Path("output") / output_str
    output_dir.mkdir(parents=True, exist_ok=True)

    import autocomp.common.my_logging
    autocomp.common.my_logging.move_log(output_dir, tag="search")
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Initialize and run
    # ------------------------------------------------------------------
    prob = Prob(prob_type, prob_id)
    initial_code = load_initial_code(backend_name, prob)
    eval_backend, agent, code_agent = create_backend_and_agents(
        backend_name, agent_name, hw_config, prob, models, code_models,
        menu_strategy=menu_strategy, fine_grained_isa=fine_grained_isa,
        example_rate=example_rate, cache_dir=output_dir,
    )

    common_kwargs = dict(
        output_dir=output_dir, eval_backend=eval_backend, agent=agent,
        orig_code=initial_code, prob=prob, metric=metric, simulator=simulator,
        give_score_feedback=give_score_feedback,
        give_util_feedback=give_util_feedback,
        give_hw_feedback=give_hw_feedback,
        include_ancestors=include_ancestors,
        plan_icl_examples=plan_icl_examples,
        code_icl_examples=code_icl_examples,
        dropout_menu_options=dropout_menu_options,
        prevent_duplicate_level=prevent_duplicate_level,
        translate_iters=translate_iters,
        translate_perf_threshold=translate_perf_threshold,
        translate_drop_original=translate_drop_original,
        translate_score=translate_score,
        code_agent=code_agent,
        early_stop_iters=early_stop_iters,
        early_stop_threshold=early_stop_threshold,
        continue_from=continue_from,
        use_edits=use_edits,
    )

    optimizer = BeamSearchStrategy(
        **common_kwargs,
        num_analyses=num_analyses,
        num_plan_candidates=num_plan_candidates,
        num_code_candidates=num_code_candidates,
        beam_size=beam_size,
        num_pairs_to_combine=num_pairs_to_combine,
        num_gen_per_combine=num_gen_per_combine,
        trigger_exhaustive_threshold=trigger_exhaustive_threshold,
        trigger_exhaustive_iters=trigger_exhaustive_iters,
        start_exhaustive_iters=start_exhaustive_iters,
        reimplement_failed=reimplement_failed,
        skip_planning=skip_planning,
    )

    optimizer.optimize(iterations)


if __name__ == "__main__":
    main()
