import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import fork_problem_gen

CASES = [
    ("fork-mm-fp8-256", "matmul_tiled_fp8_128x128x256", "matmul_fp8_128x128x256.h", "mxpipe_fork-mm-fp8-256_0_iters6"),
    ("fork-mm-fp8-128", "matmul_tiled_fp8_128x128",     "matmul_fp8_128x128.h",     "mxpipe_fork-mm-fp8-128_0_iters6"),
]
hw = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
backend = GemminiEvalBackend(hw)
for prob_type, kernel, header, outdir in CASES:
    fork_problem_gen.gen_fork_problem(kernel, header, "fp8:e4m3", prob_type, 0)
    prob = Prob(prob_type, 0)
    baseline = load_initial_code("gemmini", prob)
    winner = pathlib.Path(f"output/{outdir}/best_candidate_so_far.c").read_text()
    cands = [baseline] + [winner.replace("MAX_BLOCK_LEN", str(b)) for b in (1,2,3)]
    labels = ["baseline"] + [f"batch{b}" for b in (1,2,3)]
    stats = backend.evaluate_code(prob, cands, "spike")
    base = stats[0].get("latency")
    print(f"\n===== {prob_type} (baseline={base}) =====")
    for lab, s in zip(labels, stats):
        lat, corr = s.get("latency"), s.get("correct")
        sp = f"{base/lat:.2f}x" if (lat and base) else "-"
        print(f"  {lab:10s} correct={corr!s:5s} latency={lat} speedup={sp}")
