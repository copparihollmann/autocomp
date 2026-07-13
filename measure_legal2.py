import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import fork_problem_gen

hw = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
backend = GemminiEvalBackend(hw)
fork_problem_gen.gen_fork_problem("matmul_tiled_fp8_128x128","matmul_fp8_128x128.h","fp8:e4m3","fork-mm-fp8-128",0)
prob = Prob("fork-mm-fp8-128", 0)
baseline = load_initial_code("gemmini", prob)
# LEGAL2 body = fp8-256 winner body (validated structure) at batch2, applied to 128x128
body256 = pathlib.Path("output/mxpipe_fork-mm-fp8-256_0_iters6/best_candidate_so_far.c").read_text().replace("MAX_BLOCK_LEN","2")
stats = backend.evaluate_code(prob, [baseline, body256], "spike")
base = stats[0].get("latency")
for lab, s in zip(["baseline","LEGAL2(validated struct, batch2)"], stats):
    lat, corr = s.get("latency"), s.get("correct")
    sp = f"{base/lat:.2f}x" if (lat and base) else "-"
    print(f"  {lab:34s} correct={corr!s:5s} latency={lat} speedup={sp}")
