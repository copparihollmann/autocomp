import pathlib
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import matmul_gen, conv_gen, attention_gen, flash_attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
GEN={
 "gemmini-mx-conv": lambda: conv_gen.gen_conv_problem(2,64,8,32,prob_id=0),
 "gemmini-mx-flash-attn": lambda: flash_attention_gen.gen_flash_attention_problem(128,64,prob_id=0),
 "gemmini-mx-attn-fp6": lambda: attention_gen.gen_attention_problem(128,64,prob_type="gemmini-mx-attn-fp6",prob_id=0,fmt="fp6:e3m2"),
 "gemmini-mx-flash-fp6": lambda: flash_attention_gen.gen_flash_attention_problem(128,64,prob_type="gemmini-mx-flash-fp6",prob_id=0,fmt="fp6:e3m2"),
}
DIRS={"gemmini-mx-conv":"output/mxpipe_gemmini-mx-conv_0_iters4",
 "gemmini-mx-flash-attn":"output/mxpipe_gemmini-mx-flash-attn_0_iters5",
 "gemmini-mx-attn-fp6":"output/mxpipe_gemmini-mx-attn-fp6_0_iters5",
 "gemmini-mx-flash-fp6":"output/mxpipe_gemmini-mx-flash-fp6_0_iters5"}
for pt,gen in GEN.items():
    gen(); prob=Prob(pt,0)
    win=pathlib.Path(DIRS[pt]+"/best_candidate_so_far.c")
    if not win.exists(): print(f"{pt}: no winner"); continue
    s=be.evaluate_code(prob,[win.read_text()],"spike")[0]
    print(f"{pt:24s} winner-on-FAITHFUL-spike: correct={s.get('correct')} latency={s.get('latency')}")
