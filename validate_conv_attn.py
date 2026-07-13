from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import conv_gen, attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
ORIG={"gemmini-mx-conv":8213,"gemmini-mx-attention":824009}
conv_gen.gen_conv_problem(2,64,8,32,prob_id=0)
attention_gen.gen_attention_problem(128,64,prob_id=0)
for pt in ["gemmini-mx-conv","gemmini-mx-attention"]:
    prob=Prob(pt,0); body=load_initial_code("gemmini",prob)
    st=be.evaluate_code(prob,[body],"spike")[0]
    lat=st.get("latency"); print(f"{pt:22s} NEW baseline correct={st.get('correct')} latency={lat} vs orig {ORIG[pt]} -> {ORIG[pt]/lat:.2f}x" if lat else f"{pt} correct={st.get('correct')}")
