from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
attention_gen.gen_attention_problem(128,64,prob_id=0); prob=Prob("gemmini-mx-attention",0)
base=load_initial_code("gemmini",prob)
old=base[base.index("  // ---- 2) P = softmax"):base.index("  // ---- 3) O = P @ V")]
stub="""  // ---- 2) STUB (floor: no real softmax) ----
  for (int i = 0; i < ATT_S; i++) for (int j = 0; j < ATT_S; j++) P_q[i][j] = 1;
  for (int g = 0; g < GROUPS_S; g++) for (int i = 0; i < ATT_S; i++) P_scales[g][i] = 127;

"""
no_exp=old.replace("exp_f(row[j] - maxv)","(row[j]-maxv)")        # softmax minus exp cost
no_enc=old.replace("fp8_e4m3_rtz(row[j] * inv_scale)","(uint8_t)(row[j]*inv_scale)")  # minus fp8 encode cost
s=be.evaluate_code(prob,[base, base.replace(old,stub), base.replace(old,no_exp), base.replace(old,no_enc)],"spike")
for lab,st in zip(["full-softmax","STUB(matmul floor)","softmax w/o exp_f","softmax w/o fp8-encode"],s):
    print(f"  {lab:24s} latency={st.get('latency')} correct={st.get('correct')}")
