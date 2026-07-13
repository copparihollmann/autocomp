import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import conv_gen, attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)

def batchA(body, A, K, slot):
    old=f"""    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*){A}) + i * DIM * {K} + k * DIM),
                            {slot}, DIM, DIM);"""
    new=f"""    for (int k = 0; k < tiles_K; k += 2) {{
      int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
      gemmini_extended_mvin((void*)(((elem_t*){A}) + i * DIM * {K} + k * DIM),
                            {slot}, bb * DIM, DIM);
    }}"""
    assert old in body, f"A not found ({A})"; return body.replace(old,new)

def batchB(body, B, N, slot):
    old=f"""  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*){B}) + (k * DIM) * {N} + j * DIM),
                            {slot}, DIM, DIM);"""
    new=f"""  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j += 2) {{
      int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
      gemmini_extended_mvin((void*)(((elem_t*){B}) + (k * DIM) * {N} + j * DIM),
                            {slot}, bb * DIM, DIM);
    }}"""
    assert old in body, f"B not found ({B})"; return body.replace(old,new)

# ---- conv ----
conv_gen.gen_conv_problem(2,64,8,32,prob_id=0); prob=Prob("gemmini-mx-conv",0)
base=load_initial_code("gemmini",prob)
leg=batchA(base,"A_buf","MATMUL_K","a_base + (i * tiles_K + k) * DIM")
leg=batchB(leg,"B_in","MATMUL_N","b_base + (k * tiles_J + j) * DIM")
s=be.evaluate_code(prob,[base,leg],"spike"); bl=s[0].get("latency")
print(f"conv     baseline={bl} legal-batch2={s[1].get('latency')} correct={s[1].get('correct')} speedup={bl/s[1]['latency']:.2f}x" if s[1].get('latency') else f"conv legal correct={s[1].get('correct')}")

# ---- attention (QK: Q/KT ; PV: P_q/V) ----
attention_gen.gen_attention_problem(128,64,prob_id=0); prob=Prob("gemmini-mx-attention",0)
base=load_initial_code("gemmini",prob)
leg=batchA(base,"Q_in","ATT_D","a_base + (i * tiles_K + k) * DIM")
leg=batchB(leg,"KT_in","ATT_S","b_base + (k * tiles_J + j) * DIM")
leg=batchA(leg,"P_q","ATT_S","a_base + (i * tiles_K + k) * DIM")
leg=batchB(leg,"V_in","ATT_D","b_base + (k * tiles_J + j) * DIM")
s=be.evaluate_code(prob,[base,leg],"spike"); bl=s[0].get("latency")
print(f"attention baseline={bl} legal-batch2={s[1].get('latency')} correct={s[1].get('correct')} speedup={bl/s[1]['latency']:.2f}x" if s[1].get('latency') else f"attention legal correct={s[1].get('correct')}")
