from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
attention_gen.gen_attention_problem(128,64,prob_id=0); prob=Prob("gemmini-mx-attention",0)
base=load_initial_code("gemmini",prob)
def A(body,X,N):
    old=f"""      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*){X}) + i * DIM * {N} + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, DIM, DIM);"""
    new=f"""      for (int k = 0; k < tiles_K; k += 2) {{
        int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
        gemmini_extended_mvin((void*)(((elem_t*){X}) + i * DIM * {N} + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM);
      }}"""
    assert old in body, f"A {X}"; return body.replace(old,new)
def B(body,X,N):
    old=f"""    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*){X}) + (k * DIM) * {N} + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, DIM, DIM);"""
    new=f"""    for (int k = 0; k < tiles_K; k++)
      for (int j = 0; j < tiles_J; j += 2) {{
        int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
        gemmini_extended_mvin((void*)(((elem_t*){X}) + (k * DIM) * {N} + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM);
      }}"""
    assert old in body, f"B {X}"; return body.replace(old,new)
leg=A(base,"Q_in","ATT_D"); leg=B(leg,"KT_in","ATT_S"); leg=A(leg,"P_q","ATT_S"); leg=B(leg,"V_in","ATT_D")
s=be.evaluate_code(prob,[base,leg],"spike"); bl=s[0].get("latency"); ll=s[1].get("latency")
print(f"attention baseline={bl} legal-batch2={ll} correct={s[1].get('correct')} speedup={(bl/ll):.3f}x" if ll else f"attention correct={s[1].get('correct')}")
