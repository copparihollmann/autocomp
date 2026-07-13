import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import conv_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
conv_gen.gen_conv_problem(2,64,8,32,prob_id=0); prob=Prob("gemmini-mx-conv",0)
base=load_initial_code("gemmini",prob)

im_old="""  for (int pi = 0; pi < CONV_H / CONV_KSZ; pi++)
    for (int pj = 0; pj < PATCHES_X; pj++)
      for (int c = 0; c < CONV_C; c++)
        for (int dy = 0; dy < CONV_KSZ; dy++)
          for (int dx = 0; dx < CONV_KSZ; dx++)
            A_buf[pi * PATCHES_X + pj][c * CONV_KSZ * CONV_KSZ + dy * CONV_KSZ + dx] =
                X_in[c][pi * CONV_KSZ + dy][pj * CONV_KSZ + dx];"""
im_new="""  for (int pi = 0; pi < CONV_H / CONV_KSZ; pi++)
    for (int pj = 0; pj < PATCHES_X; pj++)
      for (int c = 0; c < CONV_C; c++)
        for (int dy = 0; dy < CONV_KSZ; dy++)
          memcpy(&A_buf[pi * PATCHES_X + pj][c * CONV_KSZ * CONV_KSZ + dy * CONV_KSZ],
                 &X_in[c][pi * CONV_KSZ + dy][pj * CONV_KSZ], CONV_KSZ * sizeof(elem_t));"""
A_old="""  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);"""
A_new="""  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k += 2) { int bb=(tiles_K-k<2)?(tiles_K-k):2;
      gemmini_extended_mvin((void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, bb*DIM, DIM); }"""
B_old="""  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, DIM, DIM);"""
B_new="""  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j += 2) { int bb=(tiles_J-j<2)?(tiles_J-j):2;
      gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, bb*DIM, DIM); }"""
for nm,old in [("im2col",im_old),("A",A_old),("B",B_old)]:
    print(f"  {nm} present: {old in base}")
v1=base.replace(im_old,im_new)                          # memcpy im2col only
v2=v1.replace(A_old,A_new).replace(B_old,B_new)         # + batch2 mvin
s=be.evaluate_code(prob,[base,v1,v2],"spike"); bl=s[0].get("latency")
for lab,st in zip(["baseline","memcpy-im2col","memcpy+batch2"],s):
    lat,c=st.get("latency"),st.get("correct"); sp=f"{bl/lat:.2f}x" if (lat and bl) else "-"
    print(f"  {lab:16s} correct={c!s:5s} latency={lat} speedup={sp}")
