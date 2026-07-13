import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
attention_gen.gen_attention_problem(128,64,prob_id=0); prob=Prob("gemmini-mx-attention",0)
base=load_initial_code("gemmini",prob)
# old 5-pass softmax block (exact)
old=base[base.index("  // ---- 2) P = softmax"):base.index("  // ---- 3) O = P @ V")]
new=r'''  // ---- 2) P = softmax(S1) rows on CPU; FUSED 3-pass (normalize folded into requant scale) ----
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];
    float maxv = -1e30f;
    for (int j = 0; j < ATT_S; j++) {                          // pass A: unpack bf16 + row max
      uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
      row[j] = bf16_to_f(b);
      if (row[j] > maxv) maxv = row[j];
    }
    float sum = 0.0f;
    float gmax_g[GROUPS_S];
    for (int g = 0; g < GROUPS_S; g++) gmax_g[g] = 0.0f;
    for (int j = 0; j < ATT_S; j++) {                          // pass B: exp + sum + per-32-group max
      float e = exp_f(row[j] - maxv);
      row[j] = e; sum += e;
      int g = j >> 5; float a = f_abs(e);
      if (a > gmax_g[g]) gmax_g[g] = a;
    }
    float inv_sum = 1.0f / sum;
    for (int g = 0; g < GROUPS_S; g++) {                       // pass C: scale + encode (normalize folded in)
      float gmax = gmax_g[g] * inv_sum;
      int e = 0;
      if (gmax > 0.0f) e = f_exp(gmax / 16.0f) + 1;
      float comb = inv_sum * pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++)
        P_q[i][j] = fp8_e4m3_rtz(row[j] * comb);
    }
  }

'''
assert old in base
fused=base.replace(old,new)
s=be.evaluate_code(prob,[base,fused],"spike"); bl=s[0].get("latency")
for lab,st in zip(["baseline","fused-softmax(3pass)"],s):
    lat,c=st.get("latency"),st.get("correct"); sp=f"{bl/lat:.3f}x" if (lat and bl) else "-"
    print(f"  {lab:22s} correct={c!s:5s} latency={lat} speedup={sp}")
