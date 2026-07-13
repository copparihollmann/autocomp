from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import attention_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); be=GemminiEvalBackend(hw)
attention_gen.gen_attention_problem(128,64,prob_id=0); prob=Prob("gemmini-mx-attention",0)
base=load_initial_code("gemmini",prob)
old=base[base.index("  // ---- 2) P = softmax"):base.index("  // ---- 3) O = P @ V")]
new=r'''  // ---- 2) softmax + fp8 requant: 4-lane unpack, no-abs gmax, normalize folded into scale ----
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];
    float maxv = -1e30f;
    for (int j = 0; j < ATT_S; j += 4) {                       // pass1: unpack 4 bf16/word + max
      uint64_t w = S1_hw[i][j / 4];
      for (int l = 0; l < 4; l++) {
        float v = bf16_to_f((uint16_t)((w >> (l * 16)) & 0xFFFF));
        row[j + l] = v; if (v > maxv) maxv = v;
      }
    }
    float sum = 0.0f;
    for (int j = 0; j < ATT_S; j++) { row[j] = exp_f(row[j] - maxv); sum += row[j]; }
    float inv_sum = 1.0f / sum;
    for (int g = 0; g < GROUPS_S; g++) {
      float gmax = 0.0f;                                        // exp >= 0 so no fabs
      for (int j = g * 32; j < (g + 1) * 32; j++) if (row[j] > gmax) gmax = row[j];
      gmax *= inv_sum;
      int e = 0;
      if (gmax > 0.0f) e = f_exp(gmax / 16.0f) + 1;
      float comb = inv_sum * pow2i(-e);                         // normalize folded in
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = fp8_e4m3_rtz(row[j] * comb);
    }
  }

'''
assert old in base
s=be.evaluate_code(prob,[base, base.replace(old,new)],"spike"); bl=s[0].get("latency")
for lab,st in zip(["baseline","opt2(unpack4+nofabs+fold)"],s):
    lat,c=st.get("latency"),st.get("correct"); sp=f"{bl/lat:.3f}x" if (lat and bl) else "-"
    print(f"  {lab:26s} correct={c!s:5s} latency={lat} speedup={sp}")
