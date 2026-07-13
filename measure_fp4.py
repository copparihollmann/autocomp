import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import fork_problem_gen
hw=GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64); backend=GemminiEvalBackend(hw)
fork_problem_gen.gen_fork_problem("matmul_tiled_fp4_128x128x512","matmul_fp4_128x128x512.h","fp4:e2m1","fork-mm-fp4-512",0)
prob=Prob("fork-mm-fp4-512",0); baseline=load_initial_code("gemmini",prob)
A_old="""  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in_hw) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }"""
A_new=A_old.replace("for (int k = 0; k < tiles_K; k++) {","for (int k = 0; k < tiles_K; k += 2) {\n      int batch = (tiles_K - k < 2) ? (tiles_K - k) : 2;").replace("sp_addr, DIM, DIM","sp_addr, batch * DIM, DIM")
B_old="""  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N / 2 + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }"""
B_new=B_old.replace("for (int j = 0; j < tiles_J; j++) {","for (int j = 0; j < tiles_J; j += 2) {\n      int batch = (tiles_J - j < 2) ? (tiles_J - j) : 2;").replace("sp_addr, DIM, DIM","sp_addr, batch * DIM, DIM")
print("present:",A_old in baseline,B_old in baseline)
legal=baseline.replace(A_old,A_new).replace(B_old,B_new)
stats=backend.evaluate_code(prob,[baseline,legal],"spike"); base=stats[0].get("latency")
for lab,s in zip(["baseline","LEGAL(batch2)"],stats):
    lat,corr=s.get("latency"),s.get("correct"); sp=f"{base/lat:.2f}x" if (lat and base) else "-"
    print(f"  {lab:18s} correct={corr!s:5s} latency={lat} speedup={sp}")
