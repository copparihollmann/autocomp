import pathlib
from autocomp.search.search import load_initial_code
from autocomp.search.prob import Prob
from autocomp.hw_config import GemminiHardwareConfig
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.gemmini.mx_pipeline import fork_problem_gen
hw = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
backend = GemminiEvalBackend(hw)
fork_problem_gen.gen_fork_problem("matmul_tiled_fp6_128x128x512","matmul_fp6_128x128x512.h","fp6:e3m2","fork-mm-fp6-512",0)
prob = Prob("fork-mm-fp6-512", 0)
baseline = load_initial_code("gemmini", prob)
A_old="""  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }"""
A_new="""  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k += 2) {
      int batch = (tiles_K - k < 2) ? (tiles_K - k) : 2;
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }"""
B_old="""  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * K_TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }"""
B_new="""  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j += 2) {
      int batch = (tiles_J - j < 2) ? (tiles_J - j) : 2;
      uint8_t *dram_ptr = (uint8_t *)B_in + k * K_TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }
  gemmini_fence();"""
print("loops present in baseline body:", A_old in baseline, B_old in baseline)
legal = baseline.replace(A_old,A_new).replace(B_old,B_new)
stats = backend.evaluate_code(prob, [baseline, legal], "spike")
base = stats[0].get("latency")
for lab,s in zip(["baseline","LEGAL(batch2,nofence)"], stats):
    lat,corr=s.get("latency"),s.get("correct"); sp=f"{base/lat:.2f}x" if (lat and base) else "-"
    print(f"  {lab:24s} correct={corr!s:5s} latency={lat} speedup={sp}")
