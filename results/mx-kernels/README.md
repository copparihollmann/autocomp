# MX-Gemmini optimized kernels

Each problem has `<name>_baseline.c` (starting kernel; defines `void solution(void)`)
and `<name>_optimized.c` (best autocomp-found variant, golden-checked vs the
baseline output on spike — bit-exact bf16 results).

Validated 2026-06-03/04. Sonnet 4.6 via Bedrock; total project spend ~$35.

| Problem | Shape | Baseline (cyc) | Optimized | Speedup |
|---|---|---|---|---|
| matmul_2 | fp8 64x128x128 | 673 | 124 | **5.43x** |
| matmul_0 | fp8 64x64x64 | 294 | 92 | **3.20x** |
| matmul_1 | fp8 32x32x32 | 96 | 70 | 1.37x |
| conv_0 | C2 H64 k8 OC32 | 8210 | 6226 | 1.32x |
| attention_0 | softmax(QK^T)V S=64 D=64 | 222010 | 175610 | 1.26x |
| matmul_3 | fp8 128x128x256 | 1277 | 1097 | 1.16x |
| flash-attn_0 | flash attn S=128 D=64 | 1343602 | 1343581 | 1.00x |
| matmul-tiled_0 | fp8 256x256x512 (outer-tiled) | 2016019 | 2016019 | 1.00x |
| matmul-tiled_1 | fp8 1024x768x768 (smolvla GEMM) | 16745669 | 16700492 | 1.00x |
| conv_1 | C2 H32 k4 OC32 | 15611 | 15576 | 1.00x |

Notes:
- The big tiled GEMMs + flash attention are dominated by CPU staging / online-softmax
  loops; shallow beam search (iters=4) found only marginal wins. Real speedups there
  need deeper search and/or restructuring CPU<->accelerator overlap.
- Flash attention (gemmini-mx-flash-attn): true K/V tiling + online softmax (running
  max/denominator), never materializes the SxS score matrix → supports long sequences.

Harnesses + golden: autocomp/harnesses/<prob_type>/.
Full artifacts: autocomp/output/mxpipe_<prob>_<id>_iters4/.
Rerun: `python -m autocomp.search.run_mx_pipeline --iterations 4 --num-plans 3 --beam 2`.
