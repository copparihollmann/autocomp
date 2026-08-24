# Radiance LLM inference kernels

This PR adds a set of LLM inference kernels for the Radiance tapeout-330 SoC, together with the measured
performance of each. The aim is to cover the operations our target models need on this hardware, and to
give every kernel a fixed, measured baseline so that later optimizations can be compared against a known
reference. These kernels are not claimed to be optimal. They are a correct, measured starting point.

Target models (abbreviations used throughout):
**TL** TinyLlama, **GM** Gemma-2 2B, **DS** DeepSeek-R1-Distill-Qwen, **SV** SmolVLA.

## What this PR adds
- Nine SIMT element-wise and normalization ops the models need that the repo did not have
  (`qkv_bias`, `geglu`, `logit_softcap`, `rmsnorm_gemma`, `embed_scale`, `gemma_4norm`, `layernorm`,
  `patch_embed`, `bias_add`).
- Flash attention at head-dim 64 with 8-way grouped-query attention (GQA) and causal masking
  (`flash_attention_mx_gqa`), plus Gemma soft-cap / sliding-window and fp6 variants.
- Fused kernels that run a SIMT stage inside the matmul: RMSNorm in the prologue (`rmsnorm_qkv_fused`),
  RoPE in the epilogue (`rope_qkv_fused`), and an activation-quantize epilogue (`simt_quant` standalone
  plus the fused `gemm_mxgemmini` requant path).
- GEMM, weight-stationary, and batched-decode drivers at real model shapes, in fp8, fp6, and fp4.
- Non-square SMEM-tile support in `gemm_mxgemmini/mxgemm_lib.hpp`, needed for batched decode where M and N
  differ. Square tiles are byte-identical to before.

---

## Methodology

**Hardware basis: one cluster.** All performance numbers are for a single cluster: one MX-Gemmini matrix
engine (a 16×16 = 256-PE mesh) and two Muon SIMT cores (8 warps × 16 lanes each). The simulator is
cycle-accurate Verilator on `RadianceSingleClusterConfig`. The full chip has two identical clusters
(`RadianceTapeoutSimConfig`), so whole-chip throughput is about 2× these figures and whole-chip
utilization is about the same, since each cluster runs an independent tile.

**Cycle counts.** Cycles cover the kernel work window; instruction launch, teardown, and any in-kernel
verification loop are excluded. For matmul kernels whose result is written by the mesh, the end of the
window is the last mesh instruction. For kernels whose result is written by the SIMT cores, the end is the
point where compute finishes and the store-out begins. The trace `cycle` field is the Muon core clock,
which runs at twice the mesh tile clock.

**Utilization, three views.** All peaks are read from the RTL, not assumed.
- **MX-Gemmini util** = matmul MACs / (peak × cycles). Peak MAC/cycle = numOutputs × 256 PEs, from
  `MxParameters.scala`: fp8 = 256, fp6 = 1024, fp4 = 1024.
- **SIMT util** = issue utilization = IPC / 2, since peak issue is 2 (one slot per core). This is low for
  matmul kernels, where the Muon cores only move operands for the mesh and do no matrix arithmetic, and it
  reflects real compute for the SIMT, attention, and fused kernels.
- **Whole-Radiance util** = total matmul FLOPs / ((MX peak FLOP + 64) × cycles), where MX peak FLOP =
  2 × MAC peak (512 fp8, 2048 fp6 and fp4) and the SIMT peak is 64 FLOP/cycle (bf16: 16 lanes × 2 cores ×
  2 FLOP per FMA; the FP pipe is bf16-native, fp32 is half at 8 lanes = 32).
- For kernels that use both engines, each engine is measured against its own peak, and whole-Radiance
  combines the two.

**IPC and lane efficiency**, both read from the RTL trace by `scripts/muon/lane_eff.py`:
- **IPC** = instructions / (last cycle − first cycle), summed over both cores (peak 2).
- **lane-eff** = mean active lanes / 16, i.e. how full the 16-wide SIMD datapath is per instruction.

**Correctness, reported as two separate things.**
- The **golden** column: each kernel is checked bit-exact against a hardware golden computed at the same
  precision (functional co-model, `tohost=0`). "exact" means the kernel reproduces what the mesh is meant
  to output for that datatype. It is not a claim that the datatype matches full precision. An fp4 kernel
  is "exact" when it matches the fp4 golden.
- The **Precision fidelity** table: how much the datatype itself costs, given as cosine similarity and
  relative error of the quantized result against an fp32 reference. This is where fp8, fp6, and fp4
  differ. A kernel can be bit-exact yet lose fidelity at low precision. For the attention kernels the mesh
  math is lossy, so fidelity is their correctness number.

**Speedup.** A single number, shown only where a fixed, named baseline exists: either a pre-existing repo
kernel or another kernel reported here. Where no such baseline exists, no speedup is claimed.

---

## Model coverage

| kernel(s) | op | TL | GM | DS | SV |
|---|---|:-:|:-:|:-:|:-:|
| `gemm_mxgemmini` (A) | Q/K/V/O and FFN projection GEMM | x | x | x | x |
| `gemm_mxgemmini_ws` (B) | weight-stationary projection | x | x | x | x |
| `gemv_batched_*` (D) | batched decode projection | x | x | x | |
| `flash_attention_mx_gqa` (C) | GQA and causal attention, d=64 | x | x | x | |
| `flash_attention_mx_gemma` | soft-cap and sliding-window attention | | x | | |
| `rmsnorm_qkv_fused`, `rope_qkv_fused` (E) | RMSNorm / RoPE fused into GEMM | x | x | x | |
| `simt_quant` + requant (E) | activation quantize (bf16 to fp8) | x | x | x | x |
| `qkv_bias` | QKV bias-add | | | x | |
| `geglu` | GeGLU (gelu-tanh) | | x | | x |
| `logit_softcap`, `rmsnorm_gemma`, `embed_scale`, `gemma_4norm` | Gemma norms, soft-cap, embed-scale | | x | | |
| `layernorm`, `patch_embed`, `bias_add` | LayerNorm, patch-embed, bias | | | | x |

## Precision fidelity vs fp32

Cosine similarity and relative error of the quantized result against an fp32 reference. The matmul kernels
are bit-exact against their same-precision golden (the "golden" column below reads "exact"), so this table
is what shows the cost of the datatype itself. fp4 is the fastest but loses the most fidelity.

| precision | GEMM (128×128×2048) | Attention (8-head GQA, d=64) |
|---|---|---|
| fp8 | cosine 0.99917, rel-err 4.07% | cosine 0.99859, rel-err 5.33% |
| fp6 | cosine 0.99706, rel-err 7.67% | cosine 0.99448, rel-err 10.55% |
| fp4 | cosine 0.98708, rel-err 16.07% | cosine 0.97074, rel-err 24.02% |

---

## Results

Utilization columns: **MX** is the matrix engine, **SIMT** is Muon issue utilization, **whole** is both
engines together (see Methodology). "golden" = bit-exact against the same-precision golden.

### A. GEMM primitives, M = N = 128, 128×128×128 tiles
| # | prec | K | RTL cyc | MX util | SIMT util | whole util | IPC | lane-eff | golden | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fp8 | 512 | 101,716 | 32.2% | 5.1% | 28.6% | 0.10 | 33.3% | exact | |
| 2 | fp8 | 2048 | 204,713 | 64.0% | 7.0% | 56.9% | 0.14 | 15.8% | exact | |
| 3 | fp8 | 5632 | 377,615 | 95.5% | 8.3% | 84.9% | 0.17 | 7.1% | exact | |
| 4 | fp6 | 2048 | 137,106 | 23.9% | 7.1% | 23.2% | 0.14 | 20.5% | exact | 1.49x vs #2 |
| 5 | fp4 | 2048 | 122,709 | 26.7% | 7.1% | 25.9% | 0.14 | 22.1% | exact | 1.67x vs #2 |
| 6 | fp4 | 5632 | 228,789 | 39.4% | 9.1% | 38.2% | 0.18 | 12.9% | exact | 1.65x vs #3 |

### B. Weight-stationary GEMM, M = 256, N = 64
| # | prec | shape | RTL cyc | MX util | SIMT util | whole util | IPC | lane-eff | DRAM wt reads | golden | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | fp8 | K=2048 | 216,514 | 60.5% | 8.7% | 53.8% | 0.18 | 22.4% | 1x | exact | 1.33x vs #8 |
| 8 | fp8 | K=2048 (re-stream) | 288,355 | 45.5% | 9.2% | 40.4% | 0.18 | 11.2% | 4x | exact | |
| 9 | fp4 | down-proj K=5632 | 293,134 | 30.7% | 9.6% | 29.8% | 0.19 | 10.3% | 1x | exact | |

"DRAM wt reads" is how many times the full weight matrix is read from DRAM over the kernel. Read-once (#7)
keeps the weights resident in the SPAD across all M rows (1x); the re-stream baseline (#8) reloads them per
M-block (4x). #9's fp8 counterpart does not run to completion on RTL, so no fp4-vs-fp8 speedup is claimed.

### C. Attention, `flash_attention_mx_gqa`, head-dim 64, fp8
8-way GQA head grouping with causal masking, flash online softmax. The d=64 matmuls do not fill the 16×16
mesh, so MX util is low and is not the limiting factor. The kernel is bound by the SFU latency for the
`exp()` in the softmax: the cores stall about 81% of cycles waiting on the SFU. The useful utilization
figure is the measured issue utilization of 19.1% at a lane-eff of 92.6% (the SIMD lanes are well packed;
the cores just cannot issue while waiting on the SFU).
| # | shape | RTL cyc | MX util | SIMT util | IPC | lane-eff | cosine | rel-err |
|---|---|---|---|---|---|---|---|---|
| 10 | 8-head GQA, causal | 495,661 | low (mesh not filled) | 19.1% | 0.38 | 92.6% | 0.99859 | 5.33% |

Occupancy 2 (4 warps) is the ceiling. At 6 or 8 warps the register file overflows and the kernel
deadlocks. Raising it further needs a vector `exp` unit in hardware, not a kernel change.

### D. Batched decode to MX-DMA, N = 128, K = 2048, cycles per token
Batch M decode rows through the mesh so each weight load is amortized over more rows.
| # | prec | M | RTL cyc | cyc/token | MX util | SIMT util | whole util | IPC | lane-eff | golden | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | fp8 | 32 | 122,899 | 3,840 | 26.7% | 8.7% | 23.7% | 0.17 | 16.4% | exact | |
| 12 | fp8 | 64 | 144,236 | 2,254 | 45.4% | 8.7% | 40.4% | 0.17 | 22.1% | exact | 1.70x vs #11 |
| 13 | fp8 | 128 | 216,321 | 1,690 | 60.6% | 8.5% | 53.9% | 0.17 | 23.7% | exact | 2.27x vs #11 |
| 14 | fp4 | 128 | 134,583 | 1,051 | 24.3% | 9.1% | 23.6% | 0.18 | 33.0% | exact | 3.65x vs #11 |

Decode is bound by weight bandwidth. Batching more rows raises MX util (28% to 61% at fp8) and lowers cost
per token. Speedups are per-token throughput against the M=32 baseline (#11). M=128 is the SPAD ceiling.
These kernels need the non-square-tile support.

### E. Fused kernels (SIMT stage inside the matmul)
A small matmul on the mesh with a SIMT element-wise stage fused into it. The element-wise stage dominates,
so MX util is modest and lane-eff and IPC carry the picture. No unfused single-kernel baseline is
committed, so no speedup is claimed.
| # | kernel | prec / shape | RTL cyc | MX util | SIMT util | whole util | IPC | lane-eff | golden |
|---|---|---|---|---|---|---|---|---|---|
| 15 | `rmsnorm_qkv_fused` | fp8 64×64×256 | 67,211 | 6.1% | 6.9% | 5.4% | 0.14 | 99.4% | exact |
| 16 | `rope_qkv_fused` | fp4 64×64×2048 | 106,147 | 7.7% | 6.5% | 7.5% | 0.13 | 29.4% | exact |
| 17 | requant epilogue | fp8 128×128×128 | 35,645 | 23.0% | 6.4% | 20.4% | 0.13 | 42.4% | exact |

The standalone `simt_quant` primitive, measured on its own, runs at lane-eff 91.5%, IPC 0.43. These fused
kernels run individually. Chaining several into one megakernel deadlocks on RTL, so fusion is per-op.

### F. SIMT coverage ops
Element-wise and normalization ops. They use no matrix engine, so MX util is 0; they are bound by memory
bandwidth, so a high lane-eff (a near-full SIMD datapath) is the ceiling. RTL cycles are the compute window
measured on Verilator, with the in-kernel verify loop excluded. All are bit-exact (golden = exact).
| kernel | op | RTL cyc | SIMT util | IPC | lane-eff |
|---|---|---|---|---|---|
| `qkv_bias` | QKV bias-add | 15,232 | 3.0% | 0.06 | 93.5% |
| `geglu` | GeGLU (gelu-tanh) | 7,656 | 5.5% | 0.11 | 87.0% |
| `logit_softcap` | logit soft-cap | 28,195 | 1.9% | 0.04 | 92.8% |
| `rmsnorm_gemma` | RMSNorm(1+w) | 17,315 | 3.1% | 0.06 | 91.7% |
| `embed_scale` | embed × sqrt(D) | 17,506 | 3.1% | 0.06 | 91.4% |
| `gemma_4norm` | 4-norm sandwich | 20,687 | 2.7% | 0.06 | 87.6% |
| `layernorm` | LayerNorm | 20,934 | 2.7% | 0.05 | 92.0% |
| `patch_embed` | patch/conv embed | 43,670 | 27.8% | 0.56 | 99.7% |
| `bias_add` | bias-add | 10,525 | 4.0% | 0.08 | 88.8% |

---

## Appendix A. How to read the results

- **GEMM (A).** MX util climbs with K because the fixed per-tile scale-load overhead is amortized over more
  K-tiles (at K=5632 the mesh runs about 95% inside the K-loop). The low-precision rows read low on util
  because the mesh is 4x denser (fp4 and fp6 do 1024 MAC/cycle) but the single Muon warp feeding it cannot
  keep up, so the mesh idles. The win is still real wall-clock time: fp4 is 1.65x faster than fp8
  at K=5632. 128×128 is the largest output tile the accumulator and SPAD staging allow, so deeper K is the
  only way to amortize further.
- **Weight-stationary (B).** Keeping the weight tile in the SPAD and streaming all M rows past it reads the
  weights from DRAM once instead of four times, which is 1.33x faster when weight bandwidth is the limit.
- **Attention (C).** Bound by SFU `exp` latency, not by the mesh or the lanes. Occupancy 2 is the register
  ceiling. See the table note.
- **Decode (D).** Bound by weight bandwidth. Batching more rows through the mesh amortizes each weight load,
  which is why fp4 at M=128 reaches 1,051 cycles per token.
- **Fused (E).** The SIMT stage dominates, so these are SIMT-bound. Low IPC (0.13 to 0.14) shows they wait
  on loads and the SFU, not on issue slots.
- **Coverage ops (F).** Memory-bandwidth-bound element-wise work. Lane-eff is high (87 to 99.7%), so the
  SIMD datapath is well used; the low IPC is the cores waiting on memory.

## Appendix B. Per-kernel notes
- `gemm_mxgemmini` (A, B, D): the projection and FFN matmul on the mesh, in fp8, fp6, and fp4, at several K.
- `flash_attention_mx_gqa` (C): flash attention, d=64, 8-way GQA, causal. Uses a thread-per-row softmax at
  occupancy 2 to fit the register file.
- `rmsnorm_qkv_fused`, `rope_qkv_fused`, `simt_quant` (E): fold a norm, a rotary embedding, or an
  activation quantize into the matmul so the element-wise pass overlaps the matmul.
- `qkv_bias`, `bias_add`: add a per-channel bias to a projection output.
- `geglu`: GeGLU activation. Needed a lower occupancy (4 to 2) to clear a back-pressure stall in the l0d
  (the per-core L0 data cache).
- `logit_softcap`: `c * tanh(x / c)` logit cap.
- `rmsnorm_gemma`, `embed_scale`, `gemma_4norm`: the Gemma-2 norm and embed-scale ops.
- `layernorm`: standard LayerNorm.
- `patch_embed`: vision patch embedding. Staged through the SPAD with a lane-to-patch remap for about 16x
  less weight traffic.

## Appendix C. Not-yet-complete kernels and caveats
- `flash_attention_mx_gemma` (Gemma soft-cap and sliding-window): runs to completion on RTL, but the
  mesh-written output cannot be read back on-device with current tooling (the co-model returns NaN for the
  mesh output, and the trace captures only SIMT stores). The soft-cap and sliding-window softmax logic is
  verified bit-exact on a mesh-free twin. Shipped for coverage; the numeric check waits on silicon or a
  host flow.
- `flash_attention_mx_fp6` (fp6 attention): does not run to completion. A final-tile store stalls in the
  l0d, the per-core L0 data cache. The fix needs the operands kept resident in the SPAD, which is a
  re-architecture. It is functionally correct in the co-model (cosine 0.99448, rel-err 10.55% vs fp32).
- On-chip behavior is silicon-faithful: the sim uses the tapeout Chisel modules at tapeout sizes (SMEM
  128 KB / 4-bank, l0d 4 KB, L2 512 KB, mesh dim 16). The compute-bound and structural results can be
  trusted.
- Off-chip DRAM is modeled as a fixed 40-cycle backing memory (one channel), not a DDR controller. So the
  absolute numbers for memory-bound kernels (decode, streaming) depend on that model, and being
  single-cluster they are optimistic (no second-cluster DRAM contention).
- The basis is one of two clusters, so whole-chip throughput is about 2x.
