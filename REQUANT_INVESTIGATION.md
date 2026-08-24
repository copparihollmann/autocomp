# MX-Gemmini fp8 Requantizer — full investigation & verdict (tapeout-330)

**Status:** the fp8 output **requantizer** is the ONE operation that is wrong on the tapeout-330 Radiance
RTL. It is a **genuine RTL requant-datapath divergence from the bit-exact co-model** = the tapeout team's
flagged WIP. It is **NOT** a kernel/packing/readout/version bug (all four ruled out with evidence below).

**Bottom line for the model:** correctness is **NOT** blocked. The requantizer is a fused-epilogue
*optimization* (activation→fp8 for chaining fp8 GEMMs), replaceable by a SIMT bf16→fp8 quantize kernel
(same proven-correct class as RMSNorm). Cost of the workaround is *performance* (an extra GMEM round-trip),
not correctness. Everything else the model needs is RTL-correct.

---

## 1. What the requantizer is (SW + pipeline role)

- Converts a GEMM **accumulator (fp32/bf16) → fp8 output codes + a per-group output scale**, written into
  the scratchpad, so the result can feed the **next fp8 GEMM** without a separate quantize pass.
- Sits on the **activation** path (weights are quantized to fp8 offline/statically — the requantizer never
  touches them). In an MX (microscaling) pipeline, activations must be re-quantized to fp8 each forward
  pass; that is the requant's job.
- In our stack it is the `QUANT_OUTPUT=true` path of `mxgemm_lib.hpp` (`GemmConfig.QUANT_OUTPUT`), exercised
  by autocomp **test39**. The plain GEMMs use `QUANT_OUTPUT=false` (bf16 output) and are all correct.

## 2. The observed failure (RTL, authoritative)

Driven from Muon via `mxgemm<C>()` on the elaborated tapeout-330 RTL (Verilator), M=N=64, K=128, fp8:
- Output is **exactly ¼ correct (1024–1025 / 4096), ¾ zeros**, uniformly scattered across all 16×16 tiles
  and all rows/cols. Surviving nonzeros are **saturated fp8 magnitudes** (0x7E/0xFE/0x78/0xF8); golden is a
  normal spread. Pattern = **wrong output scale** → ¾ underflow to 0, survivors saturate.
- The **same kernel/config is bit-exact on the cyclotron co-model** (identical source → co-model right,
  RTL wrong).

Verification method (important): NOT the in-kernel tohost read-back (unreliable on this RTL — SIMT-store
visibility artifact). The authoritative check reconstructs `C_raw`'s **final DRAM bytes from the trace-db
`dmem` LOAD log** (`scratchpad/rq_dma_verify.py` dmem path), plus offline store-trace reconstruction
(`scratchpad/rq_diagnose.py`, `/tmp/rq_map.py`).

## 3. Hypotheses tested and RULED OUT (with evidence)

| # | Hypothesis (SW) | Test | Result |
|---|---|---|---|
| 1 | fp8 output is **packed/reordered** ("2 fp8 per uint16 cell" per `radiance_xcheck`) | trace transforms: identity 10.7%, transpose 0.5%, byteswap-in-uint16 0.3%, stride-{1,2,4}×phase all <0.5%, multiset(got)≠multiset(gold) | ❌ no transform recovers it → values genuinely wrong, not reordered |
| 2 | **partial move-out** (wrong SPAD_DEST / tile count) | nonzeros are uniform (52–75 per 16×16 tile, every tile) | ❌ not a missing-tile pattern |
| 3 | `config_st` output stride wrong | rq_fix: `config_st(1*OUT_ELEM_SIZE)` for QUANT | ❌ still 89% mismatch |
| 4 | **SIMT flat readout** reads the column-packed layout wrong | switch to Gemmini DMA move-out `k_MVOUT_SPAD` (`SIMT_GMEM_MOVE_OUT=false`, `autocomp_rq_dma`) | ❌ **DMA gives identical 25% correct / 75% zeros** → readout is NOT the cause; both readouts agree |
| 5 | **wrong submodule / ISA version** | RTL `GemminiISA.scala` vs kernel header opcodes | ❌ correctly paired (see §5) |
| 6 | untested lever: `gemmini_mx_load_scales` scale-SRAM addressing | opcode 27 **absent on our silicon** | ❌ moot — op doesn't exist here |

**Conclusion:** the defect is **upstream of readout, in the requant compute/scale datapath**,
config-invariant across every lever we control, and bit-exact on the co-model.

## 4. RTL perspective

- **`generators/gemmini/src/main/scala/gemmini/MxRequantizer.scala`** (568 lines) is present and real. It:
  - reads the accumulator (`mxacc_req`) or a SIMT-fed input (`requant_data_in_gpu`),
  - computes the output scale and writes fp8 to the scratchpad via **`spad_projected_data`** — a
    *"projected" (column-packed) layout* — with a `spad_deprojected_data` path to expand it,
  - drives `scaleMem_write` for the per-group scales; LUT ports `lut0/1/2_write`.
  - For fp8, `total_bits_per_element==8`; for sub-8-bit formats `gpu_addr >>= 1` (packing).
- **`GemminiISA.scala`** MX interface on our silicon: `CONFIG_SCALE_MEM=26`, `MVOUT_SPAD=23`,
  `ConfigMxQuantRs1` (fields: `quant_lut_update_granularity`, `mem_address`, loop bounds i/j/k),
  `CONFIG_EX` MX-format fields (out/wgt/act formats at bits [15:14]/[13:12]/[11:10]).
- The ¼-correct/¾-zero, saturate-on-survivor signature points at the **output-scale / lane / LUT-update
  path** inside `MxRequantizer` producing a wrong (too-large) scale for ¾ of groups. Pinpointing the exact
  register needs an RTL designer; it is a DUT-datapath question, and we cannot modify the DUT (tapeout-330
  fidelity — trace instrumentation only).

## 5. Version / ISA — CONFIRMED CORRECT (do not re-investigate)

| component | pin | MX interface |
|---|---|---|
| RTL `generators/gemmini` | `69a1c03` (v0.1a-693) | `CONFIG_SCALE_MEM=26`, `MVOUT_SPAD=23`; **no** `MX_READ_SMEM`/`MX_LOAD_SCALES` |
| RTL `generators/radiance` | `f2755a3` (tapeout-330-1) | — |
| kernel header `radiance-kernels/lib/mxgemmini` | `62c4f85` (heads/…-radiance-tapeout) | **same** opcode set as the RTL |

**Why the "golden examples" don't port:** `chipyard-mx/.../bareMetalC/radiance_xcheck_fp8_requant.c` is a
**spike-only** cross-check (its header: "run under `spike --extension=gemmini`"; purpose = *prove the
co-model == spike*). `matmul_tiled_*_requant.c` run on **Rocket+Gemmini**. Both use a **newer Gemmini fork**
with `k_MX_LOAD_SCALES=27`, `k_MX_READ_SMEM=28`, `k_MX_LOAD_LUT=29` — opcodes **absent on our tapeout
silicon**. So every "working example" only ever chains *spike ⇔ co-model* and would emit illegal opcodes on
our RTL. There is **no passing requant example on the tapeout-330 RTL** to copy from.

## 6. SW perspective — the move-out paths in `mxgemm_lib.hpp`

- `SIMT_GMEM_MOVE_OUT=true` (default): `copy_smem_to_gmem_simt<TILE_M_QUANT,TILE_N_QUANT,OUT_ELEM_SIZE>` —
  reads the scratchpad flat as contiguous bytes. For requant fp8 this reads the *projected* layout raw →
  ¾ garbage.
- `SIMT_GMEM_MOVE_OUT=false`: `copy_C_smem_to_gmem_dma_sync` → Gemmini `k_MVOUT_SPAD` DMA (de-projecting),
  carries `// TODO: DRAM stride is wrong for re-quantized output`. Tested (`autocomp_rq_dma`): same ¼/¾
  result → the stride TODO is a red herring for THIS bug; the bad data is present before readout.
- Both paths agree → SW readout is not the lever.

## 7. Impact on the full model — NOT in trouble

- **Weights → fp8**: offline/static. Unaffected.
- **All MX GEMMs (bf16 output)**: trace-verified correct on RTL (Q/K/V proj, attention Q·Kᵀ/P·V/O-proj,
  FFN up/gate/down).
- **Activations → fp8** (the requant's job): replace the broken hardware epilogue with a **SIMT bf16→fp8
  per-group quantize** kernel (read tile → per-group max → scale → e4m3 convert → write fp8+scale). This is
  the **same reduction+elementwise class as RMSNorm/softmax**, which are RTL-verified correct.
- **Net:** the full TinyLlama/Llama-2 forward pass can run **end-to-end correctly** on the silicon. The
  requant hardware bug is **bypassable in software**.
- **Cost:** the SIMT requant adds a GMEM round-trip vs the fused hardware epilogue → a *performance* hit,
  not a correctness one. When the RTL requant is fixed, swap back to the fused epilogue to reclaim it.

## 8. Escalation package for the RTL team

- Symptom: fp8 requant output on tapeout-330 RTL = **¼ correct / ¾ zero, survivors saturated**, uniform;
  **bit-exact on the co-model** with identical kernel/config.
- Localized to: **`MxRequantizer.scala`** output-scale / lane / LUT-update path (not readout — both SIMT and
  `MVOUT_SPAD` DMA agree; not packing; not version).
- Repro: `radiance-kernels/kernels/autocomp_rq_fix` (SIMT readout) and `autocomp_rq_dma` (DMA readout);
  data header `mxgemm.data.fp8.m64n64k64.h`-class (test39); verify with `scratchpad/rq_dma_verify.py`
  (dmem-load reconstruction) and `scratchpad/rq_diagnose.py`.

## 9. Reproduction index (files, scripts, commands)

- Build dirs: `radiance-kernels/kernels/autocomp_rq_fix` (SIMT), `.../autocomp_rq_dma` (DMA move-out).
- Kernel/lib: `kernels/gemm_mxgemmini/mxgemm_lib.hpp` (`GemmConfig`, `SIMT_GMEM_MOVE_OUT`,
  `copy_smem_to_gmem_simt`, `copy_C_smem_to_gmem_dma_sync`, `SPAD_DEST()`, `QUANT_LUT_UPDATE_GRANULARITY`).
- Verifiers (scratchpad): `rq_diagnose.py` (transform hypotheses), `rq_stride.py`, `rq_map.py` (tile map),
  `rq_dma_verify.py` (dmem-load + 0x60000000 reconstruction).
- Build+sim: `autocomp/scripts/muon/sweep_util_trace.sh` (build + Verilator trace-db); one RTL run ≈
  20–25 min. Verdict = trace-db `dmem`/`inst` reconstruction, NOT tohost.
- RTL: `chipyard/generators/gemmini/src/main/scala/gemmini/{MxRequantizer,GemminiISA,MxParameters}.scala`.

## 10. Memory
Captured in `memory/requant-rtl-datapath-divergence.md` (readout-ruled-out + version-confirmed) and the
`OPTIMIZATION_LEDGER.md` requant section. Do not re-run blind config probes on requant; escalate with §8.
