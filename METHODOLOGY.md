# Measurement methodology — how every number is calculated
So any figure in OPTIMIZATION_LEDGER.md / RTL_DEVELOPER_NOTES.md can be traced to an exact definition, formula,
tool, and repro. If an RTL developer asks "how did you get X?", the answer is here.

## 0. The golden rules (read first)
- **EVERY REPORTED PERFORMANCE NUMBER COMES FROM VERILATOR OR VCS. NO EXCEPTIONS.** Cycles, speedups (×), util%,
  B/cyc, cyc/MAC, both-active% — all from RTL. Not IPC (gameable by unrolling cheap loops — see §4). **Not
  cyclotron.**
- **cyclotron is a QUICK CHEAP PRE-CHECK, used for exactly three things — NEVER as a reported perf number:**
  (a) **functional correctness** — `verify_body` tohost=0 / golden rel-err (it generated the goldens, so it's
  functionally exact); (b) a **coarse go/no-go pre-filter** before spending a 25-min RTL run; (c) **contrast** to
  show where RTL diverges (e.g. "cyclotron said 1.34×, RTL says 1.06×"). Its timing model mis-ranks
  memory/overlap/bank effects (§7), so a cyclotron *cycle count* or *speedup* is NEVER a result — it is a hint that
  must be RTL-confirmed. Every cyclotron figure in any doc must be labelled "cyclotron" and paired with its RTL number.
- **A cycle count is only valid if the run COMPLETED.** Verified by dmem store-count (§3), never by raw span —
  a killed/deadlocked sim produces a plausible truncated number. This trap cost us a false "1.26× persist win."
- **KNOWN EXCEPTION — the e2e cycle counts are cyclotron-FUNCTIONAL, not RTL, and are NOT performance results.**
  autocomp_tinyllama_e2e_real "8,487,951 cyc" (and the 3.8M/4M reduced configs) are cyclotron *functional* runs
  (no --timing) that prove the model RUNS CORRECTLY (tohost=0) — they are CORRECTNESS evidence, not a perf number.
  The full real-dim e2e does NOT retire on RTL (l0d/timing), so we have NO RTL cycle count for the whole model;
  a real e2e timing number requires per-layer RTL ×22 summed. Never quote the e2e cyclotron cycles as performance.

## 1. Hardware constants (the denominators)
Verified from RTL source, not docs:
| quantity | value | source |
|---|---|---|
| MX-Gemmini peak | **256 MAC/cyc (fp8)**, **512 MAC/cyc (fp6/fp4)** = 512/1024 flop/cyc | 16×16 systolic, MxParameters.scala/Mesh.scala |
| Muon SIMT FP peak | **32 flop/cyc (fp32) / 64 (fp16/bf16)** TOTAL both cores | MuonCore.scala: numCores=2, numWarps=8, numLanes=16, fpPipe=FPPipeParams(8,2) → 8 fp32 / 16 fp16 FMA-lanes per core × 2 cores × 2 flop/FMA |
| SMEM | 128 KB = 4 banks × 2048 rows × 16 B; `bank=(addr>>15)&3` (4×32KB), `subbank=(addr>>2)&15` (16×4B) | RadianceSharedMem.scala:320-328 |
| L2 | 512 KB | config |
| DRAM BW ceiling | **2.10 B/cyc** (RTL-MEASURED, not the docs' 4-8) | bw_17: 512KB / 250,033 cyc |
| DRAM latency | ~40 cyc (L2 outerLatencyCycles); l0d = 4 KB DIRECT-MAPPED (nSets=64,nWays=1,64B), **nMSHRs=4** (NOT 2 — the 2 was the L0i icache), no landing pads | RadianceConfigs.scala:82-88, TLNBDCache.scala:24,200 |
| clock crossing | 500 MHz (core) / 200 MHz (mem) | [[rtl-timing-model-what-we-optimize]] |
Note: MAC = 2 flop. "MAC/cyc" for the mesh, "flop/cyc" for SIMT — keep them distinct.

## 2. Total cycles (the primary metric)
- **Source:** the RTL profiler `Cycles:` line, or the trace-db max cycle at `$finish`.
- **Command:** Verilator `sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig` (NO `+verbose` —
  it dumps 190-370 MB and starves the machine; the `Cycles:` line is all that's needed).
- **net_kernel_cycles vs full-span:** two different clocks — `net_kernel_cycles` is drain-keyed (excludes the
  tohost drain spin, used for isolated GEMMs); `trace_max`/full-span includes the verify/tohost tail. THEY ARE
  NOT INTERCHANGEABLE. Only compare within the same family/metric. For A/B, set **DRAIN=0 on both sides** so the
  identical tail cancels → clean apples-to-apples.

## 3. Completeness gate (mandatory before crediting any cycle count)
A run is COMPLETE only if its **dmem store-count == the baseline's store-count** (same computation → same #
output stores). References measured this session: a complete fused TinyLlama LAYER writes **~24,349 stores**; a
complete single-tile FFN writes **~16K to out_raw**; a truncated/deadlocked run freezes at <2k (or a partial
fraction). THE TRAP: persist "350,926 cyc" wrote only 17,211/24,349 stores = 71% complete → it was a truncated
run that looked like 1.264×; the complete run is 457,900 = 0.973×. Also cross-check: a "faster" span BELOW the
pure-streaming floor for the same bytes is physically impossible = truncation (coal GEMV 166,591 < 250,033).

## 4. IPC — deliberately NOT used
IPC is gameable: unrolling a cheap loop raises instructions/cycle without doing more useful work. We report TOTAL
CYCLES for a fixed problem instead. [[muon-metric-total-cycles]]. (We do read per-cycle-bin instruction/PC density
as a diagnostic for the both-engines analyzer — §6 — but never as a headline metric.)

## 5. Utilization (three distinct kinds — don't conflate)
All are `achieved / peak`. achieved throughput = useful-work / cycles.
- **MX whole-kernel util** = (M·N·K / total_cycles) / peak_MAC_per_cyc. INCLUDES fixed overhead (scale-load,
  config, fences) → lower than mesh util. This is the AMORTIZATION metric (31% at K=512 → 80% at K=5632 fp8).
- **MX mesh util (in-K-loop)** = mesh-active cyc / K-loop cyc ≈ 95-100% at real K. Derived by decomposing
  `total = compute_at_peak + fixed_overhead`; the overhead residual clusters at ~60-90k cyc regardless of compute
  size → proves the mesh itself is at peak and the whole-kernel gap is fixed overhead, not mesh idle.
- **SIMT util** = (SIMT flop / total_cycles) / (32 or 64 flop/cyc). Note SIMT is only ~6% of machine FLOPs during
  a matmul (that's why "90% idle SIMT" is a duty-cycle artifact, not wasted compute).
- **Essential-FLOP util** (conservative headroom) = essential_model_FLOPs / (full SIMT peak) / total_cycles = 1-5%.
  util_calc.py. [[muon-utilization-conservative]]

## 6. both-engines-active% (the co-execution metric)
Fraction of cycles where the MX mesh AND SIMT lanes are BOTH doing useful work. Two estimators (report which):
- **Strict (both_engines_active.py):** separate the MANAGER warp (MX issue→drain windows) from WORKER warps
  (SIMT epilogue, detected by distinct-PC density per cycle-bin: real compute = 20-32 distinct PCs/bin, barrier-
  spin <10) → the true MX∩SIMT intersection. (hw_util_phased.py CANNOT do this — it defines the SIMT window as
  strictly AFTER the MX region, so it never sees warp-spec concurrency.)
- **Wall-clock:** (serial_span − overlap_span) / overlap_span = the SIMT work now hidden under the live mesh.
- **HARD BOUND:** both-active ≤ min(MX-work, SIMT-work)/span. On the FFN, MX-work≈17.5% of span → both-active
  capped ≤17.5% (banked layer hits 14% = 80% of ceiling). This is why co-execution can't be "50/50."

## 7. Latency & bandwidth
- **Effective bandwidth (B/cyc)** = bytes_moved / total_cycles. Decode GEMV: 512KB / 1,443,950 = 0.363 B/cyc.
  Ceiling: 512KB / 250,033 = 2.10 B/cyc. A kernel at 17% of ceiling is latency-bound, not BW-bound.
- **cyc/MAC** = total_cycles / (M·N·K). The move-bound reframe metric (∝ 2/T for tile size T → weight-move-bound).
- **Weight-move floor** = weight_bytes / DRAM_BW_ceiling. fp4 22 MB/layer / 2.10 = ~10.5M cyc/layer floor (was
  under-estimated when using the docs' 4-8 B/cyc).
- **cyclotron caveat:** under-counts global-mem ~30× and models l0d as non-blocking → its BW/latency numbers and
  its coalescing "wins" do NOT hold on RTL. [[muon-dram-calibration]]. RTL only for anything memory.

## 8. Correctness (calculated per kernel class)
- **GEMM/FFN/SIMT:** cyclotron + in-kernel `verify_body` → `tohost=0` = BIT-EXACT vs the mx_golden-generated
  golden. This is the gold standard. (A kernel that hardcodes code=0 gives a meaningless tohost=0 — verify it
  actually compares.) [[sim-workflow-tiers]]
- **★ CRITICAL — distinguish a real PASS from a TIMEOUT (2026-07-18).** cyclotron's `main()->Result` prints
  **`Error: 0` on a 10M-cycle TIMEOUT** (`Err(0)`) — which looks like a pass at a glance but is NOT one. A GENUINE
  pass prints BOTH `simulation finished after N cycles` AND `Cyclotron: isa-test passed with tohost=0`. Any
  "tohost=0" read that shows `Error: 0` WITHOUT those two lines is a TIMEOUT, not a pass. This bit us: fused
  multi-body MEGAKERNELS (the single-kernel e2e `autocomp_tinyllama_e2e_real`, whole-layer megakernels) DEADLOCK
  on cyclotron at a compiler warp-reconvergence sync (`nu.invoke.ri.pw.sync`) in layer-0 down_body — reproduced
  functional+timing, toy+real dims, rebuilt+committed ELFs, even at a 400M-cycle budget → `Err(0)` every time.
  So the earlier "e2e verified bit-exact tohost=0" for the FUSED megakernel was a timeout misread as a pass. What
  IS solid: individual ops / standalone GEMMs retire and pass (they print the two lines, e.g. gemm_fp6_k2048
  tohost=0 @383,278 cyc); per-op MULTI-LAUNCH layers retire on RTL (store-count-verified). The FUSED whole-network
  on-device execution is BLOCKED (warp-reconvergence deadlock — a compiler/runtime limit, NOT l0d, NOT barrier-
  duplication). Always confirm the two pass-lines; never accept a bare `Error: 0`.
- **MX-mesh FP (attention):** on-device verify is platform-blocked (co-model NaN-codes mesh FP; trace-db garbage)
  → correctness = Python golden **relative error** ||out−fp32ref||/||fp32ref||; fp8 attention 4-6% is in-band.
- **Weight-quant fidelity:** per-GEMM **Pearson correlation** corr(fp4_dequant, fp32) per matrix (real TinyLlama
  weights: 0.906 default → 0.951 with E8M0 scale_shift=2); cosine; median weight rel-err.
- **e2e generation:** logit **Pearson r** vs HF fp16 reference (r≳0.9 = quant-noise-only; r<0.7 = real bug),
  **Spearman** over top-50, **top-1/top-5 token agreement**, greedy longest-common-prefix.

## 9. Speedup (×)
`baseline_cycles / variant_cycles` — BOTH completion-verified (§3), BOTH same DRAIN, BOTH same
cycle-metric-family (§2). A speedup quoting mismatched metrics or an unverified span is discarded.

---
## Tools
- cyclotron: `chipyard/generators/radiance/cyclotron/target/release/cyclotron`, env `CYCLOTRON_MXGEMMINI=1`,
  `--binary-path <k>.radiance.elf --timing --log 0` (~3s). Weight preload: `CYCLOTRON_WEIGHTS=<hex_base>:<blob>`.
- Verilator: `sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig` (no license, parallel).
- `autocomp/scripts/muon/both_engines_active.py` (both-active%), `hw_util_phased.py` (per-phase util; NOT
  warp-spec-aware), `util_calc.py` (conservative essential-FLOP util), `mx_golden` (golden generator/oracle).
- Completeness: dmem store-count from the trace-db (or the profiler store counter).
