# Radiance tapeout-330 — findings for the RTL / hardware developers
Consolidated silicon/DUT-level issues surfaced while optimizing TinyLlama on Radiance. Every item is
RTL-observed (Verilator, tapeout-330) unless marked. Ranked by impact. Kernel-side/software issues are
tracked separately in OPTIMIZATION_LEDGER.md; THIS doc is only things that need HARDWARE or RTL-owner action.

Legend: [SEV] S1=blocks a whole capability class · S2=caps a major lever · S3=correctness/observability · S4=docs/model

**TRACEABILITY — how to answer RTL-team questions on the spot.** Every number here is reproducible:
- HOW each metric is defined/calculated (cycles, util, B/cyc, both-active%, completeness, correctness) →
  `METHODOLOGY.md` (same dir). If asked "how did you measure that?", it's there with formula + tool + constants.
- The RAW evidence per finding (exact kernel path, cycle counts, store-counts, DRAIN, matched A/B) →
  `OPTIMIZATION_LEDGER.md`, searchable by the kernel names cited below.
- TO REPRODUCE any RTL number: `sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig` on
  `<kernel>/kernel.soc.elf` (no `+verbose`), read the `Cycles:` line, and confirm completeness via dmem
  store-count == the baseline's (a killed sim gives a plausible truncated span — see METHODOLOGY §3).
- Deadlock findings below are evidenced by FROZEN instruction retirement (store-count stuck mid-run) while the
  RTL spins to the cycle cap — the trace_run.sqlite files are retained as evidence.

---

## Limitation-claim standard (applies to every bottleneck below)
A "hardware limitation" is a claim we're asking RTL developers to act on — so each one must carry, in its
`VERIFICATION:` block: (1) **direct evidence** (an RTL-source config value + a measured number, not just an
inference); (2) **the reasoning chain** — how we found it; (3) **ruled-out alternatives** — the other things that
could produce the same symptom and why each is excluded; (4) a **falsification/control** — an observation that
WOULD change the conclusion, ideally a natural control already in the data; (5) a **confidence** level. Claims
without a VERIFICATION block are provisional (marked ⚠) until triple-checked.

---

## S1 — l0d cache geometry (4 KB direct-mapped, no landing pads) — CLAIM CORRECTED after triple-check
⚠ **The original claim ("2 MSHRs + too few misses in flight = decode at 17%") was REFUTED by the adversarial
verification below (HIGH confidence). Do NOT hand the RTL team "add MSHRs" as the top ask — it's wrong.** Corrected
finding, source-grounded:
- **The l0d has 4 MSHRs, not 2** (the "2" was the L0i icache — a mis-attribution). It is a **4 KB DIRECT-MAPPED**
  cache (nSets=64, nWays=1, 64 B lines) — `RadianceConfigs.scala:82-88`.
- **The decode-GEMV 17%-of-BW is NOT MSHR starvation — it's CACHE THRASH from the kernel access pattern.** The
  `[N,K]` GEMV strides weight rows by exactly 4096 B = the l0d size → every row aliases the same 64 direct-mapped
  sets → **8.6× refetch amplification** (70,724 misses for ~5,300 distinct lines). In-flight occupancy averages
  ~30/core (not pinned at 4) — the l0d is *saturated moving redundant refetches*, not starved. It is **largely
  SOFTWARE-FIXABLE by coalescing** (the coalesced control `bw_17` hits the full 2.10 B/cyc through the *same* l0d).
- **What IS a genuine, source-grounded l0d limitation** (the real, smaller asks): (1) **no landing pads** —
  CONFIRMED (`TLNBDCache.scala:201` no-skid `assert(!resp.valid||tlIn.d.ready,"response must be ready!")` is live) and
  now **DIRECTLY OBSERVED to be what the coalesced/conflict-free GEMV dies on** (confidence upgraded MEDIUM-LOW → HIGH;
  see the 2026-07-17 follow-up VERIFICATION addendum below — a conflict-free padded `[N,K]` GEMV `$fatal`s on this exact
  assert, in the streaming loop, deterministically on 2 seeds + unseeded). (2) The
  **4 KB / 1-way associativity** is the thrash amplifier — arguably the higher-value HW ask than MSHR count. (3)
  At the coalesced ceiling occupancy ≈ 3.6 ≈ 4 MSHRs, so 4 MSHRs *may* cap the SIMT streaming ceiling (2.10) —
  MEDIUM confidence.
**ASK (corrected, priority order):** (1) more l0d **associativity/capacity** (4 KB 1-way is the thrash amplifier);
(2) **landing pads** (unblocks the coalesced/streaming deadlock class); (3) possibly more MSHRs (secondary). But
first: the 17% decode gap is mostly OURS to fix in software (coalesce the GEMV) — pending the deadlock resolution.
**WORKAROUND today:** MX-Gemmini DMA bypasses the l0d (batched-decode M=64 = 1.95×/token) — though part of that
edge may be L2-residency (256 KB ≤ 512 KB L2), not pure bypass.

> ### VERIFICATION (adversarial, 2026-07-17) — CLAIM SUBSTANTIALLY REFUTED / REFRAMED. Confidence: HIGH on the refutation.
> The specific mechanism as written ("decode-GEMV at 17% **because** the l0d has ~2 MSHRs / too few misses in
> flight") does **NOT** survive scrutiny. Two of its load-bearing facts are wrong, and the dominant cause of the
> 17% is a **kernel access-pattern × l0d-geometry cache-thrash**, not MSHR/landing-pad starvation. Details:
>
> **(1) RTL-SOURCE — the "~2 MSHRs" number is WRONG; it is 4.**
> - l0d = `L0dCacheConfig` (`generators/radiance/chipyard/RadianceConfigs.scala:82-88`): **`nMSHRs = 4`**
>   (set commit b2ba82d3, Kelly Tou, 2026-02-22 — committed, unmodified in the built Jul-13 Verilator binary),
>   `nSets = 64, nWays = 1, blockBytes = 64, rowBits = 512`. → the l0d is a **4 KB direct-mapped** cache, 64 B lines.
>   (The "2 MSHRs" is the **L0i** icache, `RadianceConfigs.scala:66-72`, `nMSHRs=2` — a mis-attribution.)
> - **Landing pads: CONFIRMED absent.** `MuonTile.scala:242` instantiates `TLULNBDCache(TLNBDCacheParams(...))`
>   without `makeLandingPads`, which defaults `false` (`memory/TLNBDCache.scala:24`). `TLULNBDCache` wraps
>   `TLNBDCache` (`memory/TLULNBDCache.scala:97`), so the no-skid path is live: `TLNBDCache.scala:200-203`
>   wires `tlIn.d` straight from the MSHR `resp` with `assert(!resp.valid||tlIn.d.ready,"response must be ready!")`.
> - MSHR count in effect = `cfg.nMSHRs`=4 (`memory/MuonDCache.scala:434`). L2 = InclusiveCache 512 KB 8-way,
>   `outerLatencyCycles=40` (`RadianceConfigs.scala:51-56`); `EdgeDataBits=64` (8 B/beat DRAM width, line 25).
>
> **(2) MEASUREMENT re-verified from the RTL trace-db (deterministic; exact match to ledger).**
> `autocomp_gemvlat_base/trace_run.sqlite` (this IS the Verilator trace: `MAX(inst.cycle)=1,443,950`, identical to
> the ledger and ≠ cyclotron's 1.33M). N=128,K=1024,512 KB fp32 ⇒ 512 KB/1,443,950 = **0.363 B/cyc**.
> `autocomp_bw_17` = **250,031** (ledger 250,033; profiler-vs-trace off-by-2) ⇒ **2.097 B/cyc**. 0.363/2.10 = **17%**. ✓
>
> **(3) DIRECT OCCUPANCY EVIDENCE (trace proxy — recompile-free; DUT untouched) — REFUTES "too few misses".**
> Per-core sweep-line over l0d load [req_cycle,resp_cycle) intervals (per-core = per-l0d, MSHRs are per-tile):
> - Base decode-GEMV in-flight is **NOT pinned at 2 or 4** — request-level time-avg **~30/core (max 105-130)**;
>   distinct-64B-line fetches time-avg **~13/core (max ~49)**, mass spread 1→33. The core issues *plenty* of
>   concurrent misses. Per-lane load latency median **~450 cyc** (max ~2900) ≫ the ~40-100 cyc nominal — the
>   signature of a deep **queue**, not of starvation.
> - **Smoking gun:** reconstructing l0d line-misses, **base = 70,724 misses for ~5,300 distinct lines = 8.6× refetch**;
>   raw l0d line-traffic = **3.14 B/cyc** of which **only 0.363 is useful**. The l0d is *saturated moving redundant
>   refetches*, not idle waiting on 4 MSHRs. (Caveat: a per-MSHR-valid counter would need a Verilator rebuild;
>   the trace occupancy is an upper bound on MSHR occupancy and already excludes the starvation reading.)
>
> **(4) ALTERNATIVES.** (a) **DRAM model**: 2.10 B/cyc is a *real measured* ceiling (bw_17, clean $finish), not a
> config artifact; base's 450-cyc effective latency is queue inflation from thrash, not the 40-cyc DRAM latency.
> (b) **SIMT issue rate**: NOT issue-bound — occupancy 13-49 proves the cores issue many concurrent loads.
> (c) **L2/link**: the *same* L2/link/l0d sustains 2.10 under bw_17 ⇒ not the base limiter.
> (d) **ACCESS PATTERN = THE CAUSE (decisive control).** Base `[N,K]` strides weight rows by K·4 = **4096 B = exactly
> the 4 KB l0d size** ⇒ all rows alias the same 64 sets (median 69, up to 129 distinct lines per 1-way set) ⇒
> catastrophic conflict thrash (8.6× refetch). `autocomp_bw_17` is the natural control: **same l0d / 4 MSHRs / no
> landing pads / same DRAM**, but coalesced (consecutive threads→consecutive words→1 line) ⇒ **~1.0 miss/line**,
> **2.10 B/cyc**. The entire 5.8× gap is refetch amplification (8.6× vs 2.0×), i.e. coalescing + cache geometry —
> **not** MSHR count and **not** landing pads. (e) **"Deadlock"**: `gemvlat_coal`=166,591 < the 250,031 coalesced
> floor for the same 512 KB ⇒ physically impossible if complete ⇒ it froze (incomplete); `ilp8`=1,713,415 likewise
> incomplete. Incompleteness is CONFIRMED. **UPDATE 2026-07-17 — the stall structure is now DIRECTLY OBSERVED
> (MEDIUM-LOW → HIGH):** a conflict-free re-tiled `[N,K]` decode GEMV (`autocomp_gemvlat_pad`, rows padded to 4160 B
> so a warp's 16 lanes hit 16 distinct sets → refetch amp measured 1.00× vs base 15.9×) `$fatal`s on the live
> **`TLNBDCache.scala:201 assert(!resp.valid||tlIn.d.ready,"response must be ready!")`** (the no-landing-pad l0d
> D-response path, `TLNBDCache_3.sv:177`) — firing IN the streaming reduction loop at ~17k cyc with 0/128 outputs,
> deterministic on `+verilator+seed+1`,`+2`, and unseeded. So the coalesced/conflict-free deadlock IS the l0d
> no-landing-pad backpressure. The old `_coal`/`_ilp8` "froze" traces were separately re-examined and show CONTINUOUS
> retirement to their last cycle (no terminal spin) = collaterally KILLED, not spin-to-cap — so the live padded assert,
> not those traces, is the load-bearing evidence now. Deadlock-is-l0d-backpressure: **HIGH**. (Full addendum with
> build-provenance + the SFU-barrier caveat: OPTIMIZATION_LEDGER.md "FOLLOW-UP 2026-07-17".)
> **HOW MUCH THE WALL COSTS (quantified):** the l0d overrun is CONCURRENCY-dependent — at NW≤2 the conflict-free stream
> never trips `TLNBDCache:201`, and the throttled `autocomp_gemvlat_pad_nw1` (NW1, 32 threads) **COMPLETES cleanly at
> 361,365 cyc / 128-of-128 out-stores = 1.45 B/cyc = 69% of the 2.10 ceiling = 4.0× the base 0.363** (deterministic on
> seed 1 + unseeded). So the re-tile+throttle software fix realizes **4.0× of the 5.8×**; the residual **1.45→2.10 gap
> is precisely the occupancy the no-landing-pad wall forbids** (adding warps to hide the last of the DRAM latency
> re-trips the assert). That last 1.45× is the HW ask (landing pads / skid on the l0d D-response). A second, orthogonal
> completion hazard surfaced: `SFUPipe.scala:71 assert(!reqSent)` (barrier/`IsNuInvoke`, the S2/S4 hazard) deterministically
> blocks rebuilt `_base` and NW2 — so on this build the l0d wall is dodge-able by throttling but the SFU-barrier wall
> also gates the family; NW1 happens to clear both.
>
> **(5) MX-DMA BYPASS CONTROL.** fp8 M32 batched-decode `autocomp_dec_batched` = **47,314 RTL cyc** for 256 KB fp8
> weights = **~5.5 B/cyc** (ledger cites ~7.1), i.e. **2.6× above the SIMT coalesced ceiling (2.10)** and 15× the base
> GEMV. Consistent with the l0d/SIMT path being capped below the Gemmini-DMA path. **CAVEAT for the RTL team**: 256 KB ≤
> the 512 KB L2, so part of the DMA's 5.5-vs-2.10 edge may be L2-residency, not pure l0d-bypass — do not over-claim.
>
> **REFRAMED, DEFENSIBLE residual of the claim (MEDIUM):** at the *coalesced ceiling* (bw_17), l0d miss occupancy
> averages **~3.6 ≈ nMSHRs=4** while sustaining 2.10 — so the **4 MSHRs may cap the SIMT streaming ceiling** (more
> MSHRs could raise 2.10 toward the DMA path's 5.5), and the coalesced/ILP8 deadlock may be the no-landing-pad wall.
> Those are real, source-grounded l0d limitations — but they are **NOT** why decode-GEMV runs at 17%. **The 17% is a
> kernel/access-pattern problem** (uncoalesced `[N,K]` thrashing a 4 KB direct-mapped cache); it is largely
> **fixable in software** by coalescing (→ ~2.10) *if* the coalesced-path deadlock is resolved.
>
> **FALSIFICATION.** This refutation is overturned if a *non-thrashing* `[N,K]` decode kernel (each l0d line read
> once, e.g. warp-shared-row / block-tiled) still runs at ~0.363 B/cyc — that would restore the MSHR-starvation
> reading. My data predicts it would instead approach 2.10. The reframed "MSHRs cap the ceiling" residual is
> falsified if a Verilator build with `nMSHRs`=8 leaves bw_17 at 2.10 (proving DRAM-bound, not MSHR-bound).
> **ACTION FOR RTL TEAM:** the highest-value *hardware* ask is **more l0d associativity / capacity** (4 KB 1-way is
> the thrash amplifier) at least as much as more MSHRs + landing pads; and the "2 MSHRs" figure everywhere should be
> corrected to **4** ([[muon-dram-calibration]], [[rtl-timing-model-what-we-optimize]], METHODOLOGY §1 all say 2).

## S2 — SFUPipe `assert(!reqSent)` is a FENCE/BARRIER stall assert (NOT transcendental, NOT a divide) — DUT-STRUCTURAL
> **ROOT-CAUSE CORRECTED 2026-07-19 (decisive, RTL-source-grounded).** Earlier framings of this assert as a
> "concentrated fexp/SMEM" or "SFU iterative-divide `ss/(float)K`" hazard were WRONG. The `assert(!reqSent)` lives
> in the `StallFields` case class (`radiance/muon/backend/int/SFUPipe.scala:68-72`), which is instantiated ONLY for
> **`IsNuInvoke` (barrier/nu.invoke, :85), `IsFenceI` (:93), `IsFenceD` (:98), `IsFenceS` (SMEM fence, :107)**. This
> "SFU" pipe handles CONTROL ops (barrier/fence/CSR/TMC/wspawn/tohost), not FP transcendentals or divides — those
> run on a separate FP pipe and CANNOT trip this assert. It fires when a new fence/barrier *start*s while a prior
> fence/barrier's response hasn't cleared (`reqSent` still set): a re-issued fence whose memory-drain response is
> stuck behind the l0d no-landing-pad backpressure (S1). It aborts under fence/barrier traffic that can't drain.
**IMPACT — ELEVATED (PROD-1/#89, 2026-07-19): this blocks FUSED-FFN/LAYER MEGAKERNEL RETIREMENT ENTIRELY.** Every
fused multi-body FFN/layer we tried (ffn_block_fp4, ffn_ro, noxnrt, ffn_nox256, autocomp_layer) DEADLOCKS here —
out_raw=0, only rmsnorm_body runs, then SFUPipe:71 fires on a FENCE in the rmsnorm→quant_xn body transition
(trace: last-issued `pc=1002d7c8 inst=...0000000f`, opcode 0x0f = MISC-MEM/FENCE), its drain-response held off by
l0d backpressure. NONDETERMINISTIC (autocomp_layer retired once, walled every other run) → the fused-block RTL
cycle numbers (e.g. the ex-headline ffn_block_fp4 "553,879") were DEADLOCK-SNAPSHOTS, not completions; the real
completeness gate is FINAL out_raw store-count (total-store gates were fooled by RMSNorm's Xn_store).

> **DECISIVE KERNEL EXPERIMENT (2026-07-19) — kernel-fixable vs DUT verdict = CONFIRMED DUT-STRUCTURAL.**
> Hypothesis under test: the assert is the RMSNorm `ss/(float)K` SFU-divide under l0d backpressure → avoid the
> divide and rmsnorm_body retires. FALSIFIED on two independent grounds + an empirical run:
> (1) `ss/(float)K` is ALREADY a fmul, not an SFU divide — K=FFN_HID=512 is a power of two so clang folds 1/512
>     value-preservingly; rmsnorm_body disasm has ZERO fdiv/fsqrt (my_rsqrt is pure-SW). The ELF's only 6 fdiv are
>     downstream (quant_xn/swiglu/quantize_fp4, ÷scale and silu), on the FP pipe, not this SFU pipe.
> (2) SFUPipe:71 is a fence/barrier assert (above) — no arithmetic op can trigger it.
> (3) Applied the change anyway (`ss * constexpr(1.0f/FFN_HID)`, autocomp_ffn_block_noxnrt), rebuilt, Verilator-ran:
>     DEADLOCKS at the IDENTICAL SFUPipe:71 (Vtime 267,065,000 vs orig 263,183,000), out_raw = 0/32768, ~133k cyc.
> → No kernel arithmetic change clears it; the fences are correctness-required (SIMT→Gemmini write-drain handoff).
> **The per-GEMM STANDALONE form (retires via the Gemmini-DMA operand path, out_raw-complete) is the final
> deployable answer — applies to all four target models' fused layers.** The fused megakernel is NON-VIABLE on this
> DUT. (Full evidence: OPTIMIZATION_LEDGER.md "DECISIVE: fused-FFN SFUPipe:71 deadlock", trace_noxdivfix.sqlite.)

So this + the l0d no-landing-pads (S1) together make the naive GMEM-materialized fused layer non-retiring on
silicon — the wins must ship as STANDALONE per-GEMM kernels (which retire via the Gemmini-DMA operand path). Also
bounds core-specialization.
**ASK (now high-priority):** deeper SFU **fence/barrier** request queue / skid buffer (so a re-issued fence doesn't
trip `assert(!reqSent)` while a prior fence's response is stalled) + the l0d landing pads (S1) — together they are
the prerequisite for a fused-layer megakernel to retire at all, not just a perf lever. Confirm whether SFUPipe:71
is a real hazard or an over-strict sim assert; if over-strict, its removal alone may unblock fused retirement.

## S2 — Register wall: 256 phys regs, globalOverSubscription abort at OCC≥6
`Rename.scala:123` globalOverSubscription: OCC=6 and OCC=8 are UNRUNNABLE (256 physical registers). Caps
occupancy-based latency hiding at OCC=4; blocks 8-head GQA attention at occ=3 (needs 256/256 with h-loop+causal
state), and persist+prefetch layer kernels.
**IMPACT:** occupancy is the natural lever to hide SFU/mem latency once bank conflicts are removed; the reg wall
caps it before it pays off (occ=3 single-head helps but is infeasible to stack onto the real 8-head kernel).
**ASK:** note for future sizing — more physical registers (or per-warp reg budget) would enable the occupancy
latency-hiding that the current file precludes.

## S2 — mxgemm K-loop DMA backpressure deadlock at real hidden (K≥2048)
The mxgemm K-loop at real hidden dims (K=2048, HID=2048) hits a finite-queue timing deadlock: functional
(cyclotron no-timing) PASSES bit-exact, but the timing model / RTL doesn't retire (queue fills, drain hangs).
Same l0d/queue-depth family as S1. Blocks real-dim on-device execution of the FFN/proj GEMMs.
**ASK:** likely resolved by the S1 l0d fix + deeper DMA/response queues; confirm the mxgemm operand DMA path's
in-flight limit at deep K.

## S2 — no native input-scale-load DMA → ~78k cyc/matmul fixed overhead is not kernel-reclaimable
Every MX GEMM pays a ~60-90k-cyc fixed overhead (RTL-measured: fp8 K2048 78,044 / K5632 89,653; fp4 K2048 61,433
/ K5632 77,835 — this is the whole-kernel-util gap; the mesh itself is 95-100% of peak in-loop). A big chunk is
the per-K-tile SIMT `load_scale_factors` marshalling E8M0 scales GMEM→scale-SRAM + `mu_fence_smem`. We tried to
reclaim it via a native mesh scale-load DMA (`gemmini_mx_load_scales`) — it DOES NOT EXIST in this hardware.
> ### VERIFICATION (source-grounded, HIGH confidence)
> - **No intrinsic** for an input-scale DMA in any header; the only funct-26 scale op (`CONFIG_SCALE_MEM`)
>   configures scale-mem READ selects + the OUTPUT-scale DMA addr — not an input load.
> - The co-model defines funct 27 `F_MX_LOAD_SCALES` but it is a **documented no-op**
>   (`radiance/cyclotron/src/muon/mxgemmini/mod.rs:140,268`: "scales are staged into SMEM by SIMT stores in the
>   shared-ext-mem deployment, nothing to DMA") — a spike/libgemmini concept that degenerates on radiance.
> - **Not in the RTL:** `GemminiISA.scala` functs stop at 26 (no funct 27); `ScaleFactorMem.scala`'s two write
>   ports are fed EXCLUSIVELY by the TileLink MMIO slave (`GemminiTile.scala:271-289`, SIMT 64-bit Puts); the only
>   scale-mem master/DMA port (`scalingFacClient`) is OUTPUT-only.
> - CONCLUSION: on the shared-ext-mem design the mesh reads a scale SRAM that ONLY SIMT can fill, so the SIMT-store
>   scale-load is architecturally required — ~0 of the ~78k is reclaimable by a kernel change.
**ASK:** add a funct-27 GMEM→scale-SRAM input DMA engine (an input-scale master port on ScaleFactorMem) so the
mesh can load block-scales without the SIMT store+fence tax → reclaims a big fraction of the ~78k/matmul overhead
(the dominant whole-kernel-util gap once the mesh is at peak).
**Corroborating evidence (2026-07-18, scale-prefetch experiment):** we tried to hide the scale-load in SOFTWARE
(cross-op prefetch: op N stages op N+1's tile-0 scales during its tail) — mechanism provably works (scale-load
moved off the prologue critical path) but NET ~0 on RTL: the scale-SRAM double-buffer is only 2-deep (1-bit
`double_buffer_sel`, ScaleFactorMem.scala:195), so the donor window is a single 128³ matmul — too short to hide
the SIMT scale-load, and it also forces an even-K-tile-count constraint. Confirms the SW angle is capped; the
funct-27 input DMA (and/or a deeper scale-SRAM buffer) is the real fix.
**KERNEL WORKAROUNDS meanwhile (small):** (a) widen the scale stores 32→64-bit (`store64`, MMIO slave takes 64-bit
Puts) ≈ ~1%; (b) cross-op **scale-prefetch** — hide the prologue scale-load behind the previous op's compute
(the "33% operand+scale-load prologue" bubble, ~14.5% is scale-load) — schedule-level, no HW needed.

## S4 — no vector/low-latency exp on the SFU → attention softmax exp-cost is irreducible in SW
The attention softmax's dominant residual (~100k cyc of the 195k single-head flash kernel) is 64 essential
`mu_fexp`/row of SFU exp latency — 1 SFU exp/element, and there is NO vector-exp intrinsic (Muon also has no
warp-shuffle — confirmed mu_intrinsics.h). We reclaimed the FENCE half of softmax algorithmically (thread-per-row,
0 fences vs 4 → 1.11× RTL, autocomp_fa_d64_tpr) but the exp half can't be cut without changing precision. VERIFICATION:
kernel-level — the fence saving is measured (−23,151 cyc = 18.8% of the softmax phase, RTL), the exp residual is the
per-element SFU latency floor. **ASK (low priority):** a vector-exp / lower-latency SFU exp (or a fast exp2 path)
would directly cut the largest remaining attention cost; also a warp-shuffle would enable cheaper cooperative
reductions. Not blocking — flagged as the next-tapeout softmax lever.

## S3 — Requant fp8-out datapath divergence (RTL-only, team already WIP)
fp8-output requantization is WRONG only on Radiance RTL (~¼ of elements take a survive+saturate path → wrong
scale); bit-exact on the cyclotron co-model; all spike/Rocket examples are correct — i.e. specific to this RTL
requant datapath. Kernel-side workaround = SIMT quantize (1.85× when fused). Detailed in REQUANT_INVESTIGATION.md
and [[requant-rtl-datapath-divergence]]. One HW lever noted: gemmini_mx_load_scales.
**ASK:** confirm status of the requant datapath fix; the SIMT-quantize workaround is what we ship meanwhile.

## S3 — MX-mesh FP output un-verifiable on-device (trace-db + co-model both broken here)
On-device numeric verification of MX-mesh attention/GEMM FP output is platform-blocked on BOTH engines:
(a) cyclotron co-model NaN-codes the mesh FP output (reconstructed O = 0x7fc0 bf16-NaN);
(b) the RTL trace-db `data` field reconstructs to garbage for long MX kernels (same class as the tapeout-330
trace-ABI mismatch, [[tapeout330-trace-abi-mismatch]] — needs radiance 6462826 + a4656ba backported).
Correctness of the mesh path currently rests only on Python golden rel-err (fp8 attention 4-6%).
**ASK:** (1) backport the trace-ABI fix so trace-db cycle/data fields are trustworthy; (2) fix the co-model mesh-FP
reconstruction so on-device tohost/verify works for MX outputs (today only GEMM/SIMT paths verify via tohost=0).

## S4 — DRAM bandwidth spec correction
Docs (HW_UTILIZATION, MUON_RESULTS) claim 4–8 B/cyc DRAM. RTL-measured streaming ceiling is **2.10 B/cyc**
(bw_17: 250,033 cyc for 512 KB, clean $finish; matches a 1.93 historical anchor). Weight-move floor estimates
using 4–8 B/cyc are ~2–4× optimistic. **ASK:** correct the DRAM-BW figure in the docs to the measured ~2.1 B/cyc.

## S3 — IPDOM warp-reconvergence stack overflows on deeply-nested data-dependent branches (kernel-workaroundable)
`WarpScheduler.sv:793 "ipdom stack is full"` → $stop when a warp nests too many data-dependent branches (the
runtime fp6-LUT activation quantizer: 7 nested early-returns in bf16→fp6 + a 16-way if-chain nearest-finder). The
co-model does NOT model the IPDOM stack, so it PASSED there while the RTL asserts — another "cyclotron passes, RTL
walls" case (RRT is the arbiter). KERNEL WORKAROUND (no DUT change): make the control flow fully BRANCHLESS via
mask-select (RISC-V base has no cmov, so hand-rolled `dev_sel`); bit-identical (0/65536 + 0/200000 vs the branchy
form), and the branchless build RETIRES past the former deterministic $stop. So this is not blocking — but it's a
real DUT limit + a co-model fidelity gap. **ASK:** (1) deeper IPDOM stack (or document its depth so kernels stay
under it); (2) model the IPDOM stack in the co-model so it stops hiding these. Confidence HIGH (reproduced $stop
vs branchless retirement).

## S4 — cyclotron timing-model fidelity gaps (for the co-model owner)
cyclotron is the correctness oracle (functionally exact, generated the goldens) but its TIMING model diverges from
RTL in ways that repeatedly mis-ranked optimizations — flag for whoever maintains it:
- under-counts DRAM/global-mem ~30× (models l0d as non-blocking → "completes" bursts RTL deadlocks on).
- blind to L2 residency (mis-ranked weight-stationary reuse as 1.01× when RTL shows 1.33×).
- blind to SMEM bank conflicts (issue-bound; predicted a softmax bank-fix 1.19× that RTL showed −3.5%).
- mis-ranks warp-spec overlap (showed overlap SLOWER; RTL shows 1.28×).
- over-charges per-launch barrier/wspawn (predicted persistent-launch 1.24×; RTL shows 0.973×).
**ASK:** these are the specific effects to make RTL-faithful if cyclotron is to be trusted as a perf proxy;
today it is a correctness + coarse pre-filter ONLY, and every perf claim must be RTL-gated.

## S4 (toolchain, not DUT) — barrier-duplication compiler hazard
clang duplicates `mu_barrier(1, nw)` around warp-specialized code → cluster-barrier deadlock (warp parks in
mu_schedule_workers while another waits). Documented verbatim in radiance-kernels/lib/include/mu_intrinsics.h,
but bites fresh builds of the fused tinyllama kernel across all -O levels. Not a DUT issue — flag to the kernel-lib
/ toolchain owner. Fix = source-side `if(tid){}else{nop}` guards around the warp-spec barriers.

---
## Cross-cutting recommendation for the NEXT tapeout (priority order)
1. **l0d: more MSHRs + landing pads** (S1) — unlocks ~5.8× decode BW + the whole latency-hiding kernel class.
2. **Fix requant fp8-out datapath** (S3) — removes the need for the SIMT-quantize workaround.
3. **Backport trace-ABI + fix co-model mesh-FP reconstruction** (S3) — makes on-device MX verification possible.
4. **Deeper SFU req queue + more phys regs** (S2) — enables epilogue concentration + occupancy latency-hiding.
