# "Challenge every floor" campaign — consolidated report (2026-07-17)
RTL = tapeout-330 Verilator; correctness = cyclotron verify_body tohost=0 (GEMM/FFN) or Python golden rel-err
(fp8 mesh-attention, since on-device mesh-FP verify is platform-blocked). Every number below is RTL unless noted.

## Verdict table — floors that HELD vs headroom that was HIDING
| # | Claimed "no room" | Verdict | RTL evidence |
|---|---|---|---|
| 44 | weight-reuse is only ~1% | **REFUTED — 1.33× WIN** | true read-once tall-tile: DRAM weight-reads 4×→1×, 288,355→216,514 cyc. Prior 1.013× was wrong metric + a broken A/B that never varied residency |
| 45 | decode at DRAM floor, done | **CRACKED — 1.95×/token** | fp8 M=32→M=64: +2.9% cyc for 2× tokens (1,479→760 cyc/tok). Never measured before |
| 42 | idle-SIMT harvestable for matmul | mixed | GEMM-split DEAD (FLOP asym, +1.2%); but a DIFFERENT op concurrent = overlap 1.28× (real, prefill) |
| 47/52/55 | attention optimized | softmax lever REFUTED | softmax=64% of kernel but it's SFU-exp+fences, NOT bank conflicts (bcfree = −3.5%); occ3 flat; stack ~1.0× |
| 48/53 | overlap tops at 1.28× | **CONFIRMED floor** | heavier epilogue backfires (1.01×); spatial core-spec fails on 3 RTL barriers (reg-wall/SFU-assert/wrong-bottleneck) |
| 49/56 | megakernel fusion beats 1.25× | **CONFIRMED — 1.25× is ceiling** | persistent-launch 0.973× (SLOWER; earlier 1.264× was a 71%-truncated run); Xn-elim l0d-DEADLOCKS at layer scale |
| 46 | MX mesh below peak | **CONFIRMED at peak** | mesh 95-100% in K-loop; fp4=358 MAC/cyc; tk256 only +0.85%. Overhead is scale-load (~78k cyc, ISA-reclaimable) |
| 45 | fp4/sub-fp4 helps decode | **CONFIRMED dead** | fp4 2-2.3× SLOWER at small-M; sub-fp4 needs forbidden RTL change (2-bit format field) |
| 43 | cross-layer prefetch → 2× | unmeasurable | architecturally sound; blocked (cyclotron DMA-blind + reg-wall); toy dims not weight-bound, real dims don't retire |
| 50/51 | both engines can be ~50/50 busy | **CONFIRMED capped** | SIMT elementwise <1% of FLOPs; both-active hard-capped at ~17.5% MX-work fraction; attention (d=64) mesh only 0.4% |
| 41 | decode floor = BW not latency | PENDING | BW ceiling measured 2.10 B/cyc; coalescing GEMV verdict in flight |

## The two real WINS recovered (you were right to be suspicious)
1. **True weight-stationary read-once: 1.33× on RTL** (#44) — attacks the FIRST-ORDER DRAM weight stream (the actual
   floor), cutting weight-reads 4×→1× via tall non-square tiles (TM=256/TN=64). Caveat: partly tile-efficiency;
   needs TM-sweep isolation + port to real FFN dims + fp4-compound confirmation. THE headline recovery.
2. **Decode M=64 batching: 1.95×/token** (#45) — weight read amortized over 2× rows nearly free. Standalone, real.

## Confirmed WINS already banked (isolated levers, FFN block)
fp4 precision 1.74× · amortization util 31→80% · epilogue fusion 1.25× · fused-quant 1.85× · overlap 1.28×
(single-tile warp-spec) · Xn-round-trip-elim (block-level only).

## Does it COMPOUND? (the combination question)
- **FFN BLOCK: yes, partially → ~4-5× vs fp8-serial.** fp4×amortization compound cleanly (4.4×). fusion/fused-quant/
  Xn-elim compound (different traffic). overlap is REDUNDANT with fusion (~1.0× extra for FFN). Bit-exact throughout.
- **WHOLE LAYER: NO — ~half the product is lost.** naive 2.6× → realized ~1.0-1.33×. The single shared mesh
  serializes all matmuls; co-execution can't stack; only traffic-reducers survive, and at layer scale even those
  (persistent-launch, Xn-elim) fail on RTL. The epilogue-fused per-op layer is the fastest realizable form.
- **ATTENTION: NO — stack ~1.0×.** all levers aim at a bottleneck (SFU-exp+fences) none can move.

## The recurring hard wall: the l0d has no landing pads (tapeout-330, DUT-fixed)
Killed: attention QKᵀ‖softmax overlap (deadlock), Xn-elim at layer scale (deadlock/SFU-assert), aggressive
in-flight prefetch, lane-coalesced GEMV (assert). Any kernel that floods the l0d with concurrent streaming +
compute trips backpressure. This — not kernel skill — bounds several "obvious" levers.

## Only genuinely-NEW levers still untried (targeted, not a broad sweep)
1. Algorithmic softmax: cut exp-count (64 mu_fexp/row) + fence-count — the ONLY remaining attention lever.
2. Productionize weight-stationary read-once (#44) into real-dim FFN/proj + confirm fp4-compound.
3. Scale-load overhead: native gemmini_mx_load_scales ISA path (~78k cyc/matmul reclaimable, #46).
4. Decode M=96/128 ridge sweep (extend the confirmed 1.95× decode win).
