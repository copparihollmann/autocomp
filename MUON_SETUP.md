# Autocomp on radiance Muon (cyclotron fast loop + VCS gate)

> **Search tuning, produced-file map, and the cyclotron fidelity envelope (what the model can/can't
> rank — read before trusting matmul SMEM results):** see [`MUON_SEARCH_TUNING.md`](MUON_SEARCH_TUNING.md).

## What this is
Autocomp port for the radiance Muon SIMT GPU (NO mxgemmini): optimize fp32 kernels
(matmul / conv2d / attention(64) / attention(96, "flash") / swiglu / softmax) extracted
from real models (openvla, rdt, smolvla — captured via model2MLIR). Iterate fast on
cyclotron, gate winners on VCS RTL. **$100 hard cap** on every LLM call.

## Quick start
```bash
cd /scratch/agustin/projects/autocomp
source muon.env                                 # bearer token + budget cap + paths
.venv/bin/python -m autocomp.search.run_search_muon 0   # 0..5 = problem
./watch_costs.sh                                # spend; cap = $100, mode=stop
.venv/bin/python -m autocomp.common.cost --total
./scripts/muon/run_problem.sh 0                 # eval baseline (PASS + cycles)
./scripts/muon/vcs_gate.sh 0 <candidate.cpp>    # RTL gate one candidate (~5-15min)
./run_all_muon.sh                               # search all problems sequentially
```

## Problems (prob_type "muon", baselines in sols/muon)
0 matmul 64x64x64 · 1 conv2d patch-embed (3x64x64 * 16x3x16x16 s16) · 2 attention seq64
3 attention seq96 ("flash") · 4 swiglu 64x128 · 5 softmax 64x67 — all fp32,
shapes from openvla/rdt/smolvla MLIR (scripts/muon/extract_specs.py).

## Layers
- harnesses/muon/test{N}.c (+ test{N}/: data, Makefile, gen_data.py, host.cpp)
- baseline-as-gold: data header with golden tensors; verify (CPU lane 0) → tohost
  errors via ecall rs1 ("isa-test passed" / case=N). Latency = cyclotron --timing cycles.
- autocomp/backend/muon/muon_eval.py + hw_config/muon_config.py, run_search_muon.py
- agent .built/muon (sources: agent_sources/muon)
- cyclotron config: scripts/muon/config_muon.toml (10M cycle timeout)

## Pitfalls (cost a day, don't repeat)
- Device prints unreliable; verdict ONLY via tohost ecall (rs1!), latency from sim total.
- Harness must mu_barrier(0, MU_NUM_CORES) after mu_schedule: core0 may verify
  before core1 finishes -> 16 phantom errors.
- mu_exp polynomial must match in gen_data; keep |x| <= 2 (and away from exp tie).
- Tail data corruption -> tls_guard array at end of data; volatile errors counter.
- VERIFY_COUNT must be a compile-time literal in data; M*N expression miscompiles.
- VCS gate: SOC_DIR default (radiance-kernels/soc); not chipyard!

## Cost (running, project lifetime ledger ~/.autocomp/muon-spend.jsonl)
agent build ~$0.54, smoke ~$0.21, search ~$2/problem (Sonnet 4.6).
Hard cap $100 (AUTOCOMP_SPEND_LIMIT_USD, mode=stop).

## Failure modes & gotchas (notes for generating more kernels)

### Correctness (cyclotron eval — these fail the kernel, fix before any RTL gate)
- **Cross-core race**: with `mu_schedule` over MU_NUM_CORES, results in global memory
  aren't visible across cores until `mu_barrier(0, MU_NUM_CORES)` AFTER the schedule
  call. Symptom: a fixed small number of phantom errors (e.g. SwiGLU 16). Always
  barrier across cores before the verify loop.
- **Intra-threadblock barrier**: between phases that read each other's SMEM/global
  writes use `mu_barrier(1, num_warps)`. Missing it => nondeterministic wrong values.
- **Dropped volatile stores**: the Muon LLVM backend drops volatile stores that
  follow large-trip-count loops or string literals. Never rely on device-side prints;
  the verdict must be a single ECALL with code in rs1 ((errors<<1)|1).
- **FP tolerance**: RTL divider is ~1 ULP approximate and tiled/split-K reassociates
  accumulation. Compare with rel+abs tolerance (1e-4 rel, 1e-5 abs), never bitwise.
  Gold must accumulate sequentially (not numpy pairwise) to match device order.
- **8-warp output mapping**: configs run 8 warps/core (config_muon.toml num_warps=8).
  Kernels that hard-code "threads = OC*P" or "2 outputs/thread" for 4 warps write
  out-of-bounds at 8 warps. Derive the thread->output map from
  threads_per_threadblock, never from a literal warp count.

### Performance (cyclotron --timing)
- **16-way SMEM bank conflict hang**: column-major SMEM access with stride a multiple
  of 16 floats serializes all 16 lanes onto one bank -> sim appears to hang under
  --timing. Pad every column/row stride by +16 floats (or use an odd pitch).
- **Don't stage once-read data**: conv patch-embed (each element read a few times)
  got SLOWER with every SMEM variant (im2col 1.26x slower, plain-copy/pitch/split-K
  all >= baseline). Stage only operands reused many times (matmul/attention operands).
- **Register tiling plateaus ~2x**: 4 warps x 1 instr/lane/cycle issue limit means
  matmul tops out near 2.2x; past 8 outputs/thread the inner loop is issue-bound.

### RTL gate (VCS RadianceSingleClusterConfig)  <-- the path that just failed
- **host.cpp must launch the GPU**. A stub `int main(){return 0;}` never resets/starts
  the GPU, so the sim runs to the TestDriver timeout (~546M ps) and looks like a hang.
  Use harnesses/muon/host.cpp: WRITE RAD_HOST_GPU_RESET 1 -> *tocpu=tohost ->
  RESET 0 -> poll RAD_HOST_GPU_ALL_FINISHED with SYNC_GPU().
- **Do NOT reset the GPU after it finishes** before the verdict propagates: the GPU's
  terminating tohost ecall is what ends the sim and sets PASSED/FAILED. Resetting first
  suppresses it (see kernels/launch/host.cpp comment).
- **"tohost and fromhost symbols not in ELF" is BENIGN** — known-good reference soc
  elfs (bfs, gaussian) also lack those symbols (host carrier aliases them via tocpu at
  0x100010000). Do NOT grep for the word "tohost" to detect failure; that false-matches
  this warning. Parse only `*** PASSED ***` / `*** FAILED *** (tohost = N)`; errors = N>>1.
- RTL cycles come from TestDriver's "Completed after N simulation cycles" line, not the
  $finish ps timestamp.

## RTL gate — working result (matmul baseline)
- matmul 64x64x64 baseline on RadianceSingleClusterConfig: PASS, 1,107,417 RTL
  cycles (553,708,500 ps / 500 ps-per-cycle). Both GPU cores executed ~97k instrs.
- The gate verifies "runs to completion on RTL + cycle count", NOT bytewise output
  (host.cpp returns 0 unconditionally -> host tohost=1 -> success). Bytewise
  correctness is gated on cyclotron, which is FP/perf-equivalent to RTL
  (CYCLOTRON_VS_RTL_FINDINGS). To add RTL correctness gating, host.cpp must read the
  GPU verdict from *tocpu after ALL_FINISHED and return nonzero on mismatch.
- Pass detection: success branch is a silent $finish at TestDriver.v line 158 unless
  +verbose is passed; the gate now passes +verbose and also treats a line-158 $finish
  (no "*** FAILED ***") as PASS.

## *** MAJOR: RTL physical-register limit not modeled by cyclotron ***
Muon core: numWarps=8, numPhysRegs=256 (MuonCore.scala:21,24). The 256-entry
physical register file is SHARED across all active warps; the Rename stage asserts
`globalOverSubscription` ("total register usage exceeded maximum number of physical
registers", Rename.scala:123) and the sim $finishes (FATAL) when the live physical
register demand across warps exceeds 256.

- Cyclotron's functional model does NOT enforce this. Kernels that win on cyclotron
  via deep register tiling can be ILLEGAL on RTL.
- CONFIRMED failures (both $fatal at Rename.sv ~85-94M ps):
    matmul 2x8 tile (sol0_smem_2x8, ~16 accumulators) -> over-subscription
    attention-64 SMEM (sol2_smem, 4-out tile + many SMEM addrs) -> over-subscription
- baseline matmul (1 out/thread, ~4-6 live regs) PASSED on RTL.

Implication for kernel generation / search fitness:
- Register tiling depth is hard-capped: keep (peak live regs per warp) low enough that
  the sum across active warps stays under 256. At 8 warps that's ~32 arch regs/warp
  budget INCLUDING rename headroom -> heavy tiles (>~12 live fp accumulators + address
  regs) are unsafe.
- Cyclotron speedups from heavy tiling are UPPER BOUNDS; must be re-validated on RTL
  for register legality before claiming them.
- Levers to fit: fewer outputs/thread (4 instead of 8/16), fewer live SMEM address
  regs (recompute offsets vs keep 8 base pointers), or fewer active warps.

## FIX LANDED: cyclotron now models the RTL physical-register limit
Per the finding above, cyclotron was missing the RTL Rename physical-register
constraint, so it ran register-illegal kernels that the RTL cannot. Fixed in the
cyclotron source (generators/radiance/cyclotron):
- config.rs: new MuonConfig field `num_phys_regs` (default 256; 0 disables check).
- sim/elf.rs: ElfBackedMem records the `kernel_body` symbol range from the ELF symtab.
- sim/top.rs: `check_register_pressure()` runs at load -- statically decodes the
  kernel_body instructions, counts distinct architectural registers (rd + valid
  rs1/rs2/rs3 via has_regs), and panics with `globalOverSubscription` if that count >
  num_phys_regs/num_warps (= 32 at default 256/8). config_muon.toml + config.toml set
  num_phys_regs=256.
Validated: cyclotron verdicts now match RTL exactly --
  baseline(20) RAN, lean-SMEM(22) RAN, smem-unroll8(35) REJECTED,
  2x8(51) REJECTED, attn64(55) REJECTED.
So register-illegal kernels are now caught in ~1s on cyclotron instead of a ~13-min
RTL run. The objdump-based reg_check.sh / muon_eval.py gate remain as a fast
pre-compile screen (belt and suspenders).
