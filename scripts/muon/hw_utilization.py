#!/usr/bin/env python3
"""RIGOROUS whole-Radiance hardware utilization from an RTL (Verilator/VCS) trace.

Measures COMPUTE-ONLY: cycles are counted from the FIRST kernel instruction's emission to the
last, using ELF-symbol PC ranges (kernel_body + mxgemm*) intersected with the trace-db `inst`
table. Harness setup (main), the DRAIN spin, and verify_body are excluded.

CRITICAL — the two engines are NOT interchangeable and must not be conflated:
  * MX-Gemmini does matmul on a 16x16 systolic array. Its work is NOT in the inst trace
    (the trace only records Muon instructions). MX utilization is analytical essential-MACs
    over measured cycles: essential_MACs / (PEAK_MX * cycles).
      PEAK_MX = 256 MAC/cyc (fp8, 16-wide tiles) | 512 MAC/cyc (fp6/fp4, 32-wide sub-byte tiles).
    For an MX kernel the Muon warps only ORCHESTRATE (issue MMIO + SIMT move-out), so Muon IPC
    is deliberately low -- it is NOT a compute-throughput number.
  * Muon SIMT does the elementwise/GEMV/GEMM-SIMT compute on 2 cores x 16 lanes, single-issue.
      PEAK_SIMT = 32 MAC/cyc = 64 flop/cyc ;  peak issue = 2 instr / core-cyc.
    Here Muon IPC and the SIMT FP roofline ARE the compute metrics.

So this reports, per engine, only the metric that is real for that engine, and labels which unit
does the compute. For fused kernels it splits the MX phase (mxgemm* PCs) from the SIMT phase.

Peaks verified against: RadianceSingleClusterConfig (WithSIMTConfig numWarps=8 numLanes=16 +
WithMuonCores(2); MX DIM=16) and cyclotron/src/muon/mxgemmini/mod.rs:105-106.

Usage:
  hw_utilization.py <radiance.elf> <trace.sqlite> --engine mx|simt|fused
       [--macs N | --flops N] [--fmt fp8|fp6|fp4] [--label name]
"""
import argparse, re, sqlite3, subprocess

OBJDUMP = "/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
NUM_CORES, NUM_LANES = 2, 16
PEAK_ISSUE = NUM_CORES                       # 1 warp-instruction / core / cycle (single-issue)
PEAK_SIMT_MAC = NUM_CORES * NUM_LANES        # 32 MAC/cyc  (each lane 1 FMA/cyc)
PEAK_SIMT_FLOP = PEAK_SIMT_MAC * 2           # 64 flop/cyc
PEAK_MX = {"fp8": 256, "fp6": 512, "fp4": 512}   # MAC/cyc, per mod.rs:105-106
# Memory-bandwidth ceilings for the REALISTIC roofline (bytes/cyc), from the RTL-fitted timing
# model (cyclotron config/timing/{gmem,smem}.toml): DRAM node 4 B/cyc; SMEM lane 64 B/cyc.
BW_DRAM = 4      # cold streaming from DRAM (no cache reuse)
BW_SMEM = 64     # operands/intermediates resident in cluster SMEM


def roofline(peak_flop, ach_flop, ai, cyc, ach_bw):
    """Print ideal (compute) vs realistic (min(compute, BW*AI)) rooflines + the binding limit."""
    dram_ceil = BW_DRAM * ai       # flop/cyc achievable if DRAM-BW-bound at this AI
    smem_ceil = BW_SMEM * ai
    real_ceil = min(peak_flop, dram_ceil)         # cold-DRAM realistic ceiling
    real_ceil_smem = min(peak_flop, smem_ceil)    # if operands SMEM-resident
    print(f"  --- roofline ---")
    print(f"  arithmetic intensity  : {ai:.2f} flop/byte  (ridge: DRAM {peak_flop/BW_DRAM:.0f}, SMEM {peak_flop/BW_SMEM:.0f} flop/byte)")
    print(f"  achieved              : {ach_flop:.1f} flop/cyc, {ach_bw:.2f} B/cyc DRAM")
    print(f"  IDEAL roofline (compute peak {peak_flop:.0f}) : {100*ach_flop/peak_flop:.2f}% of peak")
    print(f"  REALISTIC roofline (cold-DRAM, min(peak,{BW_DRAM}*AI)={real_ceil:.0f}) : {100*ach_flop/real_ceil:.1f}% of achievable")
    print(f"  REALISTIC roofline (SMEM-resident, ={real_ceil_smem:.0f})            : {100*ach_flop/real_ceil_smem:.1f}%")
    print(f"  DRAM-BW util          : {100*ach_bw/BW_DRAM:.1f}% of {BW_DRAM} B/cyc peak")
    bind = ("COMPUTE-bound" if real_ceil >= 0.9*peak_flop and ach_flop > 0.6*real_ceil
            else "DRAM-BW-bound" if ach_bw > 0.6*BW_DRAM
            else "OVERHEAD/LATENCY-bound (below BOTH ceilings -> fuse/enlarge tiles/overlap)")
    print(f"  binding limit         : {bind}")


def symbols(elf):
    out = subprocess.run([OBJDUMP, "-t", elf], capture_output=True, text=True).stdout
    fns = []
    for line in out.splitlines():
        m = re.match(r"([0-9a-f]+)\s+\S+\s+\S*\s*F\s+\.text\s+([0-9a-f]+)\s+(\S+)", line)
        if not m:
            continue
        a, sz, nm = int(m[1], 16), int(m[2], 16), m[3]
        if sz and not (nm == "main" or "verify_body" in nm):
            fns.append((a, a + sz, nm))
    return sorted(fns)


def span(cur, ranges):
    """(first_cycle, last_cycle, instr_count) for the union of PC ranges, from the inst trace."""
    if not ranges:
        return None
    where = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi, _ in ranges)
    r = cur.execute(f"select min(cycle),max(cycle),count(*) from inst where {where}").fetchone()
    return r if r[0] is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf"); ap.add_argument("trace")
    ap.add_argument("--engine", choices=["mx", "simt", "fused"], required=True)
    ap.add_argument("--macs", type=int, default=0, help="essential MACs (matmul: M*N*K)")
    ap.add_argument("--flops", type=int, default=0, help="essential FLOPs (elementwise ops)")
    ap.add_argument("--bytes", type=int, default=0, help="bytes moved from DRAM (for the memory roofline)")
    ap.add_argument("--fmt", choices=["fp8", "fp6", "fp4"], default="fp8")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    fns = symbols(a.elf)
    kern = [f for f in fns if "kernel_body" in f[2] or "mxgemm" in f[2]]
    mxfns = [f for f in fns if "mxgemm" in f[2]]
    simtfns = [f for f in fns if "kernel_body" in f[2]]
    cur = sqlite3.connect(f"file:{a.trace}?mode=ro", uri=True).cursor()

    full = span(cur, kern)
    if not full:
        raise SystemExit("no kernel instructions in trace (ELF/trace mismatch?)")
    first, last, instrs = full
    cyc = last - first
    print(f"=== HW utilization: {a.label or a.elf.split('/')[-1]}  (engine={a.engine}, fmt={a.fmt}) ===")
    print(f"compute window (RTL, from 1st kernel instr): [{first}, {last}] core cyc")
    print(f"  actual COMPUTE cycles : {cyc} core  ({cyc*2} tile)")
    print(f"  Muon instrs (traced)  : {instrs}   over {NUM_CORES} cores x {NUM_LANES} lanes")
    ipc = instrs / cyc if cyc else 0
    print(f"  Muon IPC (aggregate)  : {ipc:.3f} instr/core-cyc   (issue-slot util {100*ipc/PEAK_ISSUE:.1f}% of peak {PEAK_ISSUE})")

    if a.engine in ("mx", "fused"):
        peak = PEAK_MX[a.fmt]
        # MX phase cycles: span of mxgemm* instructions (the accelerator is busy during them)
        mxspan = span(cur, mxfns) if mxfns else None
        mxcyc = (mxspan[1] - mxspan[0]) if mxspan else cyc
        if a.macs:
            util_full = 100 * (a.macs / peak) / cyc
            util_mxph = 100 * (a.macs / peak) / mxcyc if mxcyc else 0
            print(f"  --- MX-Gemmini (systolic, compute here) ---")
            print(f"  essential MACs        : {a.macs}   peak {peak} MAC/cyc ({a.fmt})")
            print(f"  MX util /full-window  : {util_full:.2f}%   (essential_MACs / (peak * {cyc}))")
            if a.engine == "fused":
                print(f"  MX phase cycles       : {mxcyc}  -> MX util /MX-phase : {util_mxph:.2f}%")
            print(f"  NOTE: Muon IPC above is ORCHESTRATION only for the MX path (warps issue MMIO + move-out).")
            if a.bytes:
                flop = 2 * a.macs
                roofline(2 * peak, flop / cyc, flop / a.bytes, cyc, a.bytes / cyc)
        else:
            print(f"  (pass --macs M*N*K for the MX roofline)")

    if a.engine in ("simt", "fused"):
        print(f"  --- Muon SIMT (lanes, compute here) ---")
        if a.macs:
            util = 100 * (a.macs / PEAK_SIMT_MAC) / cyc
            print(f"  essential MACs        : {a.macs}   peak {PEAK_SIMT_MAC} MAC/cyc (2 cores x 16 lanes FMA)")
            print(f"  SIMT compute util     : {util:.2f}%   (essential_MACs / (peak * {cyc}))")
            if a.bytes:
                flop = 2 * a.macs
                roofline(PEAK_SIMT_FLOP, flop / cyc, flop / a.bytes, cyc, a.bytes / cyc)
        elif a.flops:
            util = 100 * (a.flops / PEAK_SIMT_FLOP) / cyc
            print(f"  essential FLOPs       : {a.flops}   peak {PEAK_SIMT_FLOP} flop/cyc")
            print(f"  SIMT compute util     : {util:.2f}%   (essential_FLOPs / (peak * {cyc}))")
            print(f"  NOTE: elementwise/norm kernels are MEMORY-bound; low FP util is expected -- IPC/BW is the real limiter.")
        else:
            print(f"  (pass --macs or --flops for the SIMT roofline)")


if __name__ == "__main__":
    main()
