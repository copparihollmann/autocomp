#!/usr/bin/env python3
"""Muon FP utilization calculator.

Utilization = essential FLOPs / (peak * cycles).
  numerator   = essential problem FLOPs (2*M*N*K for matmul), not executed instruction count.
  denominator = full hardware peak: 2 cores * 16 lanes * 2 flop/FMA = 64 flop/cycle, 500 MHz = 32 GFLOP/s.
  cycles      = total measured cycles. A net (overhead-excluded) figure is also reported.
"""
CORES, LANES, FLOP_PER_FMA = 2, 16, 2
PEAK_FLOP_PER_CYC = CORES * LANES * FLOP_PER_FMA      # 64
CLK_HZ = 500e6
PEAK_GFLOPS = PEAK_FLOP_PER_CYC * CLK_HZ / 1e9        # 32.0

def matmul_flops(M, N, K):  # essential MACs *2
    return 2 * M * N * K

def matmul_bytes(M, N, K, dtype=4):  # MINIMAL DRAM traffic: read A,B once + write C once (lower bound)
    return (M*K + K*N + M*N) * dtype

# Effective sustained global-BW the cyclotron model + RTL agree on after calibration: data-level
# 6 B/cyc (bw=6) => 3 GB/s at 500 MHz. Used as the EFFECTIVE roofline BW (conservative: it's the
# rate that reproduces RTL timing, not an inflated nominal DRAM peak).
EFF_BYTES_PER_CYC = 6

def roofline(M, N, K):
    f, b = matmul_flops(M,N,K), matmul_bytes(M,N,K)
    ai = f / b                                   # flop/byte (arithmetic intensity)
    comp_floor = f / PEAK_FLOP_PER_CYC           # cycles if compute-bound at peak
    mem_floor  = b / EFF_BYTES_PER_CYC           # cycles if memory-bound at eff BW
    bound = "compute" if comp_floor >= mem_floor else "memory"
    return dict(ai=round(ai,2), comp_floor=round(comp_floor), mem_floor=round(mem_floor),
                roofline_floor=round(max(comp_floor,mem_floor)), bound=bound)

def util(essential_flops, total_cycles, overhead_cycles=None):
    peak_cyc_floor = essential_flops / PEAK_FLOP_PER_CYC          # min cycles if 100% utilized
    u_total = essential_flops / (PEAK_FLOP_PER_CYC * total_cycles)
    achieved_gflops = essential_flops / (total_cycles / CLK_HZ) / 1e9
    out = dict(essential_flops=essential_flops, total_cycles=total_cycles,
               peak_cycle_floor=round(peak_cyc_floor),
               util_total_pct=round(100*u_total, 2),
               achieved_gflops=round(achieved_gflops, 3))
    if overhead_cycles is not None and total_cycles > overhead_cycles:
        comp = total_cycles - overhead_cycles
        out['util_computeregion_pct_OPTIMISTIC'] = round(100*essential_flops/(PEAK_FLOP_PER_CYC*comp), 2)
    return out

if __name__ == '__main__':
    print(f"PEAK: {PEAK_FLOP_PER_CYC} flop/cyc = {PEAK_GFLOPS} GFLOP/s (fp32, 2c*16L*2, 500MHz)\n")
    # (label, M,N,K, total_cycles, source)
    rows = [
        ("prob0 64^3 naive  RTL",      64,64,64,  843556, "RTL FastConfig"),
        ("prob0 64^3 smem   RTL",      64,64,64,  726584, "RTL FastConfig"),
        ("prob0 64^3 naive  cyc@bw6",  64,64,64,  480918, "cyclotron bw6"),
        ("prob0 64^3 smem   cyc@bw6",  64,64,64,  419797, "cyclotron bw6"),
        ("prob6 64x64x128 base cyc@bw6",64,64,128, 487016, "cyclotron bw6"),
        ("prob10 64x64x384 base cyc@bw6",64,64,384, 998938,"cyclotron bw6"),
    ]
    print(f"{'kernel':32} {'essFLOP':>9} {'cycles':>9} {'rfloor':>7} {'UTIL%':>7} {'GFLOP/s':>8} {'cyc/floor':>9}  source")
    for lab,M,N,K,cyc,src in rows:
        r = util(matmul_flops(M,N,K), cyc)
        rl = roofline(M,N,K)
        ratio = cyc / rl['roofline_floor']
        print(f"{lab:32} {r['essential_flops']:>9} {r['total_cycles']:>9} {rl['roofline_floor']:>7} "
              f"{r['util_total_pct']:>6}% {r['achieved_gflops']:>8} {ratio:>8.0f}x  {src}")
    print()
    print("Roofline (essential, per shape) -- where is the ceiling?")
    print(f"{'shape':18} {'AI flop/B':>10} {'comp_floor':>11} {'mem_floor':>10} {'bound@bw6':>10}")
    for M,N,K in [(64,64,64),(64,64,128),(64,64,384),(64,64,768),(256,256,256)]:
        rl = roofline(M,N,K)
        print(f"{f'{M}x{N}x{K}':18} {rl['ai']:>10} {rl['comp_floor']:>11} {rl['mem_floor']:>10} {rl['bound']:>10}")
    print("\nINTERPRETATION: cyc/floor = how many x slower than the roofline. Large gap that is NOT")
    print("explained by comp/mem floors = latency + fixed launch/fill/drain overhead + low occupancy.")
