#!/usr/bin/env python3
"""Measure KERNEL-ONLY utilization for a combined Muon+MX-Gemmini run.

Utilization needs the exact first and last instruction OF THE KERNEL -- not the whole
program (which includes harness setup, the write-drain spin, and the verify pass). We get
that by pinning the kernel's PC ranges from ELF symbols and intersecting with the per-
instruction trace-db (`inst`: cluster/core/warp/cycle/pc) the RTL gate embeds.

"The kernel" = the driver functions `kernel_body` and `mxgemm*` (config, scale staging,
the K-loop, and the SIMT C move-out). `main` (setup + DRAIN_ITERS spin + tohost) and
`verify_body` are excluded -- they are harness, not workload.

Reported:
  actual cycles  = last_kernel_cycle - first_kernel_cycle           (wall-clock the kernel took)
  instrs         = # traced instructions in the kernel PC ranges    (issue slots, per-warp)
  IPC            = instrs / actual cycles
  ideal cycles   = essential MACs / accelerator peak                (M*N*K / (PE_M*PE_K) for the systolic array)
  utilization    = ideal / actual                                   (how close to the matrix unit's roofline)

Trace `cycle` is the Muon CORE clock (2x the tile clock the vcs gate divides $finish_ps by);
both are printed.

Usage: kernel_utilization.py <radiance.elf> <trace.sqlite> M N K [--fmt fp8|fp6|fp4]
"""
import re, sqlite3, subprocess, sys

OBJDUMP = "/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
# accelerator peak MACs/cycle: 16x16 systolic array, 16-wide fp8 tiles (32-wide sub-byte
# tiles still feed a 16-deep array => same 256 MAC/cyc peak for the essential-work roofline).
PEAK_MAC_PER_CYC = 16 * 16

def kernel_ranges(elf):
    """[(lo,hi,name)] for the kernel driver functions (kernel_body + mxgemm*), from symbols."""
    out = subprocess.run([OBJDUMP, "-t", elf], capture_output=True, text=True).stdout
    ranges = []
    for line in out.splitlines():
        # "100050 00 l  F .text  00000638 _Z11kernel_bodyPvjjj"
        m = re.match(r"([0-9a-f]+)\s+\S+\s+\S*\s*F\s+\.text\s+([0-9a-f]+)\s+(\S+)", line)
        if not m:
            continue
        addr, size, name = int(m[1], 16), int(m[2], 16), m[3]
        if size == 0:
            continue
        dem = name
        is_kernel = ("kernel_body" in dem) or ("mxgemm" in dem)
        is_harness = ("verify_body" in dem) or dem == "main"
        if is_kernel and not is_harness:
            ranges.append((addr, addr + size, name))
    return sorted(ranges)

def in_ranges_sql(ranges, col="pc"):
    return " or ".join(f"({col}>={lo} and {col}<{hi})" for lo, hi, _ in ranges)

def main():
    elf, db = sys.argv[1], sys.argv[2]
    M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    ranges = kernel_ranges(elf)
    if not ranges:
        raise SystemExit("no kernel_body/mxgemm symbols found in ELF")
    c = sqlite3.connect(db).cursor()
    where = in_ranges_sql(ranges)
    row = c.execute(
        f"select min(cycle), max(cycle), count(*) from inst where {where}"
    ).fetchone()
    first, last, instrs = row
    if first is None:
        raise SystemExit("no kernel instructions in trace (wrong ELF/trace pairing?)")
    actual = last - first
    macs = M * N * K
    ideal = macs / PEAK_MAC_PER_CYC
    ipc = instrs / actual if actual else 0.0
    print(f"kernel PC ranges ({len(ranges)}):")
    for lo, hi, name in ranges:
        print(f"  [{lo:#010x},{hi:#010x})  {name}")
    print(f"\nfirst kernel cycle : {first} (core)")
    print(f"last  kernel cycle : {last} (core)")
    print(f"actual cycles      : {actual} core  ({actual*2} tile)")
    print(f"instrs (issue slots): {instrs}")
    print(f"IPC (per-warp)     : {ipc:.3f}")
    print(f"essential MACs     : {macs}  (M*N*K = {M}*{N}*{K})")
    print(f"ideal accel cycles : {ideal:.0f}  (@ {PEAK_MAC_PER_CYC} MAC/cyc)")
    print(f"UTILIZATION        : {100*ideal/actual:.2f}%  (ideal/actual, core-clock)")

if __name__ == "__main__":
    main()
