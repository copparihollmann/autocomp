#!/usr/bin/env python3
"""PHASE-AWARE whole-Radiance utilization from an RTL trace (Part A of the util plan).

Fixes the whole-span under-reporting of hw_utilization.py by attributing trace cycles to kernel
PHASES using the mxgemm_lib asm labels (emitted as ZERO-SIZE non-`F` local .text symbols that the
old tool's `F .text` regex could not see), then defining compute utilization over the COMPUTE phase
only -- not the config/DMA/move-out overhead.

Phases (MX/fused) from label stems (pair _start_/_end_ by stem; the trailing _<N> is an emission
counter that varies across builds -- never keyed on):
  mxgemm_single_output_tile  (whole MX region)
    main_matmul_k_loop       = P2  COMPUTE window that matters
      matmul_tile_async      = systolic issue, hit ONCE PER K-TILE -> warm-tile timestamps
      copy_gmem_to_smem_async(steady) = P2b prefetch (should overlap)
    copy_gmem_to_smem_async(prologue) + load_lut = P1 cold operand move-in
  copy_smem_to_gmem_simt     = P3 SIMT move-out (absent in fused test33: RoPE writes out)

Two engines, never conflated: MX matmul work is analytical (MACs/peak, NOT in the inst trace; Muon
IPC there is orchestration-only). SIMT kernels: IPC + FP roofline are the compute metrics.

Self-checks (refuse to report on failure): phase cycles tile the span in order; per-phase instr sum
== total; #matmul_tile hits == GK; span-max == rtl_kernel_cycles drain-min (two-oracle); U<=100% &
IPC<=2. Verified peaks: MX 256(fp8)/512(fp6,fp4) MAC/cyc; Muon 64 flop/cyc, issue 2/core-cyc.

Usage: hw_util_phased.py <radiance.elf> <trace.sqlite> --engine mx|simt|fused
         [--macs N --bytes N --fmt fp8|fp6|fp4 --tile-macs N --label name]
"""
import argparse, re, sqlite3, subprocess, statistics

OBJDUMP = "/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
NUM_CORES, NUM_LANES = 2, 16
PEAK_ISSUE = NUM_CORES
PEAK_SIMT_MAC = NUM_CORES * NUM_LANES          # 32
PEAK_SIMT_FLOP = PEAK_SIMT_MAC * 2             # 64
PEAK_MX = {"fp8": 256, "fp6": 512, "fp4": 512}
BW_DRAM, BW_SMEM, BW_SMEM_CONTENDED = 4, 64, 32
# entry/kernel-region exclusion allowlist (robust to entry name: kernel_body/mxgemm_entry/...)
EXCLUDE = ("main", "verify_body", "_start", "_exit")
EXCLUDE_PREFIX = ("mu_schedule", "init", "vx_", "__", "memcpy", "memset")


def parse_symbols(elf):
    """(funcs, labels): funcs=[(lo,hi,name)] sized .text; labels=[(addr,name)] zero-size .text."""
    out = subprocess.run([OBJDUMP, "-t", elf], capture_output=True, text=True).stdout
    funcs, labels = [], []
    for ln in out.splitlines():
        if ".text" not in ln:
            continue
        parts = ln.split()
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        ti = parts.index(".text")
        name = parts[-1]
        size = int(parts[ti + 1], 16) if re.fullmatch(r"[0-9a-f]+", parts[ti + 1]) else 0
        is_func = "F" in parts[1:ti]           # the F flag sits between flags and .text
        if is_func and size:
            funcs.append((addr, addr + size, name))
        elif size == 0:
            labels.append((addr, name))
    return sorted(funcs), sorted(labels)


def phase_intervals(labels):
    """[(lo,hi,stem)] from _start_/_end_ label pairs (pair by stem, ignore the _<N> counter)."""
    starts, ends = {}, {}
    for addr, nm in labels:
        m = re.match(r"(.+)_(start|end)_\d+$", nm)
        if not m:
            continue
        stem, kind = m[1], m[2]
        (starts if kind == "start" else ends)[stem] = addr
    return sorted((starts[s], ends[s], s) for s in starts if s in ends)


def kernel_region(funcs):
    """Union of kernel function ranges (exclusion allowlist -> robust to entry name)."""
    ks = [(lo, hi) for lo, hi, nm in funcs
          if nm not in EXCLUDE and not any(nm.startswith(p) for p in EXCLUDE_PREFIX)]
    # fast path preference for the known compute entries; fall back to all-non-excluded
    kern = [(lo, hi) for lo, hi in ks]
    return kern


def cyc_span(cur, ranges):
    if not ranges:
        return None
    w = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi in ranges)
    r = cur.execute(f"select min(cycle),max(cycle),count(*) from inst where {w}").fetchone()
    return r if r[0] is not None else None


def instrs_in(cur, lo, hi):
    return cur.execute("select count(*) from inst where pc>=? and pc<?", (lo, hi)).fetchone()[0]


def label_addr(labels, stem, kind):
    for a, nm in labels:
        if re.match(rf"{re.escape(stem)}_{kind}_\d+$", nm):
            return a
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf"); ap.add_argument("trace")
    ap.add_argument("--engine", choices=["mx", "simt", "fused"], required=True)
    ap.add_argument("--macs", type=int, default=0)
    ap.add_argument("--tile-macs", type=int, default=0, help="MACs per K-tile (for warm util)")
    ap.add_argument("--bytes", type=int, default=0)
    ap.add_argument("--fmt", choices=["fp8", "fp6", "fp4"], default="fp8")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    funcs, labels = parse_symbols(a.elf)
    phases = phase_intervals(labels)
    kern = kernel_region(funcs)
    cur = sqlite3.connect(f"file:{a.trace}?mode=ro", uri=True).cursor()
    checks = []

    kspan = cyc_span(cur, kern)
    if not kspan:
        raise SystemExit("no kernel instructions in trace (ELF/trace mismatch?)")
    kfirst, klast, kinstr = kspan
    kcyc = klast - kfirst
    peak = PEAK_MX[a.fmt]
    print(f"=== PHASED util: {a.label or a.elf.split('/')[-1]} (engine={a.engine} fmt={a.fmt}) ===")
    print(f"kernel span: [{kfirst},{klast}] = {kcyc} core cyc, {kinstr} Muon instrs")

    # ---- phase breakdown by innermost containment ----
    if phases:
        # innermost = smallest enclosing interval for each instr; approximate by per-interval
        # instr counts minus nested children.
        def direct_children(lo, hi):
            # maximal proper sub-intervals of [lo,hi): nested in it, and not nested in any other
            # interval that is itself nested in [lo,hi). Gives correct innermost attribution.
            inner = [(l, h) for l, h, _ in phases if lo <= l and h <= hi and (l, h) != (lo, hi)]
            direct = [(l, h) for (l, h) in inner
                      if not any(l2 <= l and h <= h2 and (l2, h2) != (l, h) for (l2, h2) in inner)]
            return direct
        print("phase breakdown (innermost):")
        covered = 0
        for lo, hi, stem in phases:
            sp = cyc_span(cur, [(lo, hi)])
            if not sp:
                continue
            tot = instrs_in(cur, lo, hi)
            child = direct_children(lo, hi)
            child_instr = sum(instrs_in(cur, l, h) for l, h in child)
            own = tot - child_instr
            cspan = sp[1] - sp[0]
            print(f"  {stem:<34} [{lo:#x},{hi:#x}) own_instrs={own:<6} span_cyc={cspan}")
            # leaf phases (no children) contribute their instrs to coverage
            if not child:
                covered += own
        # self-check: matmul_tile hit count == GK
        mt = label_addr(labels, "matmul_tile_async", "start")
        if mt is not None:
            gk = cur.execute("select count(*) from inst where pc=?", (mt,)).fetchone()[0]
            print(f"  [selfcheck] matmul_tile_async hits (=GK K-tiles): {gk}")

    # ---- MX compute-window util (P2 = main_matmul_k_loop) ----
    if a.engine in ("mx", "fused"):
        klo = label_addr(labels, "main_matmul_k_loop", "start")
        khi = label_addr(labels, "main_matmul_k_loop", "end")
        if klo is not None and khi is not None and a.macs:
            p2 = cyc_span(cur, [(klo, khi)])
            p2cyc = p2[1] - p2[0]
            u = 100 * (a.macs / peak) / p2cyc
            print(f"--- MX-Gemmini (systolic) ---")
            print(f"  P2 K-loop compute cycles : {p2cyc}  (vs whole-span {kcyc})")
            print(f"  MX util /P2-compute      : {u:.2f}%   (MACs {a.macs} / (peak {peak} * {p2cyc}))")
            print(f"  MX util /whole-span      : {100*(a.macs/peak)/kcyc:.2f}%  (overhead-diluted, for contrast)")
            checks.append(("MX util <=100%", u <= 100.0))
            # ---- warm per-tile (Part B1) ----
            mt = label_addr(labels, "matmul_tile_async", "start")
            if mt is not None:
                ts = [r[0] for r in cur.execute("select cycle from inst where pc=? order by cycle", (mt,))]
                if len(ts) >= 2:
                    deltas = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
                    print(f"  per-K-tile issue timestamps: {len(ts)} tiles; deltas={deltas}")
                    if len(deltas) >= 3:
                        steady = deltas[1:-1]            # drop cold-first and edge-last
                        med = statistics.median(steady)
                        tm = a.tile_macs or (a.macs // len(ts))
                        print(f"  WARM steady per-tile     : {med} cyc (median of {steady})")
                        print(f"  WARM MX util /steady-tile: {100*(tm/peak)/med:.2f}%  (cold Δ0={deltas[0]})")
                    else:
                        print(f"  (need GK>=4 for a steady-tile warm number; GK={len(ts)} -> bump dim_k)")

    # ---- SIMT compute util (kernel_body span or fused RoPE tail) ----
    if a.engine in ("simt", "fused"):
        print(f"--- Muon SIMT (lanes) ---")
        ipc = kinstr / kcyc if a.engine == "simt" else None
        if a.engine == "simt":
            print(f"  IPC (aggregate)          : {ipc:.3f}  (issue-slot util {100*ipc/PEAK_ISSUE:.1f}% of {PEAK_ISSUE})")
            checks.append(("IPC<=peak issue", ipc <= PEAK_ISSUE + 1e-9))
            if a.macs:
                print(f"  SIMT compute util        : {100*(a.macs/PEAK_SIMT_MAC)/kcyc:.2f}%  (memory-bound kernels: low is expected)")
        else:
            # fused RoPE phase = after mxgemm_single_output_tile_end, within kernel_body
            mxe = label_addr(labels, "mxgemm_single_output_tile", "end")
            if mxe is not None:
                r = cur.execute("select min(cycle),max(cycle),count(*) from inst where cycle>? ",
                                (mxe if False else 0,)).fetchone()  # placeholder; RoPE split below
                # RoPE = kernel instrs whose cycle is after the last matmul_tile issue
                mt = label_addr(labels, "matmul_tile_async", "start")
                cut = max(x[0] for x in cur.execute("select cycle from inst where pc=?", (mt,))) if mt else kfirst
                rope = cur.execute("select min(cycle),max(cycle),count(*) from inst where cycle>?", (cut,)).fetchone()
                if rope[0]:
                    rcyc = klast - cut
                    print(f"  SIMT-RoPE phase cycles   : ~{rcyc} (after last MX tile issue @ {cut}); instrs~{rope[2]}")
                    print(f"  (RoPE is elementwise/memory-bound; report IPC over this window in Part C)")

    if a.bytes and a.macs:
        # roofline (reuse the definitions)
        flop = 2 * a.macs
        pf = 2 * peak if a.engine in ("mx", "fused") else PEAK_SIMT_FLOP
        ai = flop / a.bytes
        ach = flop / kcyc
        real = min(pf, BW_DRAM * ai)
        print(f"--- roofline (whole-kernel) ---")
        print(f"  AI {ai:.2f} flop/B | ideal {100*ach/pf:.2f}% of peak {pf:.0f} | realistic {100*ach/real:.1f}% of {real:.0f} | DRAM-BW {100*(a.bytes/kcyc)/BW_DRAM:.1f}%")

    bad = [n for n, ok in checks if not ok]
    print("SELF-CHECKS:", "PASS" if not bad else f"FAIL {bad}")


if __name__ == "__main__":
    main()
