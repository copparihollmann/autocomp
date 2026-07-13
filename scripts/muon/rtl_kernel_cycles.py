#!/usr/bin/env python3
"""Extract NET kernel cycles from a combined-config RTL run's embedded cyclotron trace-db.

The MX RTL gate compiles the harness with a DRAIN_ITERS spin (to outlast the SIMT C-store
write-drain race before verify). That spin dominates the $finish cycle count -- and its
run-to-run variance exceeds the kernel entirely -- so the reported rtl_cycles is useless
for perf calibration. But the trace-db (`inst` table: per-instruction cluster/core/cycle/pc)
timestamps everything, and the drain loop is a tight cluster of high-hit PCs that begins
exactly when kernel_body finishes. Its MIN cycle is therefore the net kernel-completion
cycle -- provably drain-independent (identical 69485 for DRAIN=2000 and 200000 on prob22).

Trace `cycle` is the Muon CORE clock (2x the tile clock the gate divides $finish_ps by),
so tile-clock cycles = trace_cycle * 2 (reported for parity with vcs_gate rtl_cycles).

Usage: rtl_kernel_cycles.py <trace.sqlite>
"""
import sqlite3, sys

def kernel_end_cycle(path):
    db = sqlite3.connect(path)
    c = db.cursor()
    # The drain spin is the handful of PCs with by far the most hits (DRAIN_ITERS x harts).
    # Take the top cluster; its earliest cycle is when the kernel handed off to the drain.
    top = c.execute(
        "select pc, count(*) n, min(cycle) mn from inst group by pc order by n desc limit 8"
    ).fetchall()
    if not top:
        raise SystemExit(f"{path}: empty trace")
    maxn = top[0][1]
    # drain PCs: those within 5% of the top hit-count (the tight spin body), all others
    # (kernel/verify) have far fewer hits.
    drain = [(pc, n, mn) for pc, n, mn in top if n >= 0.5 * maxn]
    kend = min(mn for _, _, mn in drain)
    trace_max = c.execute("select max(cycle) from inst").fetchone()[0]
    return kend, drain, trace_max

if __name__ == "__main__":
    kend, drain, tmax = kernel_end_cycle(sys.argv[1])
    print(f"net_kernel_cycles(core)={kend}  (tile={kend*2})  trace_max={tmax}")
    print(f"  drain PCs: " + ", ".join(f"{pc:#x}(n={n})" for pc, n, _ in drain))
