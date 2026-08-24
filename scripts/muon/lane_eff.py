#!/usr/bin/env python3
"""SIMT lane efficiency + IPC from a Radiance trace-db.

Reads the `inst` table of a Verilator trace-db (RadianceSingleClusterConfig) and reports,
per cluster, three reproducible SIMT metrics over the recorded instructions:

  lane-eff   = mean(popcount(lane_mask & 0xFFFF)) / 16
               -> average fraction of the 16 SIMT thread-lanes active per issued instruction
                  (16 = numLanes, the lane_mask width; RadianceConfigs.scala WithSIMTConfig).
  IPC        = instructions / (max_cycle - min_cycle), aggregated over the cluster's cores.
               peak issue = 2 (one slot per Muon core, numCores=2).
  issue-util = 100 * IPC / 2.

Optionally restrict to a cycle window with --lo/--hi (e.g. to exclude a verify tail).

Usage:  lane_eff.py TRACE.sqlite [--lo N] [--hi N]
"""
import argparse
import sqlite3

LANE_BITS = 16          # numLanes / lane_mask width
PEAK_ISSUE = 2          # numCores (one issue slot per core)


def popcount16(x):
    return bin(x & 0xFFFF).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--lo", type=int, default=None, help="min cycle (inclusive)")
    ap.add_argument("--hi", type=int, default=None, help="max cycle (inclusive)")
    args = ap.parse_args()

    con = sqlite3.connect(args.trace)
    where = []
    if args.lo is not None:
        where.append(f"cycle >= {args.lo}")
    if args.hi is not None:
        where.append(f"cycle <= {args.hi}")
    clause = (" where " + " and ".join(where)) if where else ""

    clusters = [r[0] for r in con.execute(
        f"select distinct cluster_id from inst{clause} order by cluster_id")]
    if not clusters:
        print("no instructions in window")
        return

    print(f"{'cluster':<9}{'insts':<12}{'lane-eff':<11}{'IPC':<9}{'issue-util':<12}{'span(cyc)'}")
    for cl in clusters:
        rows = con.execute(
            f"select lane_mask, cycle, core_id from inst where cluster_id={cl}"
            + (" and " + " and ".join(where) if where else "")
        ).fetchall()
        n = len(rows)
        lane_eff = sum(popcount16(r[0]) for r in rows) / (LANE_BITS * n)
        cyc = [r[1] for r in rows]
        span = max(cyc) - min(cyc)
        ipc = n / span if span else 0.0
        issue = 100 * ipc / PEAK_ISSUE
        print(f"{cl:<9}{n:<12}{lane_eff*100:<11.1f}{ipc:<9.3f}{issue:<12.1f}{span}")


if __name__ == "__main__":
    main()
