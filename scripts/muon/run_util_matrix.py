#!/usr/bin/env python3
"""Part C driver: run the phased utilization matrix on every kernel in util_sweep_spec.json, classify
each kernel's single BINDING LIMIT via the decision tree, and emit per-kernel records + a summary
table + hand-vs-autocomp routing (Part D).

Reads traces produced by sweep_util_trace.sh (autocomp_<tag>/trace_<tag>.sqlite + kernel.radiance.elf).
Uses --warm-passes 2 automatically when the trace was built WARM (detected: kernel-instr count is
even AND a dominant temporal gap at the count//2 ordinal).
"""
import json, os, sys, sqlite3, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec_mod = importlib.util.spec_from_file_location("hup", os.path.join(HERE, "hw_util_phased.py"))
hup = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(hup)

RK = "/scratch/agustin/projects/radiance-kernels/kernels"
BW_DRAM, BW_SMEM_C = hup.BW_DRAM, hup.BW_SMEM_CONTENDED


def detect_warm(elf, trace):
    """Return 2 if the trace looks double-scheduled (even kernel-instr count with a dominant gap at
    the count//2 ordinal), else 1."""
    funcs, _ = hup.parse_symbols(elf)
    kern = hup.kernel_region(funcs)
    cur = sqlite3.connect(f"file:{trace}?mode=ro", uri=True).cursor()
    kw = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi in kern)
    cyc = [r[0] for r in cur.execute(f"select cycle from inst where ({kw}) order by cycle")]
    if len(cyc) < 4:
        return 1
    gaps = [(cyc[i + 1] - cyc[i], i) for i in range(len(cyc) - 1)]
    biggest_gap, at = max(gaps)
    median_gap = sorted(g for g, _ in gaps)[len(gaps) // 2] or 1
    # double-schedule signature: the largest gap is near the middle and dwarfs the typical gap
    if biggest_gap > 20 * median_gap and abs(at - len(cyc) // 2) < len(cyc) * 0.15:
        return 2
    return 1


def classify(d, k):
    """Decision tree (top-down, first match; over the compute window). Returns (limit, runners)."""
    mx, si, r = d["mx"], d["simt"], d["rad"]
    U_c = (mx.get("u_compute") or si.get("u_compute") or 0) / 100.0
    # DRAM-BW util over the whole kernel (bytes / cyc / peak)
    bw = (k.get("bytes", 0) / d["kcyc"] / BW_DRAM) if k.get("bytes") else 0
    issue = (si.get("issue_util") or 0) / 100.0
    overhead = 1.0 - r["busy_any_frac"] / 100.0        # idle fraction = launch/config/barrier
    cands = [("COMPUTE-bound", U_c), ("DRAM-BW-bound", bw), ("ISSUE-WIDTH-bound", issue),
             ("FIXED-OVERHEAD-bound", overhead)]
    cands.sort(key=lambda x: -x[1])
    if U_c >= 0.80:
        limit = "COMPUTE-bound"
    elif bw >= 0.70:
        limit = "DRAM-BW-bound"
    elif si.get("issue_util") and issue >= 0.80:
        limit = "ISSUE-WIDTH-bound"
    elif overhead > 0.30:
        limit = "FIXED-OVERHEAD-bound"
    else:
        limit = "LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)"
    runners = [f"{n} {100*v:.0f}%" for n, v in cands[:2]]
    return limit, runners


ROUTE = {
    "COMPUTE-bound": ("Hand", "drop precision fp8->fp6/fp4 (peak 256->512) or larger tiles; dtype/geometry invisible to autocomp"),
    "DRAM-BW-bound": ("autocomp + RTL-gate", "gmem model prices DRAM round-trips; ranking imperfect -> RTL is final judge"),
    "ISSUE-WIDTH-bound": ("autocomp", "fewer instrs -> fewer cyclotron cycles; gradient visible; RTL-gate top-K"),
    "FIXED-OVERHEAD-bound": ("Hand", "hoist config, DMA-vs-SIMT move-out, amortize/fuse; restructuring invisible to issue-count"),
    "LATENCY-bound (below all ceilings; needs overlap/prefetch/double-buffer)":
        ("Hand + RTL-gate", "canonical cyclotron blind spot (overlap/double-buffer); don't spend $ on search"),
}


def main():
    spec = json.load(open(os.path.join(HERE, "util_sweep_spec.json")))
    rows = []
    for k in spec["kernels"]:
        tag = k["tag"]
        elf = f"{RK}/autocomp_{tag}/kernel.radiance.elf"
        trace = f"{RK}/autocomp_{tag}/trace_{tag}.sqlite"
        if not (os.path.exists(elf) and os.path.exists(trace)):
            print(f"### {k['label']}: NO TRACE YET ({tag})\n")
            rows.append((k, None, None, None))
            continue
        wp = detect_warm(elf, trace)
        d = hup.measure(elf, trace, k["engine"], k.get("macs", 0), k.get("flops", 0),
                        k.get("bytes", 0), k.get("fmt", "fp8"), k.get("tile_macs", 0),
                        k["label"], wp)
        print("#" * 90)
        hup.report_single(d)
        limit, runners = classify(d, k)
        route = ROUTE.get(limit, ("Hand + RTL-gate", "re-instrument"))
        print(f">>> BINDING LIMIT: {limit}  | runners-up: {runners}")
        print(f">>> ROUTE: {route[0]} — {route[1]}\n")
        rows.append((k, d, limit, route))

    # summary table
    print("\n" + "=" * 90 + "\nSUMMARY (Part C+D)\n" + "=" * 90)
    hdr = f"{'kernel':<24}{'eng':<6}{'U_compute':>10}{'U_kernel':>10}{'warm/cold':>16}{'binding limit':<24}{'route'}"
    print(hdr); print("-" * len(hdr))
    for k, d, limit, route in rows:
        if d is None:
            print(f"{k['label']:<24}{k['engine']:<6}{'(no trace)':>10}"); continue
        mx, si = d["mx"], d["simt"]
        uc = mx.get("u_compute") or si.get("u_compute") or 0
        uk = mx.get("u_kernel") or si.get("u_kernel") or 0
        wc = f"{d['kcyc']}/{d['cold_penalty']}" if d.get("cold_penalty") is not None else f"{d['kcyc']}/—"
        print(f"{k['label']:<24}{k['engine']:<6}{uc:>9.2f}%{uk:>9.2f}%{wc:>16}{limit[:23]:<24}{route[0]}")


if __name__ == "__main__":
    main()
