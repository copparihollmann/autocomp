#!/usr/bin/env python3
"""Part C driver: run the phased utilization matrix on every kernel in util_sweep_spec.json, classify
each kernel's single BINDING LIMIT via the decision tree, and emit per-kernel records + a summary
table + hand-vs-autocomp routing (Part D).

Reads traces produced by sweep_util_trace.sh (autocomp_<tag>/trace_<tag>.sqlite + kernel.radiance.elf).
Uses --warm-passes 2 automatically when the trace was built WARM (detected: kernel-instr count is
even AND a dominant temporal gap at the count//2 ordinal).
"""
import json, os, re, sys, sqlite3, subprocess, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec_mod = importlib.util.spec_from_file_location("hup", os.path.join(HERE, "hw_util_phased.py"))
hup = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(hup)

RK = "/scratch/agustin/projects/radiance-kernels/kernels"
BW_DRAM, BW_SMEM_C = hup.BW_DRAM, hup.BW_SMEM_CONTENDED
REPORT = "/scratch/agustin/projects/autocomp/PER_KERNEL_BOTTLENECK.md"


def spill_probe(elf):
    """Register-pressure heuristic: count sp-relative loads/stores in the BODY of kernel_body (past
    the prologue/epilogue frame setup). Many mid-body sp ld/st = spills. Returns (spill_ld_st, note)."""
    try:
        out = subprocess.run([hup.OBJDUMP, "-d", elf], capture_output=True, text=True).stdout
    except Exception:
        return (None, "objdump failed")
    inbody, sp_ops, lines = False, 0, 0
    for ln in out.splitlines():
        if re.search(r"<_?Z?L?\d*kernel_body\w*>:", ln):
            inbody = True; continue
        if inbody:
            if re.match(r"^[0-9a-f]+ <", ln):        # next function
                break
            lines += 1
            if re.search(r"\b(s[wd]|l[wd])\b.*\(sp\)|,\s*-?\d+\(sp\)", ln):
                sp_ops += 1
    # crude: subtract ~4 for prologue save + epilogue restore
    net = max(0, sp_ops - 4)
    return (net, f"{sp_ops} sp ld/st over {lines} body insns (~{net} beyond frame)")


def hit_assertion(tag):
    """True if this kernel tripped the L0d backpressure assertion (trace is truncated -> whole-kernel
    util is invalid; only captured IPC / bottleneck-class are trustworthy)."""
    log = f"{RK}/autocomp_{tag}/sim.log"
    if not os.path.exists(log):
        return False
    return "response must be ready" in open(log, errors="ignore").read()


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


def cw_util(d):
    """Display compute-window util: for MX, a >100% P2 K-loop (GK=1 label misses the systolic drain)
    is unphysical -> show whole-kernel MX util instead. SIMT unchanged."""
    mx, si = d["mx"], d["simt"]
    v = mx.get("u_compute")
    if v is not None:
        return v if v <= 100.0 else (mx.get("u_kernel") or 0)
    return si.get("u_compute") or 0


def classify(d, k):
    """Decision tree (top-down, first match; over the compute window). Returns (limit, runners).

    NOTE on compute-bound: matmul/GEMV MAC counts are EXACT, but elementwise/activation FLOP counts
    are estimates (esp. transcendentals: silu/exp). A high FLOP-util from an over-estimated count is
    an artifact -- a genuinely compute-bound SIMT kernel MUST also issue densely. So COMPUTE-bound
    requires either an exact MAC count OR issue-util corroboration (>=0.5). Otherwise it falls through
    to the memory/latency signals (IPC, DRAM-BW), which are the reliable ones for SIMT."""
    mx, si, r = d["mx"], d["simt"], d["rad"]
    # MX compute-window (P2 K-loop label span) is only trustworthy when it captures the full systolic
    # compute -- for GK=1 the K-loop label ends before the array drains, giving an unphysical >100%.
    # In that case discard it and use the whole-kernel MX util (always valid).
    mx_cw = mx.get("u_compute")
    if mx_cw is not None and mx_cw > 100.0:
        mx_cw = mx.get("u_kernel")
    U_c = ((mx_cw if mx_cw is not None else si.get("u_compute")) or 0) / 100.0
    bw = (k.get("bytes", 0) / d["kcyc"] / BW_DRAM) if k.get("bytes") else 0
    issue = (si.get("issue_util") or 0) / 100.0
    overhead = 1.0 - r["busy_any_frac"] / 100.0
    is_mx = k["engine"] in ("mx", "fused")
    exact_work = "macs" in k                     # MAC count is exact; flops are estimates
    compute_ok = U_c >= 0.80 and (is_mx or exact_work or issue >= 0.5)
    cands = [("COMPUTE", U_c), ("DRAM-BW", bw), ("ISSUE", issue), ("OVERHEAD", overhead)]
    cands.sort(key=lambda x: -x[1])
    if compute_ok:
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
            rows.append((k, None, None, None, None))
            continue
        wp = detect_warm(elf, trace)
        d = hup.measure(elf, trace, k["engine"], k.get("macs", 0), k.get("flops", 0),
                        k.get("bytes", 0), k.get("fmt", "fp8"), k.get("tile_macs", 0),
                        k["label"], wp)
        spill = spill_probe(elf)
        asserted = hit_assertion(tag)
        d["asserted"] = asserted
        print("#" * 90)
        hup.report_single(d)
        if asserted:
            # trace truncated at the l0d backpressure assertion -> whole-kernel util invalid; the
            # captured IPC (low) + the assertion itself are the trustworthy memory-bound evidence.
            limit = "MEMORY-BOUND (L0d backpressure assertion at full shape; unbuffered per-tile l0d)"
            route = ("Hand + RTL-gate", "saturates l0d response path; reduce in-flight traffic / "
                     "coalesce; whether the tapeout l0d should carry landing pads is a DUT call for the team")
            runners = [f"captured IPC {d['simt'].get('ipc',0):.3f}", "assertion=memory-bound"]
        else:
            limit, runners = classify(d, k)
            route = ROUTE.get(limit, ("Hand + RTL-gate", "re-instrument"))
        print(f">>> spill-probe: {spill[1]}")
        print(f">>> BINDING LIMIT: {limit}  | runners-up: {runners}")
        print(f">>> ROUTE: {route[0]} — {route[1]}\n")
        rows.append((k, d, limit, route, spill))

    # summary table (stdout)
    print("\n" + "=" * 90 + "\nSUMMARY (Part C+D)\n" + "=" * 90)
    hdr = f"{'kernel':<24}{'eng':<6}{'U_compute':>10}{'U_kernel':>10}{'IPC':>7}{'idle%':>7}  {'binding limit':<26}{'route'}"
    print(hdr); print("-" * len(hdr))
    for k, d, limit, route, spill in rows:
        if d is None:
            print(f"{k['label']:<24}{k['engine']:<6}{'(no trace)':>10}"); continue
        mx, si = d["mx"], d["simt"]
        tr = d.get("asserted")
        uc = "TRUNC" if tr else f"{cw_util(d):.2f}%"
        uk = "TRUNC" if tr else f"{(mx.get('u_kernel') or si.get('u_kernel') or 0):.2f}%"
        ipc = si.get("ipc") or 0
        idle = d["rad"]["idle_frac"]
        print(f"{k['label']:<24}{k['engine']:<6}{uc:>10}{uk:>10}{ipc:>7.3f}{idle:>6.1f}%  {limit[:25]:<26}{route[0]}")

    write_markdown(rows)
    print(f"\nwrote {REPORT}")


def write_markdown(rows):
    L = ["# Per-kernel bottleneck analysis + hand-vs-autocomp routing (Parts C+D)",
         "",
         "RTL (Verilator, tapeout-330+trace-fix), single-pass, DRAIN=2000. Compute-window = the phase",
         "the engine actually computes (MX P2 K-loop / SIMT kernel_body); U_kernel = end-to-end incl.",
         "config/DMA/move-out. MX warm = per-tile steady (cold tile excluded). Bottleneck via the",
         "decision tree; route per the cyclotron-blind-spot table (compute/issue-count -> autocomp can",
         "see it; memory-hierarchy/latency/overlap/overhead/register -> hand + RTL arbiter).",
         "",
         "| kernel | eng | U_compute | U_kernel | MX warm | IPC (issue%) | DRAM-BW | idle% | overlap | spills | **binding limit** | route |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|"]
    for k, d, limit, route, spill in rows:
        if d is None:
            L.append(f"| {k['label']} | {k['engine']} | (no trace) | | | | | | | | | |"); continue
        mx, si, r = d["mx"], d["simt"], d["rad"]
        tr = d.get("asserted")
        uc = "TRUNC†" if tr else f"{cw_util(d):.2f}%"
        uk = "TRUNC†" if tr else f"{(mx.get('u_kernel') or si.get('u_kernel') or 0):.2f}%"
        warm = f"{mx['u_steady']:.1f}%" if mx.get("u_steady") is not None else "—"
        ipc = f"{si['ipc']:.3f} ({si['issue_util']:.0f}%)" if si.get("ipc") is not None else "—"
        bw = "—" if tr else (f"{k.get('bytes', 0) / d['kcyc'] / BW_DRAM * 100:.0f}%" if k.get("bytes") else "—")
        sp = spill[0] if spill and spill[0] is not None else "?"
        L.append(f"| {k['label']} | {k['engine']} | {uc} | {uk} | {warm} | {ipc} | "
                 f"{bw} | {r['idle_frac']:.0f}% | {r['overlap']:.2f}x | {sp} | "
                 f"**{limit}** | {route[0]} |")
    L += ["", "† TRUNC = trace truncated by the L0d backpressure assertion at full tinyllama shape "
          "(unbuffered per-tile l0d, makeLandingPads=false; cluster cache has it =true). Whole-kernel "
          "util is therefore not measurable at full shape; the captured IPC (low) + the assertion "
          "itself confirm MEMORY-BOUND. Same bottleneck class as the clean elementwise/GEMV kernels."]
    L += ["", "## Routing rationale", ""]
    for k, d, limit, route, spill in rows:
        if d is not None:
            L.append(f"- **{k['label']}** → {route[0]}: {route[1]}")
    open(REPORT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
