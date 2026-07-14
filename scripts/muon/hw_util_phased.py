#!/usr/bin/env python3
"""PHASE-AWARE whole-Radiance utilization from an RTL trace (Parts A+B of the util plan).

Reports utilization at MULTIPLE SCOPES x BOTH ENGINES + a WHOLE-RADIANCE combined block, filling
only the cells that are physically meaningful for the kernel (a pure-MX matmul has no SIMT compute
util; a SIMT elementwise kernel has no MX util). Every number is measured from the RTL trace; MX
work (not in the Muon inst trace) is analytical essential-MACs / peak over the measured window.

SCOPES (columns of "how much of what window"):
  inner / steady   warm per-tile, cold+edge K-tiles excluded (MX; needs GK>=4). The reachable ceiling.
  compute-window   the phase where the engine actually computes -- MX = P2 main_matmul_k_loop;
                   SIMT = kernel_body compute region. Excludes config/DMA/move-out overhead.
  whole-kernel     first->last kernel instr, INCLUDING config + operand-DMA + move-out. End-to-end.
  fused-layer      (fused kernels only) the union region spanning BOTH engines = whole kernel here;
                   for a single op the layer == the kernel.  Multi-op layers: use --layer manifest.

ENGINES (never conflated -- the trace records only Muon instrs):
  MX-Gemmini  16x16 systolic. U = essential_MACs / (PEAK_MX[fmt] * window_cyc). Analytical; Muon IPC
              on an MX kernel is ORCHESTRATION only (warps issue MMIO + move-out), not throughput.
  Muon-SIMT   2 cores x 16 lanes. U = essential_MACs / (32 * window) or FLOPs/64; + IPC/issue-util.
  Radiance    both engines over the whole-kernel window: per-engine ACTIVE fraction, any-engine busy,
              engine OVERLAP (>1x = genuine concurrency; 1x = serialized w/ idle silicon), and a
              throughput efficiency vs combined peak (mixed-precision op count -- see the printed NOTE).

Verified peaks: MX 256(fp8)/512(fp6,fp4) MAC/cyc; Muon 32 MAC/cyc = 64 flop/cyc; issue 2/core-cyc.
Memory BW (RTL-fitted): DRAM 4 B/cyc; SMEM 64 B/cyc (32 contended, serialize_cores=true).

Self-checks (refuse to report on failure): phase cycles tile the span; #matmul_tile hits == GK;
U<=100%; IPC<=2. A wrong --fmt trips the >100% failure loudly.

Usage:
  single kernel: hw_util_phased.py <radiance.elf> <trace.sqlite> --engine mx|simt|fused
                   [--macs N --flops N --bytes N --fmt fp8|fp6|fp4 --tile-macs N --label name]
  whole layer:   hw_util_phased.py --layer manifest.json     (JSON list of the above arg-sets)
"""
import argparse, json, re, sqlite3, subprocess, statistics

OBJDUMP = "/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
NUM_CORES, NUM_LANES = 2, 16
PEAK_ISSUE = NUM_CORES
PEAK_SIMT_MAC = NUM_CORES * NUM_LANES          # 32
PEAK_SIMT_FLOP = PEAK_SIMT_MAC * 2             # 64
PEAK_MX = {"fp8": 256, "fp6": 512, "fp4": 512}
BW_DRAM, BW_SMEM, BW_SMEM_CONTENDED = 4, 64, 32
EXCLUDE = ("main", "verify_body", "_start", "_exit")
EXCLUDE_PREFIX = ("mu_schedule", "init", "vx_", "__", "memcpy", "memset")


# ---------------------------------------------------------------- symbol / trace helpers
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
        is_func = "F" in parts[1:ti]
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
    return [(lo, hi) for lo, hi, nm in funcs
            if nm not in EXCLUDE and not any(nm.startswith(p) for p in EXCLUDE_PREFIX)]


def cyc_span(cur, ranges, floor=0):
    if not ranges:
        return None
    w = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi in ranges)
    r = cur.execute(f"select min(cycle),max(cycle),count(*) from inst where ({w}) and cycle>=?",
                    (floor,)).fetchone()
    return r if r[0] is not None else None


def instrs_in(cur, lo, hi, floor=0):
    return cur.execute("select count(*) from inst where pc>=? and pc<? and cycle>=?",
                       (lo, hi, floor)).fetchone()[0]


def label_addr(labels, stem, kind):
    for a, nm in labels:
        if re.match(rf"{re.escape(stem)}_{kind}_\d+$", nm):
            return a
    return None


def union_len(intervals):
    """Total length covered by a set of [lo,hi] cycle intervals (merged)."""
    iv = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    if not iv:
        return 0
    tot, clo, chi = 0, iv[0][0], iv[0][1]
    for lo, hi in iv[1:]:
        if lo <= chi:
            chi = max(chi, hi)
        else:
            tot += chi - clo
            clo, chi = lo, hi
    return tot + (chi - clo)


def pct(num, den):
    return 100.0 * num / den if den else 0.0


# ---------------------------------------------------------------- core measurement
def measure(elf, trace, engine, macs=0, flops=0, byts=0, fmt="fp8", tile_macs=0, label="",
            warm_passes=1):
    """Return a structured dict of every scope x engine number, measured from the RTL trace.

    warm_passes=2: split a double-scheduled trace at the count//2 ordinal and measure only the warm
    2nd pass. CAVEAT (measured): the Muon scheduler launches the 8 warp-slots staggered and interleaves
    the two dispatches, so on real SIMT kernels the two passes do NOT form two clean temporal clusters
    -- the count//2 split can land mid-pass. Use warm_passes=2 ONLY when a clean split is confirmed.
    For rigorous warm numbers prefer: MX -> the per-tile steady window (always correct, phase-labelled);
    SIMT memory-bound kernels -> the whole-kernel number IS the honest cost (cold-DRAM is the actual
    bottleneck, not an artifact to hide). warm_passes=1 (default): single pass, floor=0."""
    funcs, labels = parse_symbols(elf)
    phases = phase_intervals(labels)
    kern = kernel_region(funcs)
    cur = sqlite3.connect(f"file:{trace}?mode=ro", uri=True).cursor()

    floor, cold_cyc_penalty = 0, None
    full = cyc_span(cur, kern, 0)
    if not full:
        raise SystemExit(f"no kernel instructions in trace ({elf} / {trace} mismatch?)")
    if warm_passes == 2 and full[2] >= 2:
        kw = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi in kern)
        mid = cur.execute(f"select cycle from inst where ({kw}) order by cycle limit 1 offset ?",
                          (full[2] // 2,)).fetchone()[0]
        floor = mid
        cold_cyc_penalty = mid - full[0]     # pass-1 (cold) span up to the 2nd-pass start

    kspan = cyc_span(cur, kern, floor)
    kfirst, klast, kinstr = kspan
    kcyc = klast - kfirst
    peak_mx = PEAK_MX[fmt]
    d = {"label": label or elf.split("/")[-1], "engine": engine, "fmt": fmt,
         "kfirst": kfirst, "klast": klast, "kinstr": kinstr, "kcyc": kcyc,
         "macs": macs, "flops": flops, "bytes": byts, "peak_mx": peak_mx,
         "warm_passes": warm_passes, "cold_penalty": cold_cyc_penalty,
         "phase_rows": [], "checks": [], "gk": None,
         "win": {}, "mx": {}, "simt": {}, "rad": {}}

    # ---- phase breakdown (innermost containment) ----
    if phases:
        def direct_children(lo, hi):
            inner = [(l, h) for l, h, _ in phases if lo <= l and h <= hi and (l, h) != (lo, hi)]
            return [(l, h) for (l, h) in inner
                    if not any(l2 <= l and h <= h2 and (l2, h2) != (l, h) for (l2, h2) in inner)]
        for lo, hi, stem in phases:
            sp = cyc_span(cur, [(lo, hi)], floor)
            if not sp:
                continue
            child = direct_children(lo, hi)
            own = instrs_in(cur, lo, hi, floor) - sum(instrs_in(cur, l, h, floor) for l, h in child)
            d["phase_rows"].append((stem, lo, hi, own, sp[1] - sp[0], sp[0], sp[1]))
        mt = label_addr(labels, "matmul_tile_async", "start")
        if mt is not None:
            d["gk"] = cur.execute("select count(*) from inst where pc=? and cycle>=?",
                                  (mt, floor)).fetchone()[0]

    # ---- MX windows ----
    mx_active = 0
    if engine in ("mx", "fused") and macs:
        klo = label_addr(labels, "main_matmul_k_loop", "start")
        khi = label_addr(labels, "main_matmul_k_loop", "end")
        if klo is not None and khi is not None:
            p2 = cyc_span(cur, [(klo, khi)], floor)
            p2cyc = p2[1] - p2[0]
            mx_active = p2cyc
            d["win"]["mx_compute"] = (p2cyc, p2[0], p2[1])
            u_comp = pct(macs / peak_mx, p2cyc)
            u_whole = pct(macs / peak_mx, kcyc)
            d["mx"] = {"peak": peak_mx, "u_compute": u_comp, "u_kernel": u_whole,
                       "compute_cyc": p2cyc, "u_steady": None, "steady_cyc": None, "cold_cyc": None}
            d["checks"].append(("MX util(compute)<=100%", u_comp <= 100.0 + 1e-9))
            # warm per-tile
            mt = label_addr(labels, "matmul_tile_async", "start")
            if mt is not None:
                ts = [r[0] for r in cur.execute(
                    "select cycle from inst where pc=? and cycle>=? order by cycle", (mt, floor))]
                if len(ts) >= 4:
                    deltas = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
                    steady = deltas[1:-1]
                    med = statistics.median(steady)
                    tm = tile_macs or (macs // len(ts))
                    d["mx"].update(u_steady=pct(tm / peak_mx, med), steady_cyc=med,
                                   cold_cyc=deltas[0], deltas=deltas, gk=len(ts))
                    d["checks"].append(("MX util(steady)<=100%", d["mx"]["u_steady"] <= 100.0 + 1e-9))

    # ---- SIMT windows ----
    simt_compute = simt_move = 0
    if engine in ("simt", "fused"):
        if engine == "simt":
            ipc = kinstr / kcyc if kcyc else 0
            simt_compute = kcyc
            d["simt"] = {"ipc": ipc, "issue_util": pct(ipc, PEAK_ISSUE),
                         "compute_cyc": kcyc, "u_compute": None, "u_kernel": None}
            if macs:
                d["simt"]["u_compute"] = pct(macs / PEAK_SIMT_MAC, kcyc)
                d["simt"]["u_kernel"] = d["simt"]["u_compute"]
            elif flops:
                d["simt"]["u_compute"] = pct(flops / PEAK_SIMT_FLOP, kcyc)
                d["simt"]["u_kernel"] = d["simt"]["u_compute"]
            d["checks"].append(("IPC<=peak issue", ipc <= PEAK_ISSUE + 1e-9))
        else:  # fused: RoPE tail = SIMT instrs AFTER the MX region ends (issue != completion, so
               # the cut must be the MX-region END cycle, not the last tile issue -- else the RoPE
               # window overlaps the matmul drain and reports spurious concurrency + IPC>peak).
            mxreg = cyc_span(cur, [(label_addr(labels, "mxgemm_single_output_tile", "start"),
                                    label_addr(labels, "mxgemm_single_output_tile", "end"))], floor)
            cut = mxreg[1] if mxreg and mxreg[1] is not None else kfirst
            # RoPE instrs = KERNEL-region PCs only (never harness/verify/drain), warm-pass floor..cut
            kw = " or ".join(f"(pc>={lo} and pc<{hi})" for lo, hi in kern)
            rope = cur.execute(
                f"select min(cycle),max(cycle),count(*) from inst where cycle>? and ({kw})", (cut,)).fetchone()
            if rope[0] is not None:
                rcyc = klast - cut
                simt_compute = rcyc
                ripc = rope[2] / rcyc if rcyc else 0
                d["simt"] = {"ipc": ripc, "issue_util": pct(ripc, PEAK_ISSUE),
                             "compute_cyc": rcyc, "rope_instrs": rope[2], "cut": cut,
                             "u_compute": pct(flops / PEAK_SIMT_FLOP, rcyc) if flops else None,
                             "u_kernel": None}
                d["checks"].append(("fused RoPE IPC<=peak issue", ripc <= PEAK_ISSUE + 1e-9))
    # SIMT move-out on an MX kernel (copy_smem_to_gmem_simt) = SIMT lanes doing data movement
    mo_lo = label_addr(labels, "copy_smem_to_gmem_simt", "start")
    mo_hi = label_addr(labels, "copy_smem_to_gmem_simt", "end")
    if mo_lo is not None and mo_hi is not None:
        mo = cyc_span(cur, [(mo_lo, mo_hi)], floor)
        if mo and mo[0] is not None:
            simt_move = mo[1] - mo[0]
            d["simt"]["move_out_cyc"] = simt_move

    # ---- WHOLE-RADIANCE combined (over the whole-kernel window) ----
    mx_iv = [d["win"]["mx_compute"][1:]] if "mx_compute" in d["win"] else []
    simt_comp_iv = []
    if engine == "simt":
        simt_comp_iv = [(kfirst, klast)]
    elif engine == "fused" and simt_compute:
        simt_comp_iv = [(d["simt"]["cut"], klast)]
    simt_move_iv = [(mo[0], mo[1])] if (mo_lo is not None and simt_move) else []
    busy_compute = union_len(mx_iv + simt_comp_iv)
    busy_any = union_len(mx_iv + simt_comp_iv + simt_move_iv)
    sum_active = mx_active + simt_compute
    macs_mx = macs if engine in ("mx", "fused") else 0
    macs_simt = macs if engine == "simt" else 0
    flop_total = 2 * macs_mx + (2 * macs_simt if macs_simt else flops)
    combined_peak_mac = peak_mx + PEAK_SIMT_MAC
    combined_peak_flop = 2 * peak_mx + PEAK_SIMT_FLOP
    d["rad"] = {
        "window": kcyc,
        "mx_active": mx_active, "mx_active_frac": pct(mx_active, kcyc),
        "simt_compute": simt_compute, "simt_compute_frac": pct(simt_compute, kcyc),
        "simt_move": simt_move, "simt_move_frac": pct(simt_move, kcyc),
        "busy_compute_frac": pct(busy_compute, kcyc),
        "busy_any_frac": pct(busy_any, kcyc),
        "idle_frac": pct(kcyc - busy_any, kcyc),
        "overlap": (sum_active / busy_compute) if busy_compute else 0.0,
        "thru_eff": pct(flop_total, combined_peak_flop * kcyc),
        "combined_peak_flop": combined_peak_flop,
    }
    return d


# ---------------------------------------------------------------- reporting
def _u(x):
    return f"{x:6.2f}%" if isinstance(x, (int, float)) else f"{'—':>7}"


def report_single(d):
    print(f"=== PHASED util: {d['label']} (engine={d['engine']} fmt={d['fmt']}) ===")
    warm = f"  [WARM 2nd pass; cold pass-1 penalty {d['cold_penalty']} cyc]" if d.get("cold_penalty") is not None else ""
    print(f"kernel span: [{d['kfirst']},{d['klast']}] = {d['kcyc']} core cyc, {d['kinstr']} Muon instrs{warm}")
    if d["phase_rows"]:
        print("phase breakdown (innermost):")
        for stem, lo, hi, own, cyc, _, _ in d["phase_rows"]:
            print(f"  {stem:<34} [{lo:#x},{hi:#x}) own_instrs={own:<6} span_cyc={cyc}")
        if d["gk"] is not None:
            print(f"  [selfcheck] matmul_tile_async hits (=GK K-tiles): {d['gk']}")

    mx, si = d["mx"], d["simt"]
    print("\nSCOPE x ENGINE utilization matrix  (— = not meaningful for this kernel)")
    print(f"  {'scope':<16}{'MX-Gemmini':>12}{'Muon-SIMT':>12}   window_cyc / note")
    # inner/steady
    dash = lambda v: v if v is not None else "—"
    print(f"  {'inner/steady':<16}{_u(mx.get('u_steady')):>12}{_u(None):>12}"
          f"   {dash(mx.get('steady_cyc'))}  warm per-tile, cold Δ0={dash(mx.get('cold_cyc'))}")
    # compute-window
    cw = mx.get("compute_cyc") or si.get("compute_cyc") or "—"
    print(f"  {'compute-window':<16}{_u(mx.get('u_compute')):>12}{_u(si.get('u_compute')):>12}"
          f"   {cw}  P2 K-loop / SIMT body")
    # whole-kernel
    print(f"  {'whole-kernel':<16}{_u(mx.get('u_kernel')):>12}{_u(si.get('u_kernel')):>12}"
          f"   {d['kcyc']}  end-to-end (overhead-diluted)")
    if d["engine"] == "fused":
        print(f"  {'fused-layer':<16}{_u(mx.get('u_kernel')):>12}{_u(si.get('u_compute')):>12}"
              f"   {d['kcyc']}  both engines, one launch")
    if si.get("ipc") is not None:
        print(f"  Muon IPC {si['ipc']:.3f} (issue-slot {si['issue_util']:.1f}% of {PEAK_ISSUE})"
              + ("  — ORCHESTRATION only on the MX path" if d["engine"] == "mx" else ""))

    r = d["rad"]
    print("\nWHOLE-RADIANCE (both engines over the whole-kernel window):")
    print(f"  MX-compute active     : {r['mx_active']:>8} cyc  ({r['mx_active_frac']:.1f}% of {r['window']})")
    print(f"  SIMT-compute active   : {r['simt_compute']:>8} cyc  ({r['simt_compute_frac']:.1f}%)")
    if r["simt_move"]:
        print(f"  SIMT data-move active : {r['simt_move']:>8} cyc  ({r['simt_move_frac']:.1f}%)  [move-out, not FMA]")
    print(f"  any-engine busy       : {r['busy_any_frac']:.1f}%   (idle {r['idle_frac']:.1f}% = launch/config/barrier)")
    print(f"  engine overlap        : {r['overlap']:.2f}x  ({'CONCURRENT' if r['overlap']>1.05 else 'serialized — no concurrent compute'})")
    print(f"  throughput efficiency : {r['thru_eff']:.2f}% of combined peak {r['combined_peak_flop']:.0f} flop/cyc")
    print(f"     NOTE: combined peak sums MX({d['fmt']}) + SIMT(fp32) op-throughput; low value = idle")
    print(f"           silicon (the other engine), NOT numeric error. Overlap>1x needs true fusion.")

    # roofline at whole-kernel scope: --bytes is the DRAM footprint, so the DRAM ceiling and DRAM-BW
    # util are only meaningful end-to-end (the compute window runs SMEM-resident, near-zero DRAM
    # traffic -> it is compute-bound at its matrix %, above any cold-DRAM ceiling; not re-rooflined).
    if d["bytes"] and (d["macs"] or d["flops"]):
        flop = 2 * d["macs"] if d["macs"] else d["flops"]
        pf = 2 * d["peak_mx"] if d["engine"] in ("mx", "fused") else PEAK_SIMT_FLOP
        ai = flop / d["bytes"]
        ach = flop / d["kcyc"]
        real = min(pf, BW_DRAM * ai)
        bw = d["bytes"] / d["kcyc"]
        print(f"\nroofline (whole-kernel; AI {ai:.2f} flop/B DRAM; ridge DRAM {pf/BW_DRAM:.0f}, SMEM {pf/BW_SMEM:.0f})")
        print(f"  ideal {pct(ach,pf):.2f}% of {pf:.0f} | realistic(cold-DRAM) {pct(ach,real):.1f}% of {real:.0f} "
              f"| DRAM-BW {pct(bw,BW_DRAM):.1f}% of {BW_DRAM}")
        bind = ("COMPUTE-bound" if ach > 0.6 * pf else "DRAM-BW-bound" if bw > 0.6 * BW_DRAM
                else "OVERHEAD/LATENCY-bound (below both ceilings -> fuse/enlarge/overlap)")
        print(f"  binding limit (end-to-end): {bind}")

    bad = [n for n, ok in d["checks"] if not ok]
    print("\nSELF-CHECKS:", "PASS" if not bad else f"FAIL {bad}")
    return not bad


def report_layer(manifest):
    """Aggregate multiple kernel measurements into whole-layer / fused-layer utilization.

    On this single cluster the ops run SERIALLY (one kernel launch each), so layer cycles = sum of
    per-op kernel cycles and layer util(engine) = sum(essential work on that engine) / (peak * sum
    of the cycles where that engine is the compute engine)."""
    ops = []
    for m in manifest:
        d = measure(m["elf"], m["trace"], m["engine"], m.get("macs", 0), m.get("flops", 0),
                    m.get("bytes", 0), m.get("fmt", "fp8"), m.get("tile_macs", 0),
                    m.get("label", ""))
        ops.append(d)
        print(f"  · {d['label']:<28} {d['engine']:<6} {d['kcyc']:>8} cyc  "
              f"MX/compute {d['mx'].get('u_compute','—')}  SIMT/compute {d['simt'].get('u_compute','—')}")
    layer_cyc = sum(d["kcyc"] for d in ops)
    mx_cyc = sum(d["mx"].get("compute_cyc", 0) for d in ops)
    simt_cyc = sum(d["simt"].get("compute_cyc", 0) for d in ops)
    mx_macs = sum(d["macs"] for d in ops if d["engine"] in ("mx", "fused"))
    simt_macs = sum(d["macs"] for d in ops if d["engine"] == "simt")
    simt_flops = sum(d["flops"] for d in ops if d["engine"] == "simt")
    # each op uses ONE mx peak; if mixed fmt, weight by op. Use per-op peak for the analytic MAC sum.
    mx_peak_cycs = sum(d["peak_mx"] * d["mx"].get("compute_cyc", 0) for d in ops)
    print(f"\n=== WHOLE-LAYER utilization ({len(ops)} ops, serialized) ===")
    print(f"  layer cycles (Σ kernel) : {layer_cyc}")
    print(f"  MX   compute cyc {mx_cyc:>8} ({pct(mx_cyc,layer_cyc):.1f}% of layer) | "
          f"layer MX util(compute-cyc) {pct(mx_macs, mx_peak_cycs) if mx_peak_cycs else 0:.2f}%")
    simt_num = simt_macs / PEAK_SIMT_MAC if simt_macs else simt_flops / PEAK_SIMT_FLOP
    print(f"  SIMT compute cyc {simt_cyc:>8} ({pct(simt_cyc,layer_cyc):.1f}% of layer) | "
          f"layer SIMT util(compute-cyc) {pct(simt_num, simt_cyc) if simt_cyc else 0:.2f}%")
    # whole-layer throughput efficiency: total useful flop / (combined peak * layer cyc)
    flop_total = 2 * mx_macs + 2 * simt_macs + simt_flops
    comb = sum(2 * d["peak_mx"] for d in ops) / len(ops) + PEAK_SIMT_FLOP  # avg combined peak
    print(f"  any-engine busy         : {pct(mx_cyc+simt_cyc, layer_cyc):.1f}% of layer "
          f"(idle {pct(layer_cyc-mx_cyc-simt_cyc, layer_cyc):.1f}% = per-op launch/overhead)")
    print(f"  layer throughput eff.   : {pct(flop_total, comb*layer_cyc):.2f}% of combined peak")
    print(f"     (the levers: fuse ops to cut per-launch overhead + overlap MX & SIMT)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf", nargs="?"); ap.add_argument("trace", nargs="?")
    ap.add_argument("--engine", choices=["mx", "simt", "fused"])
    ap.add_argument("--macs", type=int, default=0)
    ap.add_argument("--flops", type=int, default=0)
    ap.add_argument("--tile-macs", type=int, default=0)
    ap.add_argument("--bytes", type=int, default=0)
    ap.add_argument("--fmt", choices=["fp8", "fp6", "fp4"], default="fp8")
    ap.add_argument("--label", default="")
    ap.add_argument("--warm-passes", type=int, default=1,
                    help="2 = harness double-scheduled kernel_body; measure the warm 2nd pass")
    ap.add_argument("--layer", help="JSON manifest (list of arg-sets) for whole-layer aggregation")
    a = ap.parse_args()

    if a.layer:
        with open(a.layer) as f:
            report_layer(json.load(f))
        return
    if not (a.elf and a.trace and a.engine):
        ap.error("need <elf> <trace> --engine  (or --layer manifest.json)")
    d = measure(a.elf, a.trace, a.engine, a.macs, a.flops, a.bytes, a.fmt, a.tile_macs, a.label,
                a.warm_passes)
    ok = report_single(d)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
