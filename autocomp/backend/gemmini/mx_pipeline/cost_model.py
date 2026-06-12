"""Static instruction cost model for MX-Gemmini kernels — the shared currency.

Predicts spike cycles from a kernel's instruction mix WITHOUT running the sim, so
autocomp can (a) rank candidates cheaply and sim only the top-k, and (b) optimize
the same currency our profiling validates against (instead of its opaque score).

cycles ≈ Σ_op  coef[op] · count[op]

Counts are extracted statically from the kernel C, trip-weighted by the loop
structure (mvin/compute inside `for` loops over tile counts are multiplied by the
nest's symbolic trip product derived from the problem shape). Coefficients are
CALIBRATED by least-squares against measured `Generated implementation latency`
over the existing corpus of (code, latency) pairs — so each coefficient is an
interpretable per-instruction cost (mvin ∝ rows·⌈cols/dim⌉; fence = drain; etc.).

v0 limitation: static trip estimation is approximate for free-form LLM code; the
path to tighter calibration is dynamic event counts (spike GEMMINI trace / VCS).
Reports its own MAE/R² so no decision acts on a margin thinner than the error band.

  python -m autocomp.backend.gemmini.mx_pipeline.cost_model --fit --family matmul
"""

import argparse
import json
import re
import pathlib

import numpy as np

from autocomp.common import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output"

# Per-problem (rough) inner-loop trip multiplier = tiles_I·tiles_J·tiles_K and the
# mvin trip = (tiles_I+tiles_J)·tiles_K. Keyed by prob_type→prob_id.
SHAPES = {
    ("gemmini-mx-matmul", 0): (64, 64, 64),
    ("gemmini-mx-matmul", 1): (32, 32, 32),
    ("gemmini-mx-matmul", 2): (64, 128, 128),
    ("gemmini-mx-matmul", 3): (128, 128, 256),
}
DIM = 16

# Instruction op -> regex that matches a call site in the kernel C.
_OPS = {
    "mvin":      r"gemmini_(?:extended_)?mvin\b",
    "mvin2":     r"gemmini_(?:extended_)?mvin2\b",
    "mvin3":     r"gemmini_(?:extended_)?mvin3\b",
    "block_mvin": r"gemmini_block_mvin\b",
    "read_smem": r"gemmini_mx_read_smem\b",
    "load_scales": r"gemmini_mx_load_scales\b",
    "loop_ws":   r"gemmini_loop_ws_spad\b",
    "config_ld": r"gemmini_config_ld\b",
    "config_st": r"gemmini_config_st\b",
    "config_ex": r"gemmini_extended3_config_ex\b",
    "config_mvout": r"gemmini_mxquant_config_mvout\b",
    "fence":     r"gemmini_fence\b|(?<!_)\bfence\s*\(",
    "flush":     r"gemmini_flush\b",
    "preload":   r"gemmini_(?:extended_)?preload\b",
    "compute":   r"gemmini_(?:extended_)?compute_\w+\b",
    "memcpy":    r"\bmemcpy\b",
    "expf":      r"\bexp_?f?\b",
}


def extract_features(code: str, shape: tuple | None) -> dict:
    """Static, loop-trip-weighted instruction counts for a kernel body."""
    feats = {op: 0.0 for op in _OPS}
    if shape:
        M, K, N = shape
        tI, tJ, tK = M // DIM, N // DIM, K // DIM
    else:
        tI = tJ = tK = 1
    # crude loop-nest weighting: a call's weight = product of enclosing for-loop
    # trip counts (tiles_* and DIM-derived bounds resolved from the shape).
    bound_val = {"tiles_I": tI, "tiles_J": tJ, "tiles_K": tK, "tiles": max(tI, tJ),
                 "BLK": 4, "DIM": DIM}
    lines = code.splitlines()
    depth_trip = [1]
    for ln in lines:
        # entering a for loop: estimate its trip count
        fm = re.search(r"for\s*\([^;]*;\s*\w+\s*<\s*([A-Za-z_0-9]+)", ln)
        opens = ln.count("{")
        closes = ln.count("}")
        if fm:
            trip = bound_val.get(fm.group(1))
            if trip is None:
                try:
                    trip = int(fm.group(1))
                except ValueError:
                    trip = 4
            depth_trip.append(depth_trip[-1] * max(1, trip))
        weight = depth_trip[-1]
        for op, pat in _OPS.items():
            feats[op] += weight * len(re.findall(pat, ln))
        for _ in range(closes):
            if len(depth_trip) > 1:
                depth_trip.pop()
        for _ in range(max(0, opens - (1 if fm else 0))):
            depth_trip.append(depth_trip[-1])
    return feats


def _corpus(family: str):
    """Gather (features, latency) from existing runs for a family."""
    X, y, meta = [], [], []
    feat_order = list(_OPS)
    # plain matmul family = gemmini-mx-matmul_<id> (exclude -tiled)
    for run in sorted(OUTPUT_DIR.glob(f"mxpipe_gemmini-mx-{family}_[0-9]*")):
        mp = run / "run_metadata.json"
        if not mp.exists():
            continue
        m = re.search(r"Prob\((.+?),\s*(\d+)\)", json.loads(mp.read_text()).get("problem", ""))
        if not m:
            continue
        shape = SHAPES.get((m.group(1), int(m.group(2))))
        for cdir in sorted(run.glob("generated-code-iter-*")):
            it = cdir.name.rsplit("-", 1)[1]
            for impl in cdir.glob("impl_*_[0-9].txt"):
                if impl.name.endswith("_full.txt"):
                    continue
                cm = re.search(r"impl_\d+_(\d+)_", impl.name)
                if not cm:
                    continue
                rf = run / f"eval-results-iter-{it}" / f"code_{cm.group(1)}_result.txt"
                if not rf.exists():
                    continue
                try:
                    res = json.loads(rf.read_text())
                except Exception:
                    continue
                if not res.get("correct") or not res.get("latency"):
                    continue
                f = extract_features(impl.read_text(), shape)
                X.append([f[o] for o in feat_order])
                y.append(float(res["latency"]))
                meta.append((run.name, impl.name, shape))
    return np.array(X), np.array(y), feat_order, meta


def fit(family: str = "matmul"):
    X, y, order, meta = _corpus(family)
    if len(y) < 6:
        print(f"too few samples ({len(y)})"); return
    # non-negative-ish least squares with intercept
    A = np.hstack([X, np.ones((len(y), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = 1 - ss_res / ss_tot
    print(f"Cost model (family={family}): n={len(y)}  R²={r2:.3f}  MAE={mae:.1f} cyc "
          f"(mean latency {y.mean():.0f})")
    print("  learned per-instruction cost (cycles):")
    for o, c in sorted(zip(order, coef[:-1]), key=lambda kv: -abs(kv[1])):
        if abs(c) > 1e-6:
            print(f"    {o:<14} {c:>10.2f}")
    print(f"    {'intercept':<14} {coef[-1]:>10.2f}")
    np.savez(OUTPUT_DIR / f"cost_model_{family}.npz", coef=coef, order=order)
    return coef, order, (r2, mae)


import subprocess

GSW = REPO_ROOT  # placeholder; set in _gsw()


def _gsw():
    from autocomp.backend.gemmini.gemmini_eval import INT8_16PE_CHIPYARD_PATH
    return pathlib.Path(INT8_16PE_CHIPYARD_PATH) / "generators/gemmini/software/gemmini-rocc-tests"


# Dynamic feature order (real executed counts from the spike trace + shape-derived
# compute, which the fused MX loop doesn't emit).
DYN_ORDER = ["mvin_count", "mvin_work", "config_ex", "config_ld", "config_mvout",
             "flush", "compute_trips", "read_smem"]


def trace_features(test_code: str, shape: tuple, gsw: pathlib.Path) -> dict | None:
    """Build+run a kernel with spike --log-commits; count real GEMMINI events.

    mvin_work = Σ rows·⌈cols/DIM⌉ over mvin events (the DMA-cost proxy). Returns
    None on compile/run failure. compute_trips/read_smem are shape/static (the MX
    fused loop + smem read don't emit trace events).
    """
    (gsw / "bareMetalC" / "auto_comp_test.c").write_text(test_code)
    b = subprocess.run(["sh", "./build_spike.sh", "BAREMETAL_ONLY=1"], cwd=gsw,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if b.returncode != 0:
        return None
    try:
        p = subprocess.run(["spike", "--extension=gemmini", "--log-commits",
                            "./build_spike/bareMetalC/auto_comp_test-baremetal"],
                           cwd=gsw, capture_output=True, text=True, errors="ignore", timeout=600)
    except subprocess.TimeoutExpired:
        return None
    out = p.stdout
    if "Correct result" not in out and "PASSED" not in out:
        # still parse events if it ran; latency comes from the caller's eval
        pass
    f = {k: 0.0 for k in DYN_ORDER}
    for ln in out.split("\n"):
        m = re.search(r"GEMMINI:\s*(\w+)", ln)
        if not m:
            continue
        ev = m.group(1)
        if ev == "mvin":
            cm = re.search(r"0x([0-9a-f]+) cols and 0x([0-9a-f]+) rows", ln)
            f["mvin_count"] += 1
            if cm:
                cols, rows = int(cm.group(1), 16), int(cm.group(2), 16)
                f["mvin_work"] += rows * max(1, -(-cols // DIM))
        elif ev == "config_ex":
            f["config_ex"] += 1
        elif ev == "config_mvin":
            f["config_ld"] += 1
        elif ev == "config_mvout":
            f["config_mvout"] += 1
        elif ev == "flush":
            f["flush"] += 1
    if shape:
        M, K, N = shape
        f["compute_trips"] = (M // DIM) * (N // DIM) * (K // DIM)
    f["read_smem"] = float(len(re.findall(r"gemmini_mx_read_smem", test_code)))
    # config_ex/mvin are per-config-instruction; collapse duplicates spike logs (3 lines/config_ex)
    f["config_ex"] = round(f["config_ex"] / 3) if f["config_ex"] else 0
    return f


def fit_dynamic(family: str = "matmul", max_n: int = 60):
    """Recalibrate against REAL dynamic event counts from the spike trace."""
    from autocomp.search.prob import Prob
    gsw = _gsw()
    # collect (test_code, latency, shape) for correct kernels, deduped by code
    samples, seen = [], set()
    for run in sorted(OUTPUT_DIR.glob(f"mxpipe_gemmini-mx-{family}_[0-9]*")):
        mp = run / "run_metadata.json"
        if not mp.exists():
            continue
        m = re.search(r"Prob\((.+?),\s*(\d+)\)", json.loads(mp.read_text()).get("problem", ""))
        shape = SHAPES.get((m.group(1), int(m.group(2)))) if m else None
        prob = Prob(m.group(1), int(m.group(2)))
        from autocomp.backend.gemmini.gemmini_eval import clean_code
        for cdir in sorted(run.glob("generated-code-iter-*")):
            it = cdir.name.rsplit("-", 1)[1]
            for impl in cdir.glob("impl_*_[0-9].txt"):
                if impl.name.endswith("_full.txt"):
                    continue
                cm = re.search(r"impl_\d+_(\d+)_", impl.name)
                rf = run / f"eval-results-iter-{it}" / f"code_{cm.group(1)}_result.txt" if cm else None
                if not rf or not rf.exists():
                    continue
                try:
                    res = json.loads(rf.read_text())
                except Exception:
                    continue
                if not res.get("correct") or not res.get("latency"):
                    continue
                code = impl.read_text()
                h = hash(code)
                if h in seen:
                    continue
                seen.add(h)
                tc = prob.tests[0].get_test_code([clean_code(code)]) if "void solution" in code \
                    else prob.tests[0].get_test_code([code])
                samples.append((tc, float(res["latency"]), shape))
                if len(samples) >= max_n:
                    break
            if len(samples) >= max_n:
                break
        if len(samples) >= max_n:
            break

    print(f"tracing {len(samples)} unique kernels with --log-commits ...")
    X, y = [], []
    for i, (tc, lat, shape) in enumerate(samples):
        f = trace_features(tc, shape, gsw)
        if f is None:
            continue
        X.append([f[k] for k in DYN_ORDER]); y.append(lat)
    X, y = np.array(X), np.array(y)
    print(f"  traced {len(y)} ok")
    A = np.hstack([X, np.ones((len(y), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    mae = float(np.mean(np.abs(pred - y)))
    r2 = 1 - float(np.sum((pred - y) ** 2)) / (float(np.sum((y - y.mean()) ** 2)) or 1)
    print(f"DYNAMIC cost model (family={family}): n={len(y)}  R²={r2:.3f}  MAE={mae:.1f} cyc (mean {y.mean():.0f})")
    print("  learned per-event cost (cycles):")
    for o, c in sorted(zip(DYN_ORDER, coef[:-1]), key=lambda kv: -abs(kv[1])):
        print(f"    {o:<14} {c:>10.3f}")
    print(f"    {'intercept':<14} {coef[-1]:>10.3f}")
    np.savez(OUTPUT_DIR / f"cost_model_dyn_{family}.npz", coef=coef, order=DYN_ORDER)
    return coef, r2, mae


def predict(code: str, shape: tuple, family: str = "matmul") -> float:
    d = np.load(OUTPUT_DIR / f"cost_model_{family}.npz", allow_pickle=True)
    coef, order = d["coef"], list(d["order"])
    f = extract_features(code, shape)
    return float(np.dot([f[o] for o in order], coef[:-1]) + coef[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="calibrate from real spike trace event counts")
    ap.add_argument("--family", default="matmul")
    ap.add_argument("--max-n", type=int, default=60)
    a = ap.parse_args()
    if a.dynamic:
        fit_dynamic(a.family, a.max_n)
    elif a.fit:
        fit(a.family)
