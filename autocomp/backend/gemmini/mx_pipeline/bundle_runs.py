"""Export the full autocomp MX-Gemmini optimization corpus as a portable bundle.

Composes the existing per-attempt miner (transform_log) with the beam-tree state
(candidates-iter-*) and the cost/metrics records every run already writes, to emit
a machine-readable dataset that captures, per kernel, the start (baseline) -> full
optimization journey (every beam node + transformation) -> end (best), plus the
token/$ cost of getting there. Outputs:

  dataset/journeys/<run>.json  : per-run beam tree (baseline, kept beam per iter,
                                 every attempt, best + iteration reached, cost rollup)
  dataset/journeys.jsonl       : flat one-row-per-attempt master (transform_ledger
                                 + family/shape/baseline/speedup_vs_baseline/cost)
  dataset/index.csv, index.md  : per-kernel baseline/best/speedup/cost/location

Scope: MX-Gemmini runs only (prob_type in gemmini-mx*/real-*/fork-mm*/smolvla*);
muon and _arch runs are excluded.

Usage:
  python -m autocomp.backend.gemmini.mx_pipeline.bundle_runs --out <dataset_dir>
"""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

from autocomp.common import REPO_ROOT
from autocomp.backend.gemmini.mx_pipeline.transform_log import (
    OUTPUT_DIR, extract_run, _cand_score,
)

MX_PREFIXES = ("gemmini-mx", "real-", "fork-mm", "smolvla")
# Runs evaluated under the faithful-cycle spike model (GEMMINI_FAITHFUL_CYCLES=1),
# not recorded in run_metadata; tag by run name so the metric is not misread as base spike.
FAITHFUL_RUNS = {
    "mxpipe_gemmini-mx-matmul-fp4_0_iters7",
    "mxpipe_gemmini-mx-matmul-fp6_0_iters7",
    "mxpipe_real-pi05-ffn_0_iters7",
}
_DIM_RE = re.compile(r"\b(?:MATMUL_)?M\s*=?\s*(\d+).*?\b(?:MATMUL_)?K\s*=?\s*(\d+).*?\b(?:MATMUL_)?N\s*=?\s*(\d+)", re.S)

# Curated shapes (M x K x N for matmul, matching RESULTS.md and the gen_matmul_problem
# signatures in search/run_mx_pipeline.py; S/D/H for attention). Best-effort; runs not
# listed fall back to "".
SHAPE_MAP = {
    "gemmini-mx-matmul-fp4": "128x128x256", "gemmini-mx-matmul-fp6": "128x128x256",
    "gemmini-mx-matmul-tiled": "128x128x128", "gemmini-mx-tiled-fp4": "128x128x128",
    "gemmini-mx-tiled-fp6": "128x128x128", "gemmini-mx-conv": "2x32x4x32 (NCHW patch)",
    "gemmini-mx-conv-fp4": "2x32x4x32", "gemmini-mx-conv-fp6": "2x32x4x32",
    "gemmini-mx-attention": "S128xD64", "gemmini-mx-attn-fp4": "S128xD64",
    "gemmini-mx-attn-fp6": "S128xD64", "gemmini-mx-flash-attn": "S128xD64",
    "gemmini-mx-flash-fp4": "S128xD64", "gemmini-mx-flash-fp6": "S128xD64",
    "fork-mm-fp8-128": "128x128x128", "fork-mm-fp8-256": "256x256x256",
    "smolvla-proj": "1024x768x768", "smolvla-mlp": "1024x768x3072", "smolvla-act": "128x320x320",
    "real-groot-patch": "41x512x512", "real-smolvla-vis": "113x256x256",
    "real-smolvla-ffn": "50x256x512", "real-llm-attn": "8x512x512",
    "real-pi05-ffn": "256x256x512", "real-bitvla": "32x256x256",
    "real-gemv-1024": "1x1024x1024", "real-gemv-3072": "1x1024x3072",
    "real-attn-causal": "S128xD64 causal", "real-mha": "S64xD64xH4",
    "real-gqa": "S64xD64xH4kv2", "real-mha-causal": "S64xD64xH4 causal",
}


def _is_mx(prob_type: str) -> bool:
    return prob_type.startswith(MX_PREFIXES)


def _family(prob_type: str) -> str:
    if prob_type.startswith("real-"):
        return "real-model"
    if prob_type.startswith("fork-mm"):
        return "golden-fork"
    if prob_type.startswith("smolvla"):
        return "real-model"
    return "autocomp-generated"


def _shape_from_code(code: str) -> str:
    if not code:
        return ""
    m = _DIM_RE.search(code)
    if m:
        return f"{m.group(1)}x{m.group(3)}x{m.group(2)}"  # MxNxK
    return ""


def _load_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _baseline(run_dir: pathlib.Path):
    """sol0 baseline = iter-0 candidate_0 (parent=None, plan=None). Return (score, code)."""
    f = run_dir / "candidates-iter-0" / "candidate_0.txt"
    if not f.exists():
        return None, ""
    try:
        c = eval(f.read_text())
        return c.score, (c.code or "")
    except Exception:
        return _cand_score(f), ""


def _kept_beam(run_dir: pathlib.Path):
    """Per iteration, the SELECTED candidates (the surviving beam) with their own score
    and immediate-parent score. candidates-iter-N/ already holds only the kept beam."""
    beams = {}
    for cdir in sorted(run_dir.glob("candidates-iter-*")):
        it = int(cdir.name.rsplit("-", 1)[1])
        nodes = []
        for cf in sorted(cdir.glob("candidate_*.txt")):
            idx = int(cf.stem.split("_")[1])
            score = parent_score = None
            plan = None
            try:
                c = eval(cf.read_text())
                score = c.score
                plan = (c.plan or None)
                parent_score = c.parent.score if getattr(c, "parent", None) else None
            except Exception:
                score = _cand_score(cf)
            nodes.append({
                "cand_idx": idx,
                "score": score,
                "parent_score": parent_score,
                "is_seed": (it == 0 and plan is not None),  # iter-0 non-sol0 = prior-winner seed
                "strategy_snippet": (plan.strip().splitlines()[0][:120] if plan else None),
            })
        beams[it] = nodes
    return beams


def build_run(run_dir: pathlib.Path) -> dict:
    meta = _load_json(run_dir / "run_metadata.json")
    metrics = _load_json(run_dir / "run_metrics.json")
    prob = meta.get("problem", run_dir.name)
    m = re.search(r"Prob\((.+?),\s*(\d+)\)", prob)
    prob_type, prob_id = (m.group(1), int(m.group(2))) if m else (run_dir.name, -1)

    attempts = extract_run(run_dir)            # flat per-attempt rows (reused miner)
    beams = _kept_beam(run_dir)
    base_score, base_code = _baseline(run_dir)
    shape = SHAPE_MAP.get(prob_type) or _shape_from_code(base_code)

    # best = min correct latency across attempts, fall back to run_metrics best_score
    correct_lat = [a["latency"] for a in attempts if a["correct"] and a["latency"]]
    best_score = metrics.get("best_score")
    if correct_lat:
        best_score = min([best_score] + correct_lat) if best_score else min(correct_lat)
    best_iter = None
    for a in attempts:
        if a["latency"] == best_score and a["correct"]:
            best_iter = a["iteration"]
            break

    speedup = (base_score / best_score) if (base_score and best_score) else None
    cost = {
        "total_input_tokens": metrics.get("total_input_tokens"),
        "total_output_tokens": metrics.get("total_output_tokens"),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "by_model_usd": metrics.get("estimated_cost_by_model_usd"),
        "run_total_s": metrics.get("run_total_s"),
        "per_iteration": [
            {"iteration": it.get("iteration"),
             "num_candidates": it.get("evaluation", {}).get("num_candidates")}
            for it in metrics.get("iterations", [])
        ],
    }
    return {
        "run": run_dir.name,
        "prob_type": prob_type,
        "prob_id": prob_id,
        "family": _family(prob_type),
        "shape_MxKxN": shape,
        "simulator": meta.get("simulator", "spike"),
        "metric": "faithful-cycles" if run_dir.name in FAITHFUL_RUNS else meta.get("metric", "latency"),
        "plan_models": meta.get("plan_models"),
        "code_models": meta.get("code_models"),
        "beam_size": meta.get("beam_size"),
        "num_iterations": len(metrics.get("iterations", [])),
        "baseline": {"score": base_score, "code_excerpt": base_code[:400]},
        "best": {"score": best_score, "iteration_reached": best_iter},
        "speedup_vs_baseline": round(speedup, 4) if speedup else None,
        "num_attempts": len(attempts),
        "kept_beam_per_iter": beams,
        "attempts": attempts,
        "cost": cost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_DIR / "dataset"),
                    help="dataset output dir (journeys/, journeys.jsonl, index.csv/md)")
    ap.add_argument("--glob", default="mxpipe_*")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "journeys").mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir in sorted(OUTPUT_DIR.glob(args.glob)):
        if not (run_dir.is_dir() and (run_dir / "run_metadata.json").exists()):
            continue
        meta = _load_json(run_dir / "run_metadata.json")
        m = re.search(r"Prob\((.+?),\s*(\d+)\)", meta.get("problem", run_dir.name))
        prob_type = m.group(1) if m else run_dir.name
        if not _is_mx(prob_type) or "_arch" in run_dir.name:
            continue
        runs.append(build_run(run_dir))

    # per-run journey JSON
    for r in runs:
        (out / "journeys" / f"{r['run']}.json").write_text(json.dumps(r, indent=2))

    # flat journeys.jsonl: one row per attempt with run/journey context
    with open(out / "journeys.jsonl", "w") as f:
        for r in runs:
            ctx = {"run": r["run"], "family": r["family"], "shape_MxKxN": r["shape_MxKxN"],
                   "baseline_score": r["baseline"]["score"], "metric": r["metric"]}
            for a in r["attempts"]:
                row = dict(a)
                row.update(ctx)
                bl = r["baseline"]["score"]
                row["speedup_vs_baseline"] = (round(bl / a["latency"], 4)
                                              if a["correct"] and a["latency"] and bl else None)
                f.write(json.dumps(row) + "\n")

    # index.csv + index.md
    cols = ["family", "prob_type", "prob_id", "shape_MxKxN", "metric", "baseline_cyc",
            "best_cyc", "speedup", "num_iters", "num_attempts", "input_tokens",
            "output_tokens", "cost_usd", "run_dir"]
    rows = []
    for r in sorted(runs, key=lambda x: (x["family"], x["prob_type"], x["run"])):
        rows.append([
            r["family"], r["prob_type"], r["prob_id"], r["shape_MxKxN"], r["metric"],
            r["baseline"]["score"], r["best"]["score"], r["speedup_vs_baseline"],
            r["num_iterations"], r["num_attempts"],
            r["cost"]["total_input_tokens"], r["cost"]["total_output_tokens"],
            (round(r["cost"]["estimated_cost_usd"], 4) if r["cost"]["estimated_cost_usd"] else None),
            r["run"],
        ])
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    tot_cost = sum(r["cost"]["estimated_cost_usd"] or 0 for r in runs)
    tot_in = sum(r["cost"]["total_input_tokens"] or 0 for r in runs)
    tot_out = sum(r["cost"]["total_output_tokens"] or 0 for r in runs)
    tot_att = sum(r["num_attempts"] for r in runs)
    with open(out / "index.md", "w") as f:
        f.write("# MX-Gemmini autocomp run index\n\n")
        f.write(f"{len(runs)} MX runs | {tot_att} total attempts | "
                f"{tot_in:,} input + {tot_out:,} output tokens | "
                f"${tot_cost:.2f} run-attributed cost.\n\n")
        f.write("Cycle counts are under each run's metric (spike / faithful-cycles). Spike speedups "
                "are inflated ~1.5-2x vs RTL; see results/MX_AUTOCOMP_RESULTS.md for the silicon-true "
                "RTL numbers.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for row in rows:
            f.write("| " + " | ".join("" if v is None else str(v) for v in row) + " |\n")

    print(f"Bundled {len(runs)} MX runs -> {out}")
    print(f"  journeys/: {len(runs)} per-run trees | journeys.jsonl: {tot_att} attempt rows")
    print(f"  cost: ${tot_cost:.2f} | tokens: {tot_in:,} in / {tot_out:,} out")


if __name__ == "__main__":
    main()
