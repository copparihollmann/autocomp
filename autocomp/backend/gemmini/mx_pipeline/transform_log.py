"""Transformation-provenance miner for autocomp MX-Gemmini runs.

Walks the per-iteration artifacts every search run already writes and emits a
clean ledger of (transformation -> outcome): which optimization strategy was
applied to which kernel, whether it compiled / stayed correct, and the latency
change vs its parent. Aggregated, this is a dataset of "which transformations
reliably help on which kernel shapes" — reusable later as compiler heuristics.

Per run dir (output/<run>/):
  - generated-plans-iter-N/plan_parent{P}_{model}_{C}.txt : the transformation
    (parsed for "## Selected Strategy: ... Strategy N — <name>")
  - eval-results-iter-N/code_{C}_result.txt : {compiled, correct, latency}
  - candidates-iter-{N-1}/candidate_{P}.txt : parent (eval'd for .score)
  - run_metadata.json : problem, models

Outcome classes: improved | correct_no_gain | regressed | incorrect | compile_error.

Usage:
  python -m autocomp.backend.gemmini.mx_pipeline.transform_log            # all runs
  python -m autocomp.backend.gemmini.mx_pipeline.transform_log --out x.jsonl
"""

import argparse
import json
import pathlib
import re

from autocomp.common import REPO_ROOT
from autocomp.search.code_repo import CodeCandidate  # noqa: F401 — needed by eval()

OUTPUT_DIR = REPO_ROOT / "output"
LEDGER = OUTPUT_DIR / "transform_ledger.jsonl"

_PLANFILE = re.compile(r"plan_parent(\d+)_(.+)_(\d+)\.txt$")
# Ordered patterns: try the strongest "this is the chosen strategy" markers first.
_STRAT_PATS = [
    r"Selected Strategy:?\s*\*{0,2}(?:Strategy\s*#?\d+\s*[—\-:]\s*)?(.+)",
    r"(?:New |Optimization )?Strategy:?\s*\*{0,2}(?:#?\d+\s*[—\-:]\s*)?(.+)",
    r"^#+\s*(?:Strategy\s*#?\d+\s*[—\-:]\s*)?(.+)",
    r"^#?\d+\s*[—\-]\s*(.+)",
]
_STRAT_NUM = re.compile(r"Strategy\s*#?(\d+)|^#(\d+)\b", re.M)


def _parse_strategy(plan_text: str) -> tuple[str, str]:
    num_m = _STRAT_NUM.search(plan_text)
    num = f"S{num_m.group(1) or num_m.group(2)}" if num_m else ""
    for pat in _STRAT_PATS:
        m = re.search(pat, plan_text, re.M)
        if m:
            name = re.sub(r"[`*#]", "", m.group(1)).strip().rstrip(":").strip()
            if name and not name.lower().startswith(("analysis", "the ", "looking", "current")):
                return (num, name[:70])
    return (num, num or "unknown")


def _cand_score(path: pathlib.Path):
    """Eval a saved candidate repr and return its own latency score."""
    try:
        c = eval(path.read_text())
        return c.score
    except Exception:
        # fallback: top-level score= (last one is the leaf candidate's own)
        t = path.read_text()
        ms = re.findall(r"score=([0-9.]+|None)", t)
        return float(ms[0]) if ms and ms[0] != "None" else None


def extract_run(run_dir: pathlib.Path) -> list[dict]:
    meta = {}
    mp = run_dir / "run_metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    prob = meta.get("problem", run_dir.name)
    m = re.search(r"Prob\((.+?),\s*(\d+)\)", prob)
    prob_type, prob_id = (m.group(1), int(m.group(2))) if m else (run_dir.name, -1)

    rows = []
    for plan_dir in sorted(run_dir.glob("generated-plans-iter-*")):
        it = int(plan_dir.name.rsplit("-", 1)[1])
        prev = run_dir / f"candidates-iter-{it-1}"
        for pf in sorted(plan_dir.glob("plan_parent*.txt")):
            fm = _PLANFILE.search(pf.name)
            if not fm:
                continue
            parent_idx, model, cand_idx = int(fm.group(1)), fm.group(2), int(fm.group(3))
            snum, sname = _parse_strategy(pf.read_text())

            rf = run_dir / f"eval-results-iter-{it}" / f"code_{cand_idx}_result.txt"
            if not rf.exists():
                continue
            try:
                res = json.loads(rf.read_text())
            except Exception:
                continue
            compiled = bool(res.get("compiled", False))
            correct = bool(res.get("correct", False))
            latency = res.get("latency")

            parent_lat = _cand_score(prev / f"candidate_{parent_idx}.txt") if prev.exists() else None
            speedup = (parent_lat / latency) if (correct and latency and parent_lat) else None
            if not compiled:
                cls = "compile_error"
            elif not correct:
                cls = "incorrect"
            elif speedup is None:
                cls = "correct_no_gain"
            elif speedup > 1.001:
                cls = "improved"
            elif speedup < 0.999:
                cls = "regressed"
            else:
                cls = "correct_no_gain"

            rows.append({
                "run": run_dir.name, "prob_type": prob_type, "prob_id": prob_id,
                "iteration": it, "parent_idx": parent_idx, "cand_idx": cand_idx,
                "model": model, "strategy_num": snum, "strategy": sname,
                "compiled": compiled, "correct": correct,
                "latency": latency, "parent_latency": parent_lat,
                "speedup": round(speedup, 4) if speedup else None,
                "outcome": cls, "plan_path": str(pf.relative_to(OUTPUT_DIR)),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Per-strategy success/speedup stats, overall and per kernel family."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "compiled": 0, "correct": 0, "improved": 0,
                               "speedups": [], "by_family": defaultdict(int)})
    for r in rows:
        a = agg[r["strategy"]]
        a["n"] += 1
        a["compiled"] += r["compiled"]
        a["correct"] += r["correct"]
        if r["outcome"] == "improved":
            a["improved"] += 1
            a["speedups"].append(r["speedup"])
        a["by_family"][r["prob_type"]] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LEDGER))
    ap.add_argument("--glob", default="*", help="run-dir glob under output/")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(OUTPUT_DIR.glob(args.glob)):
        if run_dir.is_dir() and (run_dir / "run_metadata.json").exists():
            rows.extend(extract_run(run_dir))

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    imp = sum(r["outcome"] == "improved" for r in rows)
    inc = sum(r["outcome"] == "incorrect" for r in rows)
    ce = sum(r["outcome"] == "compile_error" for r in rows)
    print(f"Transformation ledger: {n} attempts across {len(set(r['run'] for r in rows))} runs → {args.out}")
    print(f"  improved: {imp}  correct-no-gain: {n-imp-inc-ce}  incorrect: {inc}  compile_error: {ce}")
    # Heuristics view: per MX kernel family, the transformations that improved it.
    from collections import defaultdict
    fam_rows = defaultdict(list)
    for r in rows:
        if r["prob_type"].startswith("gemmini-mx"):
            fam_rows[r["prob_type"]].append(r)
    print("\n=== Heuristics by kernel family (transformations that IMPROVED, best speedup) ===")
    for fam in sorted(fam_rows):
        wins = [r for r in fam_rows[fam] if r["outcome"] == "improved"]
        wins.sort(key=lambda r: -(r["speedup"] or 0))
        print(f"\n  {fam}:  {len(wins)} improving / {len(fam_rows[fam])} attempts")
        seen = set()
        for r in wins:
            key = r["strategy"][:40]
            if key in seen:
                continue
            seen.add(key)
            print(f"    {r['speedup']:.2f}x  {r['strategy'][:60]}")
            if len(seen) >= 6:
                break


if __name__ == "__main__":
    main()
