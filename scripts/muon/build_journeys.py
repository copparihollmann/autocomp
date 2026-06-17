#!/usr/bin/env python3
"""Assemble per-problem optimization-journey records from autocomp Muon search output.

Reads existing search artifacts (no new runs) and emits, into a staging directory:
  journeys/prob<N>/journey.jsonl   one row per attempted candidate (start -> ... -> best)
  journeys/prob<N>/cost.json       per-iteration and run-total tokens + USD
  journeys/prob<N>/JOURNEY.md      human-readable timeline
  index/all_transforms.jsonl       every journey row across all problems
  index/summary.csv                per-problem baseline/best/speedup/counts/cost
  index/cost_by_problem.csv        per-problem and per-iteration cost

Journey rows come from transform_ledger.jsonl when the run is covered there (probs 1,2,3,7);
otherwise they are reconstructed from the run's eval-results files (probs 0,6,10) and marked
source="reconstructed".
"""
import json, os, re, sys, glob, csv

AC = "/scratch/agustin/projects/autocomp"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/journeys_build"
LEDGER = f"{AC}/muon_search_data/ledgers/transform_ledger.jsonl"
RESULTS_CSV = "/scratch/agustin/projects/chipyard/generators/radiance/autocomp/metrics/results.csv"

def load_ledger():
    by_run = {}
    for line in open(LEDGER):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except: continue
        if d.get("prob_type") != "muon": continue
        by_run.setdefault(d["run"], []).append(d)
    return by_run

def sum_costs(obj, acc):
    """Recursively sum cost_usd / input_tokens / output_tokens from a metrics dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cost_usd" and isinstance(v, (int, float)): acc["usd"] += v
            elif k == "input_tokens" and isinstance(v, (int, float)): acc["in_tok"] += v
            elif k == "output_tokens" and isinstance(v, (int, float)): acc["out_tok"] += v
            else: sum_costs(v, acc)
    elif isinstance(obj, list):
        for v in obj: sum_costs(v, acc)

def iter_cost(run_dir):
    per_iter = {}
    for f in sorted(glob.glob(f"{run_dir}/metrics-iter-*.json")):
        m = re.search(r"metrics-iter-(\d+)\.json", f)
        if not m: continue
        it = int(m.group(1))
        acc = {"usd": 0.0, "in_tok": 0, "out_tok": 0}
        try: sum_costs(json.load(open(f)), acc)
        except: pass
        per_iter[it] = acc
    total = {"usd": 0.0, "in_tok": 0, "out_tok": 0, "calls": 0}
    cl = f"{run_dir}/cost_live.json"
    if os.path.exists(cl):
        try:
            d = json.load(open(cl))
            total = {"usd": d.get("total_usd", 0.0), "in_tok": d.get("input_tokens", 0),
                     "out_tok": d.get("output_tokens", 0), "calls": d.get("calls", 0)}
        except: pass
    return per_iter, total

def eval_rows(run_dir):
    """Reconstruct attempts from eval-results: {iter, cand_idx, compiled, correct, latency}."""
    rows = []
    for d in sorted(glob.glob(f"{run_dir}/eval-results-iter-*")):
        m = re.search(r"eval-results-iter-(\d+)", d)
        if not m: continue
        it = int(m.group(1))
        for rf in sorted(glob.glob(f"{d}/code_*_result.txt")):
            if rf.endswith("_full.txt"): continue
            ci = re.search(r"code_(\d+)_result\.txt", rf)
            ci = int(ci.group(1)) if ci else -1
            try: r = json.load(open(rf))
            except: continue
            rows.append({"iteration": it, "cand_idx": ci,
                         "compiled": r.get("compiled"), "correct": r.get("correct"),
                         "latency": r.get("latency"), "source": "reconstructed"})
    return rows

def baseline_score(run_dir):
    for f in sorted(glob.glob(f"{run_dir}/candidates-iter-0/candidate_*.txt")):
        m = re.search(r"score=(\d+)", open(f).read())
        if m: return int(m.group(1))
    return None

def prob_id_of(dirname):
    m = re.search(r"built:muon_muon_(\d+)_", dirname)
    return int(m.group(1)) if m else None

def main():
    os.makedirs(f"{OUT}/index", exist_ok=True)
    by_run = load_ledger()
    run_dirs = sorted(glob.glob(f"{AC}/output/built:muon_muon_*"))
    # results.csv anchors (baseline/best/provenance)
    anchors = {}
    if os.path.exists(RESULTS_CSV):
        for r in csv.DictReader(open(RESULTS_CSV)):
            try: anchors[int(r["prob"])] = r
            except: pass

    all_rows, summary, cost_rows = [], [], []
    for rd in run_dirs:
        name = os.path.basename(rd)
        pid = prob_id_of(name)
        if pid is None: continue
        runtag = re.sub(r"^built:muon_muon_\d+_", "", name)
        runtag = re.sub(r"_cyclotron$", "", runtag)
        pdir = f"{OUT}/journeys/prob{pid}/{runtag}"
        os.makedirs(pdir, exist_ok=True)
        per_iter, total = iter_cost(rd)
        base = baseline_score(rd)

        if name in by_run:
            rows = sorted(by_run[name], key=lambda x: (x.get("iteration", 0), x.get("cand_idx", 0)))
            for r in rows: r["source"] = "transform_ledger"
            src = "transform_ledger"
        else:
            rows = eval_rows(rd)
            src = "reconstructed"
        for r in rows:
            r["prob_id"], r["run"] = pid, name
        # best correct latency
        lats = [r["latency"] for r in rows if r.get("correct") and isinstance(r.get("latency"), (int, float))]
        best = min(lats) if lats else None
        n_comp = sum(1 for r in rows if r.get("compiled"))
        n_corr = sum(1 for r in rows if r.get("correct"))
        n_impr = sum(1 for r in rows if r.get("outcome") == "improved")

        json.dump({"prob_id": pid, "run": name, "iterations": len(per_iter),
                   "per_iteration": {str(k): v for k, v in sorted(per_iter.items())},
                   "run_total": total}, open(f"{pdir}/cost.json", "w"), indent=2)
        with open(f"{pdir}/journey.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        all_rows.extend(rows)

        a = anchors.get(pid, {})
        # JOURNEY.md
        with open(f"{pdir}/JOURNEY.md", "w") as f:
            f.write(f"# Problem {pid} optimization journey ({src})\n\n")
            f.write(f"- run: `{name}`\n- iterations: {len(per_iter)}\n")
            f.write(f"- baseline cycles: {a.get('baseline_total','?')} | best cycles: {a.get('best_total','?')} "
                    f"| speedup_total: {a.get('speedup_total','?')} | speedup_net: {a.get('speedup_net','NA')}\n")
            f.write(f"- candidates: {len(rows)} | compiled: {n_comp} | correct: {n_corr} | improved: {n_impr}\n")
            f.write(f"- cost: ${total['usd']:.2f}, {total['in_tok']+total['out_tok']} tokens, {total['calls']} calls\n\n")
            f.write("## Attempts (iteration -> candidate)\n\n")
            if src == "transform_ledger":
                f.write("| iter | parent | cand | model | strategy | compiled | correct | latency | speedup | outcome |\n")
                f.write("|---|---|---|---|---|---|---|---|---|---|\n")
                for r in rows:
                    strat = (r.get("strategy") or "")[:60].replace("|", "/").replace("\n", " ")
                    f.write(f"| {r.get('iteration')} | {r.get('parent_idx')} | {r.get('cand_idx')} | "
                            f"{(r.get('model') or '').split('::')[-1][:20]} | {strat} | {r.get('compiled')} | "
                            f"{r.get('correct')} | {r.get('latency')} | {r.get('speedup')} | {r.get('outcome')} |\n")
            else:
                f.write("(reconstructed from eval results; strategies are in `plans/`)\n\n")
                f.write("| iter | cand | compiled | correct | latency |\n|---|---|---|---|---|\n")
                for r in rows:
                    f.write(f"| {r.get('iteration')} | {r.get('cand_idx')} | {r.get('compiled')} | "
                            f"{r.get('correct')} | {r.get('latency')} |\n")

        summary.append({"prob": pid, "run": name, "source": src,
                        "baseline_total": a.get("baseline_total", ""), "best_total": a.get("best_total", ""),
                        "speedup_total": a.get("speedup_total", ""), "speedup_net": a.get("speedup_net", ""),
                        "provenance": a.get("provenance", ""), "iterations": len(per_iter),
                        "candidates": len(rows), "compiled": n_comp, "correct": n_corr, "improved": n_impr,
                        "best_correct_latency": best, "total_usd": round(total["usd"], 3),
                        "total_tokens": total["in_tok"] + total["out_tok"]})
        for it, c in sorted(per_iter.items()):
            cost_rows.append({"prob": pid, "run": name, "iteration": it,
                              "usd": round(c["usd"], 4), "input_tokens": c["in_tok"], "output_tokens": c["out_tok"]})

    with open(f"{OUT}/index/all_transforms.jsonl", "w") as f:
        for r in all_rows: f.write(json.dumps(r) + "\n")
    def write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    write_csv(f"{OUT}/index/summary.csv", sorted(summary, key=lambda x: x["prob"]),
              ["prob", "run", "source", "provenance", "baseline_total", "best_total", "speedup_total",
               "speedup_net", "iterations", "candidates", "compiled", "correct", "improved",
               "best_correct_latency", "total_usd", "total_tokens"])
    write_csv(f"{OUT}/index/cost_by_problem.csv", cost_rows,
              ["prob", "run", "iteration", "usd", "input_tokens", "output_tokens"])
    print(f"problems with journeys: {sorted(set(s['prob'] for s in summary))}")
    print(f"total journey rows: {len(all_rows)}  | summary rows: {len(summary)}")
    print(f"output staged in: {OUT}")

if __name__ == "__main__":
    main()
