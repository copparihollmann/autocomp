"""Distill autocomp MX search runs into a compiler-relevant knowledge base.

For every output/mxpipe_*/ run it links, per candidate, the optimization TRANSFORMATION
the LLM applied (from generated-plans-iter-N) to its measured CYCLE OUTCOME + correctness
(from eval-results-iter-N), and the best-score trajectory. Emits:
  - per-kernel summary: baseline -> best cycles, speedup, the winning transformation(s)
  - a transformation-effectiveness table: for each strategy bucket, how often it was
    tried / compiled-correct / appeared in an iteration that improved the best, per dtype.
This is the artifact that tells a compiler which transforms pay off for which shape/dtype.

Caveat: the strategy menu is randomly subsampled per iteration (dropout 0.25), so strategy
NUMBERS are not stable across iterations -> we bucket on strategy TEXT keywords, not numbers.

Usage: python -m autocomp.backend.gemmini.mx_pipeline.mine_runs [output_glob] [--md out.md]
"""

import argparse
import glob
import json
import os
import re
import pathlib

# keyword buckets -> canonical transformation name (compiler-facing vocabulary)
BUCKETS = [
    ("dual-port mvin (overlap A/B DMA)", r"mvin2|dual[- ]?port|two ports|port 1|interleav.*mvin|overlap.*dma"),
    ("remove premature fence",            r"fence|premature|stale.*state|clear.*state"),
    ("tile layout / avoid bank conflict", r"bank conflict|tile layout|scratchpad bank|avoid.*conflict"),
    ("B-tile reuse / I-band split",       r"i-band|band partition|reuse.*b|b.*reuse|split.*i-tile"),
    ("fix config_st stride",              r"config_st"),
    ("double buffering",                  r"double[- ]?buffer|ping[- ]?pong"),
    ("contiguous tile packing",           r"contiguous|pack.*tile|tile.*pack"),
    ("loop reorder / fuse",               r"reorder|fuse|merge.*loop|joint loop"),
    ("batch mvins (block_mvin)",          r"block_mvin|max_block_len|batch.*mvin|consecutive.*mvin"),
    ("raise lut_update_granularity",      r"lut_update_granularity|requant.*refresh|requantiz"),
    ("SPAD_DEST / output placement",      r"spad_dest|output placement|acc.*offset"),
    ("tune tile/param values",            r"new parameter|tune|parameter values|tile size"),
]


def bucket_of(text):
    t = text.lower()
    hits = [name for name, pat in BUCKETS if re.search(pat, t)]
    return hits or ["(other)"]


def _latency(path):
    try:
        d = json.loads(pathlib.Path(path).read_text())
        return (d.get("latency"), d.get("correct"))
    except Exception:
        # fall back to regex on the text result
        t = pathlib.Path(path).read_text()
        m = re.search(r'"latency":\s*(\d+)', t)
        c = '"correct": true' in t or '"correct":true' in t
        return (int(m.group(1)) if m else None, c)


def _strategy_texts(plan_dir):
    """Return list of strategy descriptions found in this iteration's plans."""
    out = []
    for p in sorted(glob.glob(os.path.join(plan_dir, "plan_*.txt"))):
        txt = pathlib.Path(p).read_text()
        # grab the 'Strategy ... — desc' / 'Selected Strategy: #N — desc' line
        m = re.search(r"(?:Selected Strategy|Strategy)\s*:?\s*#?\d*\s*[—:\-]+\s*(.+)", txt)
        if m:
            out.append(m.group(1).strip()[:90])
        else:
            out.append("(unparsed)")
    return out


def mine_run(run_dir):
    run = pathlib.Path(run_dir)
    name = run.name  # mxpipe_<ptype>_<id>_itersN
    m = re.match(r"mxpipe_(.+)_(\d+)_iters(\d+)", name)
    ptype = m.group(1) if m else name
    rm = {}
    rmf = run / "run_metrics.json"
    if rmf.exists():
        rm = json.loads(rmf.read_text())
    best = rm.get("best_score")
    # baseline = iter-0 candidate latency
    base = None
    z = run / "eval-results-iter-0" / "code_0_result.txt"
    if z.exists():
        base = _latency(z)[0]
    # per-iteration: candidate latencies + strategies tried
    iters = []
    running_best = base
    for it in range(0, 50):
        ed = run / f"eval-results-iter-{it}"
        if not ed.exists():
            continue
        cands = []
        for rf in sorted(glob.glob(str(ed / "code_*_result.txt"))):
            if rf.endswith("_full.txt"):
                continue
            lat, corr = _latency(rf)
            cands.append((lat, corr))
        strategies = _strategy_texts(str(run / f"generated-plans-iter-{it}"))
        valid = [l for (l, c) in cands if c and l]
        it_best = min(valid) if valid else None
        improved = it_best is not None and (running_best is None or it_best < running_best)
        if it_best is not None and (running_best is None or it_best < running_best):
            running_best = it_best
        iters.append(dict(iter=it, cands=cands, strategies=strategies,
                          it_best=it_best, improved=improved))
    return dict(ptype=ptype, baseline=base, best=best, iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["output/mxpipe_*"])
    ap.add_argument("--md", default="results/mx-kernels/STRATEGY_LEDGER.md")
    args = ap.parse_args()
    globs = args.globs or ["output/mxpipe_*"]
    dirs = sorted({d for g in globs for d in glob.glob(g) if os.path.isdir(d)})
    runs = [mine_run(d) for d in dirs]

    # transformation-effectiveness aggregation
    agg = {}  # bucket -> dict(tried, in_improving)
    lines = ["# MX autocomp — transformation → effect ledger", ""]
    lines.append("| kernel | baseline | best | speedup | winning transforms |")
    lines.append("|---|---|---|---|---|")
    for r in runs:
        spd = (r["baseline"] / r["best"]) if (r["baseline"] and r["best"]) else None
        winners = []
        for it in r["iters"]:
            for s in it["strategies"]:
                for b in bucket_of(s):
                    a = agg.setdefault(b, dict(tried=0, improving=0))
                    a["tried"] += 1
                    if it["improved"]:
                        a["improving"] += 1
                if it["improved"]:
                    winners.append(s)
        lines.append(f"| {r['ptype']} | {r['baseline']} | {r['best']} | "
                     f"{spd:.2f}× | {'; '.join(dict.fromkeys(winners)) or '—'} |")
    lines += ["", "## Transformation effectiveness (across all runs)", "",
              "| transformation | times tried | appeared in an improving iter |",
              "|---|---|---|"]
    for b, a in sorted(agg.items(), key=lambda kv: -kv[1]["improving"]):
        lines.append(f"| {b} | {a['tried']} | {a['improving']} |")
    out = pathlib.Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
