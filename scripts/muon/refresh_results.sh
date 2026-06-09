#!/usr/bin/env bash
# Standing post-search step: refresh the transformation ledger (incl. negatives)
# and regenerate the auto heuristics summary. Run this after ANY autocomp search.
#   scripts/muon/refresh_results.sh
set -uo pipefail
AC=/scratch/agustin/projects/autocomp
cd "$AC"
source .venv/bin/activate 2>/dev/null || true

# 1) Rebuild the canonical ledger from every run dir (output/transform_ledger.jsonl).
python -m autocomp.backend.gemmini.mx_pipeline.transform_log >/dev/null 2>&1

# 2) Regenerate the auto heuristics summary (Muon rows) -> HEURISTICS.md.
python - <<'PY'
import json, collections, datetime, pathlib
led = pathlib.Path("output/transform_ledger.jsonl")
rows = [json.loads(l) for l in led.read_text().splitlines() if l.strip()]
muon = [r for r in rows if str(r.get("run", "")).startswith("built:muon")]
out = ["# Muon autocomp heuristics (auto-generated)",
       "",
       f"Source: `output/transform_ledger.jsonl` ({len(rows)} rows total, {len(muon)} muon).",
       "Regenerate with `scripts/muon/refresh_results.sh`. Curated insights live in MUON_RESULTS.md.",
       ""]
if muon:
    oc = collections.Counter(r["outcome"] for r in muon)
    out += ["## Outcome distribution (muon)",
            "```",
            "  ".join(f"{k}={v}" for k, v in oc.most_common()),
            "```", ""]
    out += ["## Transforms that improved (by speedup)", "```"]
    imp = sorted((r for r in muon if r["outcome"] == "improved" and r.get("speedup")),
                 key=lambda r: -r["speedup"])
    for r in imp[:20]:
        out.append(f"{r['speedup']:.2f}x  P{r['prob_id']}  {r['strategy'][:60]}")
    out += ["```", "", "## Negatives — what fails (compile_error / incorrect)", "```"]
    neg = collections.Counter(
        (r["outcome"], r["strategy"][:55]) for r in muon
        if r["outcome"] in ("compile_error", "incorrect"))
    for (oc_, strat), n in neg.most_common(20):
        out.append(f"{n:>3}x  {oc_:13} {strat}")
    out += ["```"]
pathlib.Path("HEURISTICS.md").write_text("\n".join(out) + "\n")
print("wrote HEURISTICS.md")
PY
echo "ledger: output/transform_ledger.jsonl | summary: HEURISTICS.md"
