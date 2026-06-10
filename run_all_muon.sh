#!/usr/bin/env bash
cd "$(dirname "$0")"
source ./muon.env
for p in 0 1 2 3 5 4; do
  echo "=== search problem $p @ $(date) ==="
  .venv/bin/python -m autocomp.search.run_search_muon $p
done

# Always refresh the transform ledger + heuristics after searches (see MUON_RESULTS.md).
echo "=== refreshing results/heuristics @ $(date) ==="
bash scripts/muon/refresh_results.sh
