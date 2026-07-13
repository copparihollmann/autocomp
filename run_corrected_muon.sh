#!/usr/bin/env bash
# Wait for the running search batch; then re-search with kernel-only latency metric.
cd "$(dirname "$0")"
while pgrep -f run_all_muon >/dev/null; do sleep 60; done
source ./muon.env
for p in 0 2 4; do
  echo "=== corrected search problem $p @ $(date) ==="
  .venv/bin/python -m autocomp.search.run_search_muon $p
done
