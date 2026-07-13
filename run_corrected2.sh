#!/usr/bin/env bash
cd "$(dirname "$0")"
while pgrep -f run_all_muon >/dev/null; do sleep 60; done
source ./muon.env
export MUON_TAG=_v2
for p in 0 2; do
  echo "=== corrected v2 search problem $p @ $(date) ==="
  .venv/bin/python -m autocomp.search.run_search_muon $p
done
