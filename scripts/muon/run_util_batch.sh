#!/usr/bin/env bash
# Part C batch: build + Verilator-trace every kernel in util_sweep_spec.json (single-pass, WARM=1),
# with a concurrency cap (Verilator sims are heavy). Each kernel uses its own autocomp_<tag> build
# dir so they are independent. Re-runnable: skips a kernel whose trace already exists.
#   run_util_batch.sh [max_parallel]   (default 4)
set -uo pipefail
HERE=/scratch/agustin/projects/autocomp/scripts/muon
RK=/scratch/agustin/projects/radiance-kernels/kernels
MAXP=${1:-4}
mapfile -t SPECS < <(python3 -c "
import json
for k in json.load(open('$HERE/util_sweep_spec.json'))['kernels']:
    print(f\"{k['n']} {k['tag']}\")
")
launched=0
for line in "${SPECS[@]}"; do
  N=${line%% *}; TAG=${line##* }
  TR="$RK/autocomp_$TAG/trace_$TAG.sqlite"
  if [ -s "$TR" ]; then echo "[$TAG] trace exists, skip"; continue; fi
  # throttle
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
  echo "[$TAG] launching test$N ..."
  ( WARM=1 DRAIN=2000 bash "$HERE/sweep_util_trace.sh" "$N" "$TAG" \
      > "$RK/../$TAG.sweep.log" 2>&1; echo "[$TAG] done exit=$?" ) &
  launched=$((launched+1))
done
wait
echo "ALL DONE ($launched launched)"
