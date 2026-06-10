#!/usr/bin/env bash
# Live view of autocomp spend for the muon project.
#   ./watch_costs.sh          - one snapshot
#   watch -n10 ./watch_costs.sh   - live
cd "$(dirname "$0")"
[ -z "$AUTOCOMP_SPEND_LOG" ] && source ./muon.env >/dev/null 2>&1

echo "=== muon project lifetime spend (cap: \$${AUTOCOMP_SPEND_LIMIT_USD:-unset}) ==="
python -m autocomp.common.cost --total 2>/dev/null || echo "(no spend recorded yet)"

latest=$(ls -dt output/*/ 2>/dev/null | head -1)
if [ -n "$latest" ] && [ -f "$latest/cost_live.json" ]; then
  echo
  echo "=== current run ($latest) ==="
  python -c "
import json
d = json.load(open('$latest/cost_live.json'))
print(f\"  calls: {d.get('calls', d.get('call_count', '?'))}\")
print(f\"  cost:  \${d.get('total_cost_usd', d.get('cost_usd', 0)):.4f}\")
for m, v in (d.get('models') or {}).items():
    print(f\"  {m}: \${v.get('cost_usd', 0):.4f}\")
"
fi
