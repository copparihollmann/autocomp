#!/usr/bin/env bash
# Unattended: poll the Bedrock daily-token quota; the moment it clears, launch the
# matmul (prob6, smolvla K=128) and flash-attention (prob7, seq=128) searches.
# Budget is enforced by autocomp.common.cost (AUTOCOMP_SPEND_LIMIT_USD=100, mode=stop).
set +e
cd /scratch/agustin/projects/autocomp || exit 9
source muon.env 2>/dev/null
LOG=/tmp/auto_relaunch.log
echo "=== auto_relaunch started $(date) ===" > "$LOG"

probe() {  # returns 0 if quota OK
  .venv/bin/python - <<'PY' 2>/dev/null
import os, boto3
try:
    brt = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION","us-east-1"))
    r = brt.converse(modelId="us.anthropic.claude-sonnet-4-6",
        messages=[{"role":"user","content":[{"text":"ok"}]}],
        inferenceConfig={"maxTokens":5})
    print("OK"); raise SystemExit(0)
except SystemExit: raise
except Exception as e:
    print("THROTTLED" if ("Throttling" in str(e) or "Too many tokens" in str(e)) else "ERR:"+str(e)[:80])
    raise SystemExit(1)
PY
}

# poll up to 18h, every 20 min
for i in $(seq 1 54); do
  out=$(probe); rc=$?
  echo "$(date) probe#$i -> $out (rc=$rc)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "$(date) QUOTA CLEARED — launching searches" >> "$LOG"
    export MUON_ITERS=6 MUON_PLANS=2 MUON_CODES=2 MUON_BEAM=2
    MUON_TAG=_smolvla_k128 nohup .venv/bin/python -m autocomp.search.run_search_muon 6 > /tmp/search_matmul_p6.out 2>&1 &
    echo "  matmul search PID $!" >> "$LOG"
    sleep 30  # stagger so they don't collide on first-token burst
    MUON_TAG=_flash_seq128 nohup .venv/bin/python -m autocomp.search.run_search_muon 7 > /tmp/search_attn_p7.out 2>&1 &
    echo "  attention search PID $!" >> "$LOG"
    echo "$(date) both launched; auto_relaunch exiting" >> "$LOG"
    touch /tmp/auto_relaunch.launched
    exit 0
  fi
  sleep 1200
done
echo "$(date) gave up after 18h — quota never cleared" >> "$LOG"
touch /tmp/auto_relaunch.gaveup
exit 0
