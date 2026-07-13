"""Verify autocomp cost tracking against ground truth, cheaply.

Makes a few tiny real Bedrock calls (fractions of a cent) and asserts:
  1. autocomp records EXACTLY the token counts Bedrock returns (billable basis),
     cross-checked against a raw boto3 Converse call with identical input.
  2. USD math = (in*price_in + out*price_out)/1e6 * region_multiplier, per call.
  3. Multi-call + multi-phase aggregation is correct.
  4. The live files (cost_live.json, cost_ledger.jsonl) and the running ledger agree.
  5. The long-context (>200K) tier and region multiplier behave (unit-level, no spend).

Run inside the venv with creds sourced:
  source env.sh; set -a; source /scratch2/agustin/ModelBlaster/.env; set +a
  export AWS_REGION=us-west-2; cd autocomp; source .venv/bin/activate
  python verify_cost_tracking.py
"""

import json
import os
import pathlib
import tempfile

import boto3

from autocomp.common import cost as C
from autocomp.common import llm_utils as L

MODEL = "us.anthropic.claude-sonnet-4-6"
REGION = os.environ.get("AWS_REGION", "us-west-2")
MULT = C._DEFAULT_REGION_MULTIPLIER
PRICE = C.price_for(MODEL)  # {"in":3,"out":15,...} per 1M

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print(f"Model={MODEL}  region={REGION}  region_multiplier={MULT}  price/1M in/out={PRICE['in']}/{PRICE['out']}")

# ---- Part A: unit-level (no spend) -----------------------------------------
print("\n[A] Pure-math checks (no API spend):")
# region multiplier baked in
base = (1000 * PRICE["in"] + 1000 * PRICE["out"]) / 1e6
check("region multiplier applied", abs(C.estimate_cost(MODEL, 1000, 1000, region_multiplier=1.0) - base) < 1e-9,
      f"@1.0x = ${base:.6f}")
check("1.10x = 1.0x * 1.10", abs(C.estimate_cost(MODEL, 1000, 1000) - base * MULT) < 1e-9)
# long-context tier triggers above 200k input
lo = C.estimate_cost(MODEL, 200_000, 100, region_multiplier=1.0)
hi = C.estimate_cost(MODEL, 200_001, 100, region_multiplier=1.0)
check("long-context tier jumps >200k input", hi > lo * 1.9,
      f"200k=${lo:.4f} -> 200k+1=${hi:.4f} (input rate ~2x)")
# unknown model -> 0 and flagged
check("unknown model -> $0", C.estimate_cost("meta.llama3-70b", 1000, 1000) == 0.0)

# ---- Part B: live calls + faithfulness -------------------------------------
print("\n[B] Live Bedrock calls (tiny, real spend ~<1 cent):")
d = pathlib.Path(tempfile.mkdtemp(prefix="costverify-"))
C.start_run(d)
client = L.LLMClient(MODEL, provider="aws")

# Three calls, tagged across two phases to exercise per-phase aggregation.
prompts = [
    ("plan_generation", "Reply with exactly: OK"),
    ("plan_generation", "List the first 3 prime numbers, comma-separated, nothing else."),
    ("code_generation", "Reply with exactly: DONE"),
]
for phase, prompt in prompts:
    L.llm_phase.set(phase)
    client.chat(prompt, num_samples=1, temperature=0)

per_call = list(client.collect_usage())  # list of usage dicts
print(f"  recorded {len(per_call)} calls")
check("recorded one usage entry per call", len(per_call) == len(prompts))

# B1: per-call USD == recomputed from its own tokens
ok = True
for u in per_call:
    exp = round((u["input_tokens"] * PRICE["in"] + u["output_tokens"] * PRICE["out"]
                 + u["cache_write_tokens"] * PRICE["in"] * 1.25
                 + u["cache_read_tokens"] * PRICE["in"] * 0.10) / 1e6 * MULT, 6)
    if abs(u["cost_usd"] - exp) > 1e-6:
        ok = False
        print(f"      mismatch: tokens in={u['input_tokens']} out={u['output_tokens']} "
              f"recorded=${u['cost_usd']} expected=${exp}")
check("per-call USD matches token*price*multiplier", ok)

# B2: ledger totals == sum of per-call
led = C.running_total()
check("ledger call count", led["calls"] == len(per_call))
check("ledger input tokens == sum", led["input_tokens"] == sum(u["input_tokens"] for u in per_call))
check("ledger output tokens == sum", led["output_tokens"] == sum(u["output_tokens"] for u in per_call))
check("ledger total USD == sum per-call",
      abs(led["total_usd"] - round(sum(u["cost_usd"] for u in per_call), 6)) < 1e-6,
      f"${led['total_usd']:.6f}")

# B3: per-phase aggregation
agg = L.aggregate_usage(per_call)
check("aggregation split into 2 phases", set(agg.keys()) == {"plan_generation", "code_generation"},
      f"phases={sorted(agg.keys())}")

# B4: live files agree with ledger
snap = json.loads((d / "cost_live.json").read_text())
check("cost_live.json total == ledger", abs(snap["total_usd"] - led["total_usd"]) < 1e-9)
lines = [json.loads(x) for x in (d / "cost_ledger.jsonl").read_text().splitlines()]
check("cost_ledger.jsonl has one line per call", len(lines) == len(per_call))
check("ledger cumulative is monotonic & ends at total",
      all(lines[i]["cumulative_usd"] <= lines[i+1]["cumulative_usd"] for i in range(len(lines)-1))
      and abs(lines[-1]["cumulative_usd"] - led["total_usd"]) < 1e-6)

# B5: FAITHFULNESS — autocomp's recorded input tokens == raw Converse usage for identical input
rt = boto3.client("bedrock-runtime", region_name=REGION)
probe = "Reply with exactly: OK"
raw = rt.converse(modelId=MODEL,
                  messages=[{"role": "user", "content": [{"text": probe}]}],
                  inferenceConfig={"maxTokens": 8, "temperature": 0})
raw_in = raw["usage"]["inputTokens"]
# the first autocomp call used the same prompt text
auto_in = per_call[0]["input_tokens"]
check("autocomp input tokens == raw Converse inputTokens (same prompt)",
      auto_in == raw_in, f"autocomp={auto_in} raw={raw_in}")

print(f"\nLive snapshot dir: {d}")
print("cost_live.json:")
print((d / "cost_live.json").read_text())

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)
