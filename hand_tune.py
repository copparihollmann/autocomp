#!/usr/bin/env python3
"""Hand-tuning measurement loop for MX-Gemmini kernels.

Reuses the SAME spike metric autocomp uses (read_cycles around the whole solution() = host+accelerator,
see search/prob.py) so hand-tuned kernels are directly comparable to autocomp scores. Each measured
attempt is appended to hand_tune_ledger.jsonl (schema mirrors autocomp's output/transform_ledger.jsonl)
so the transformations + reasoning + measured deltas are mineable / re-injectable into autocomp.

Usage:
  python hand_tune.py measure <prob_type> <prob_id> [kernel.c]      # latency+correct of a kernel (or baseline)
  python hand_tune.py record <prob_type> <prob_id> <kernel.c> --transformation "..." --reasoning "..." \
                       --lever name:before:after --base-latency N    # measure + append a ledger record
"""
import argparse, hashlib, json, os, pathlib, time
from autocomp.search.prob import Prob
from autocomp.search.search import load_initial_code
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.hw_config import GemminiHardwareConfig

LEDGER = pathlib.Path(__file__).parent / "hand_tune_ledger.jsonl"


def _backend():
    return GemminiEvalBackend(GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64))


def measure(prob_type: str, prob_id: int, body: str | None = None):
    """Return (latency_cycles_or_None, correct_bool, float_ops). body=None -> the generated baseline."""
    prob = Prob(prob_type, prob_id)
    if body is None:
        body = load_initial_code("gemmini", prob)
    from autocomp.backend.gemmini.gemmini_eval import _count_host_float_ops
    stats = _backend().evaluate_code(prob, [body], "spike")[0]
    lat = stats.get("latency") if stats.get("correct") else None
    return lat, bool(stats.get("correct")), _count_host_float_ops(body), stats.get("stdout", "")


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def record(prob_type, prob_id, kernel_path, transformation, reasoning, lever, base_latency, base_path=""):
    body = pathlib.Path(kernel_path).read_text()
    lat, correct, nflt, _ = measure(prob_type, prob_id, body)
    speedup = (base_latency / lat) if (lat and base_latency) else None
    outcome = ("incorrect" if not correct else
               "improved" if (lat and base_latency and lat < base_latency) else
               "regressed" if (lat and base_latency and lat > base_latency) else "correct_no_gain")
    name, _, rest = (lever or "").partition(":")
    before, _, after = rest.partition(":")
    rec = {
        "ts": int(time.time()), "kernel_family": prob_type, "prob_id": prob_id,
        "transformation": transformation, "reasoning": reasoning,
        "lever": {"name": name, "before": before, "after": after} if name else None,
        "latency_before": base_latency, "latency_after": lat,
        "speedup": round(speedup, 4) if speedup else None,
        "correct": correct, "host_float_ops": nflt, "outcome": outcome,
        "after_sha": _sha(body), "after_path": str(kernel_path), "before_path": base_path,
        "measured_on": "spike",
    }
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("prob_type"); m.add_argument("prob_id", type=int)
    m.add_argument("kernel", nargs="?", default=None)
    r = sub.add_parser("record"); r.add_argument("prob_type"); r.add_argument("prob_id", type=int)
    r.add_argument("kernel"); r.add_argument("--transformation", required=True); r.add_argument("--reasoning", required=True)
    r.add_argument("--lever", default=""); r.add_argument("--base-latency", type=int, required=True); r.add_argument("--base-path", default="")
    a = ap.parse_args()
    if a.cmd == "measure":
        body = pathlib.Path(a.kernel).read_text() if a.kernel else None
        lat, correct, nflt, _ = measure(a.prob_type, a.prob_id, body)
        print(f"latency={lat} correct={correct} host_float_ops={nflt}")
    else:
        record(a.prob_type, a.prob_id, a.kernel, a.transformation, a.reasoning, a.lever, a.base_latency, a.base_path)
