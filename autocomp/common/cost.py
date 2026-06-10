"""Estimated USD cost for LLM usage (primarily AWS Bedrock).

This is a *local estimate* layered on top of the token counts autocomp already
tracks (see ``aggregate_usage`` in ``llm_utils.py``). It is NOT a billing
source of truth — reconcile against AWS Cost Explorer (filtered to Bedrock) for
actual spend. Token-price tables drift and Bedrock has charges this estimate
does not capture (CloudWatch/S3 invocation-log storage, etc.).

Prices below are USD per 1,000,000 tokens, on-demand, standard region. Verify
and pin to the exact model IDs enabled in your account against
https://aws.amazon.com/bedrock/pricing/ before trusting the numbers.

Cross-region inference pricing (per AWS docs, verified 2026-06-03): there is NO
routing surcharge. A GEOGRAPHIC profile (the "us." prefix we use,
us.anthropic.claude-sonnet-4-6) bills at STANDARD on-demand price for the source
region — i.e. multiplier 1.0, $3/$15 per 1M for Sonnet 4.6. A GLOBAL profile
("global." prefix) is ~10% CHEAPER (multiplier ~0.9). So the default multiplier
is 1.0; set AUTOCOMP_BEDROCK_REGION_MULTIPLIER=0.9 only if you switch to a global
profile. (The "+10% premium" some blogs cite is regional-relative-to-global, not
an extra charge on the geographic profile.)
Ref: docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html

Long-context tier: for Anthropic Claude, a request whose *input* exceeds
LONG_CONTEXT_THRESHOLD (200K) tokens is billed at a premium rate for the whole
request (e.g. Sonnet 4.6: $6/$22.50 instead of $3/$15). Because this triggers
per-request, cost is computed per call (where the exact input size is known),
not on summed aggregates.

Prompt caching is priced relative to the applicable input rate: cache *writes*
~1.25x input, cache *reads* ~0.1x input. Caches are regional, so cross-region
routing can cause cache misses + re-writes.
"""

import datetime
import json
import os
import threading

LONG_CONTEXT_THRESHOLD = 200_000

# USD per 1,000,000 tokens. "long_in"/"long_out" apply when a single request's
# input exceeds LONG_CONTEXT_THRESHOLD; omit them if a model has no such tier.
# Cache rates are derived from the applicable input rate (1.25x write, 0.1x read).
# Matched against the model ID by substring (longest match wins), so both
# "anthropic.claude-sonnet-4-6" and "us.anthropic.claude-sonnet-4-6-v1:0" resolve.
_PRICES_PER_M = {
    # --- Anthropic Claude (Bedrock == direct API in standard regions) ---
    # Sonnet 4.6 is the model in use here (confirmed via aws.amazon.com/bedrock/pricing).
    "sonnet": {"in": 3.0, "out": 15.0, "long_in": 6.0, "long_out": 22.5},
    "opus":   {"in": 5.0, "out": 25.0},
    "haiku":  {"in": 1.0, "out": 5.0},
    # --- Amazon Nova (approximate; no long-context tier modeled) ---
    "nova-micro": {"in": 0.035, "out": 0.14},
    "nova-lite":  {"in": 0.06,  "out": 0.24},
    "nova-pro":   {"in": 0.80,  "out": 3.20},
}

_DEFAULT_REGION_MULTIPLIER = float(
    os.environ.get("AUTOCOMP_BEDROCK_REGION_MULTIPLIER", "1.0")
)


def price_for(model: str):
    """Return the rate dict for a model id, or None if unknown (cost -> 0)."""
    if not model:
        return None
    m = model.lower()
    best_key = None
    for key in _PRICES_PER_M:
        if key in m and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return _PRICES_PER_M[best_key] if best_key else None


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    region_multiplier: float | None = None,
) -> float:
    """Estimated USD for one *request's* token usage. Returns 0.0 for unknown models.

    Applies the long-context premium when input_tokens exceeds the threshold, so
    pass per-call counts (not summed aggregates) for accurate long-context pricing.
    """
    rate = price_for(model)
    if rate is None:
        return 0.0
    long_ctx = input_tokens > LONG_CONTEXT_THRESHOLD and "long_in" in rate
    in_r = rate["long_in"] if long_ctx else rate["in"]
    out_r = rate["long_out"] if long_ctx else rate["out"]
    cw_r = in_r * 1.25   # cache write
    cr_r = in_r * 0.10   # cache read
    mult = _DEFAULT_REGION_MULTIPLIER if region_multiplier is None else region_multiplier
    usd = (
        input_tokens * in_r
        + output_tokens * out_r
        + cache_write_tokens * cw_r
        + cache_read_tokens * cr_r
    ) / 1_000_000.0
    return round(usd * mult, 6)


# ---------------------------------------------------------------------------
# Live ledger: a process-global running total updated on EVERY LLM call, so
# cost is visible in real time (not just at iteration end). record_call() writes
# an atomic snapshot to cost_live.json and appends to cost_ledger.jsonl in the
# run's output dir. Watch it live with, e.g.:
#     watch -n 2 python -m autocomp.common.cost <run_output_dir>
# or just:  watch -n 2 cat <run_output_dir>/cost_live.json
# ---------------------------------------------------------------------------

LIVE_SNAPSHOT_NAME = "cost_live.json"
LEDGER_NAME = "cost_ledger.jsonl"

_lock = threading.Lock()
_ledger = {
    "calls": 0, "input_tokens": 0, "output_tokens": 0,
    "cache_read_tokens": 0, "cache_write_tokens": 0,
    "total_usd": 0.0, "by_model": {},
}
_ledger_dir = None  # pathlib.Path | None — where snapshot/jsonl are written


def start_run(output_dir) -> None:
    """Begin a fresh live ledger for a run, writing snapshots into output_dir.
    Call once at the start of optimize(). Safe to call with None (disables files)."""
    global _ledger_dir
    with _lock:
        _ledger.update({
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "total_usd": 0.0, "by_model": {},
        })
        _ledger_dir = output_dir
    _write_snapshot()


def running_total() -> dict:
    """Return a copy of the current live ledger."""
    with _lock:
        snap = dict(_ledger)
        snap["by_model"] = dict(_ledger["by_model"])
        return snap


def record_call(model: str, usage: dict) -> dict:
    """Fold one LLM call's usage into the live ledger and persist a snapshot.
    Returns the running total. Never raises."""
    try:
        usd = usage.get("cost_usd")
        if usd is None:
            usd = estimate_cost(
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_tokens", 0),
                usage.get("cache_write_tokens", 0),
            )
        with _lock:
            _ledger["calls"] += 1
            _ledger["input_tokens"] += usage.get("input_tokens", 0)
            _ledger["output_tokens"] += usage.get("output_tokens", 0)
            _ledger["cache_read_tokens"] += usage.get("cache_read_tokens", 0)
            _ledger["cache_write_tokens"] += usage.get("cache_write_tokens", 0)
            _ledger["total_usd"] = round(_ledger["total_usd"] + (usd or 0.0), 6)
            bm = _ledger["by_model"]
            bm[model] = round(bm.get(model, 0.0) + (usd or 0.0), 6)
            running = dict(_ledger)
            running["by_model"] = dict(bm)
        _write_snapshot()
        _append_ledger_line(model, usage, usd or 0.0)
        _record_project_spend(model, usd or 0.0)  # cumulative lifetime total
        return running
    except Exception:
        return running_total()


def _write_snapshot() -> None:
    if _ledger_dir is None:
        return
    try:
        with _lock:
            data = dict(_ledger)
            data["by_model"] = dict(_ledger["by_model"])
        path = _ledger_dir / LIVE_SNAPSHOT_NAME
        tmp = _ledger_dir / (LIVE_SNAPSHOT_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic
    except Exception:
        pass


def _append_ledger_line(model: str, usage: dict, usd: float) -> None:
    if _ledger_dir is None:
        return
    try:
        line = {
            "model": model,
            "phase": usage.get("phase"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_tokens", 0),
            "cache_write_tokens": usage.get("cache_write_tokens", 0),
            "cost_usd": round(usd, 6),
            "cumulative_usd": _ledger["total_usd"],
        }
        with open(_ledger_dir / LEDGER_NAME, "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Persistent PROJECT spend: a cumulative lifetime total across ALL runs, so you
# always know exactly how much this project has cost and can stop when it's too
# much. Unlike the per-run live ledger above (which resets each run), this never
# resets. Location: $AUTOCOMP_SPEND_LOG (recommended: one file per project),
# else ~/.autocomp/spend.jsonl. A sidecar <name>_total.json holds the rolling
# grand total so we never re-scan the whole jsonl.
#
#   View anytime:   python -m autocomp.common.cost --total
#   Optional cap:   AUTOCOMP_SPEND_LIMIT_USD=50    (warns when crossed; with
#                   AUTOCOMP_SPEND_LIMIT_MODE=stop the next LLM call aborts)
# ---------------------------------------------------------------------------

SPEND_LOG_ENV = "AUTOCOMP_SPEND_LOG"
SPEND_LIMIT_ENV = "AUTOCOMP_SPEND_LIMIT_USD"
SPEND_LIMIT_MODE_ENV = "AUTOCOMP_SPEND_LIMIT_MODE"  # "warn" (default) | "stop"

_project_spend = None  # lazy dict
_budget_warned = False


class BudgetExceeded(RuntimeError):
    """Raised (in 'stop' mode) when project spend has reached the configured limit."""


def _spend_paths():
    p = os.environ.get(SPEND_LOG_ENV) or os.path.join(
        os.path.expanduser("~"), ".autocomp", "spend.jsonl"
    )
    base, _ = os.path.splitext(p)
    return p, base + "_total.json"


def _load_project_spend() -> dict:
    global _project_spend
    if _project_spend is not None:
        return _project_spend
    _, total_path = _spend_paths()
    try:
        with open(total_path) as f:
            _project_spend = json.load(f)
    except Exception:
        _project_spend = {"total_usd": 0.0, "calls": 0, "by_model": {}, "by_day": {}, "since": None}
    return _project_spend


def project_total() -> dict:
    """Return the cumulative lifetime project spend (a copy)."""
    ps = _load_project_spend()
    out = dict(ps)
    out["by_model"] = dict(ps.get("by_model", {}))
    out["by_day"] = dict(ps.get("by_day", {}))
    return out


def spend_limit() -> float | None:
    v = os.environ.get(SPEND_LIMIT_ENV)
    try:
        return float(v) if v else None
    except ValueError:
        return None


def assert_within_budget() -> None:
    """In 'stop' mode, raise BudgetExceeded once project spend hits the limit.
    Call before issuing an LLM request so spend can be hard-capped."""
    lim = spend_limit()
    if lim is None or os.environ.get(SPEND_LIMIT_MODE_ENV, "warn").lower() != "stop":
        return
    if _load_project_spend()["total_usd"] >= lim:
        raise BudgetExceeded(
            f"Project spend ${_load_project_spend()['total_usd']:.4f} has reached the "
            f"limit ${lim:.2f} (AUTOCOMP_SPEND_LIMIT_USD). Aborting before further spend."
        )


def _record_project_spend(model: str, usd: float) -> None:
    global _budget_warned
    try:
        jsonl, total_path = _spend_paths()
        os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
        now = datetime.datetime.now()
        day = now.strftime("%Y-%m-%d")
        with _lock:
            ps = _load_project_spend()
            ps["total_usd"] = round(ps.get("total_usd", 0.0) + usd, 6)
            ps["calls"] = ps.get("calls", 0) + 1
            ps["by_model"][model] = round(ps["by_model"].get(model, 0.0) + usd, 6)
            ps["by_day"][day] = round(ps["by_day"].get(day, 0.0) + usd, 6)
            if not ps.get("since"):
                ps["since"] = now.isoformat(timespec="seconds")
            ps["updated"] = now.isoformat(timespec="seconds")
            total_now = ps["total_usd"]
            # append audit line + rewrite sidecar total atomically
            with open(jsonl, "a") as f:
                f.write(json.dumps({"ts": now.isoformat(timespec="seconds"),
                                    "model": model, "cost_usd": round(usd, 6),
                                    "cumulative_usd": total_now}) + "\n")
            tmp = total_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(ps, f, indent=2)
            os.replace(tmp, total_path)
        lim = spend_limit()
        if lim is not None and total_now >= lim and not _budget_warned:
            _budget_warned = True
            import logging
            logging.getLogger(__name__).warning(
                "PROJECT SPEND $%.4f has reached limit $%.2f (AUTOCOMP_SPEND_LIMIT_USD).%s",
                total_now, lim,
                " Next call will abort (stop mode)." if os.environ.get(SPEND_LIMIT_MODE_ENV, "warn").lower() == "stop"
                else " (warn mode — set AUTOCOMP_SPEND_LIMIT_MODE=stop to hard-stop.)",
            )
    except Exception:
        pass


def cost_by_phase(usage_by_phase: dict, region_multiplier: float | None = None) -> dict:
    """Given the dict returned by aggregate_usage() ({phase: {model: {tokens...}}}),
    return {"by_phase": {phase: {model: usd}}, "by_model": {model: usd}, "total_usd": float}.

    NOTE: this recomputes from *summed* token counts, so it cannot detect per-request
    long-context crossings — it is a coarse fallback. Prefer summing the per-call
    "cost_usd" that aggregate_usage carries when available."""
    by_phase: dict[str, dict[str, float]] = {}
    by_model: dict[str, float] = {}
    unknown: set[str] = set()
    total = 0.0
    for phase, model_map in (usage_by_phase or {}).items():
        if not isinstance(model_map, dict):
            continue
        for model, data in model_map.items():
            if not isinstance(data, dict):
                continue
            usd = estimate_cost(
                model,
                data.get("input_tokens", 0),
                data.get("output_tokens", 0),
                data.get("cache_read_tokens", 0),
                data.get("cache_write_tokens", 0),
                region_multiplier,
            )
            if price_for(model) is None and (
                data.get("input_tokens") or data.get("output_tokens")
            ):
                unknown.add(model)
            by_phase.setdefault(phase, {})[model] = usd
            by_model[model] = round(by_model.get(model, 0.0) + usd, 6)
            total += usd
    return {
        "by_phase": by_phase,
        "by_model": by_model,
        "total_usd": round(total, 6),
        "unknown_models": sorted(unknown),
    }


def _fmt_live(snap: dict) -> str:
    lines = [
        f"  total:  ${snap.get('total_usd', 0.0):.4f}   ({snap.get('calls', 0)} calls)",
        f"  tokens: in {snap.get('input_tokens', 0):,}  out {snap.get('output_tokens', 0):,}"
        f"  cache(r/w) {snap.get('cache_read_tokens', 0):,}/{snap.get('cache_write_tokens', 0):,}",
    ]
    for model, usd in sorted(snap.get("by_model", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"    {model}: ${usd:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Two modes:
    #   python -m autocomp.common.cost --total        cumulative LIFETIME project spend
    #   python -m autocomp.common.cost <run_dir>      this run's live snapshot
    # Pair either with `watch -n 2 ...` for a live view.
    import sys
    args = sys.argv[1:]
    if args and args[0] in ("--total", "-t", "--project"):
        ps = project_total()
        jsonl, total_path = _spend_paths()
        print(f"AutoComp PROJECT spend — {total_path}")
        print(f"  total:  ${ps.get('total_usd', 0.0):.4f}   ({ps.get('calls', 0)} calls"
              f", since {ps.get('since', '?')})")
        lim = spend_limit()
        if lim is not None:
            pct = 100.0 * ps.get("total_usd", 0.0) / lim if lim else 0
            print(f"  limit:  ${lim:.2f}  ({pct:.1f}% used, mode={os.environ.get(SPEND_LIMIT_MODE_ENV, 'warn')})")
        for model, usd in sorted(ps.get("by_model", {}).items(), key=lambda kv: -kv[1]):
            print(f"    {model}: ${usd:.4f}")
        recent = sorted(ps.get("by_day", {}).items())[-7:]
        if recent:
            print("  last days: " + "  ".join(f"{d}:${v:.4f}" for d, v in recent))
    else:
        target = args[0] if args else "."
        snap_path = os.path.join(target, LIVE_SNAPSHOT_NAME)
        if not os.path.exists(snap_path) and os.path.basename(target) == LIVE_SNAPSHOT_NAME:
            snap_path = target
        try:
            with open(snap_path) as f:
                snap = json.load(f)
            print(f"AutoComp live cost — {snap_path}")
            print(_fmt_live(snap))
        except FileNotFoundError:
            print(f"No cost snapshot yet at {snap_path} (run hasn't made an LLM call).")
