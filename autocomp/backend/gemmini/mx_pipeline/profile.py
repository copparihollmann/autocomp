"""Cycle-phase profiler for MX-Gemmini kernels.

Splits a kernel's spike cycle count by phase so optimization effort is directed
by data, not guesswork. A kernel body marks region boundaries with comments:

    // PROF cpu_stage
    ...host staging...
    // PROF accel
    ...gemmini launch...
    // PROF cpu_store

`instrument(body)` rewrites each `// PROF <name>` into a `_prof_switch("<name>")`
call; the harness preamble accumulates read_cycles() deltas per name and prints
`PROFILE <name>=<cycles>` lines. Phases accumulate across loop iterations.

Used in Phase 1 (discover bottlenecks across all problems) and Phase 2 (re-profile
after rebuilding a baseline to confirm the CPU share dropped). $0 — no LLM.
"""

import re

from autocomp.backend.gemmini import gemmini_eval as E
from autocomp.backend.gemmini.mx_pipeline.matmul_gen import GEMMINI_PATH

# Inserted near the top of a profiling program; defines the phase accumulators.
PROF_PREAMBLE = r'''
#define PROF_MAX 12
static const char *_prof_names[PROF_MAX];
static uint64_t _prof_cyc[PROF_MAX];
static int _prof_n = 0, _prof_cur = -1;
static uint64_t _prof_t0 = 0;
static int _prof_id(const char *nm) {
  for (int i = 0; i < _prof_n; i++) if (_prof_names[i] == nm) return i;
  _prof_names[_prof_n] = nm; _prof_cyc[_prof_n] = 0; return _prof_n++;
}
static inline void _prof_switch(const char *nm) {
  uint64_t t = read_cycles();
  if (_prof_cur >= 0) _prof_cyc[_prof_cur] += t - _prof_t0;
  _prof_cur = _prof_id(nm); _prof_t0 = read_cycles();
}
static inline void _prof_stop(void) {
  uint64_t t = read_cycles();
  if (_prof_cur >= 0) _prof_cyc[_prof_cur] += t - _prof_t0;
  _prof_cur = -1;
}
static void _prof_report(void) {
  for (int i = 0; i < _prof_n; i++)
    printf("PROFILE %s=%llu\n", _prof_names[i], (unsigned long long)_prof_cyc[i]);
}
'''


def instrument(body: str) -> str:
    """Turn `// PROF <name>` markers into phase switches."""
    return re.sub(r"//\s*PROF\s+(\w+)", lambda m: f'_prof_switch("{m.group(1)}");', body)


def run_profile(src: str, timeout: float = 3000) -> dict:
    """Build+run a profiling program on spike; return {phase: cycles}."""
    rd = {}
    E.run_spike(src, rd, GEMMINI_PATH, "0", timeout)
    out = rd.get("retval", "")
    if not isinstance(out, str):
        raise RuntimeError(f"profile run failed: {out}")
    phases = {m.group(1): int(m.group(2)) for m in re.finditer(r"PROFILE (\w+)=(\d+)", out)}
    if not phases:
        raise RuntimeError(f"no PROFILE lines (compile/run issue):\n{out[-400:]}")
    return phases


def report(phases: dict) -> str:
    tot = sum(phases.values()) or 1
    cpu = sum(v for k, v in phases.items() if k.startswith("cpu"))
    lines = [f"  total: {tot:,}   CPU share: {100*cpu/tot:.1f}%"]
    for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:<12} {v:>12,}  ({100*v/tot:.1f}%)")
    return "\n".join(lines)
