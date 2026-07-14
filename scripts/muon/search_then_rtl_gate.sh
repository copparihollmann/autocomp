#!/usr/bin/env bash
# Close the performance ladder: cyclotron search -> RTL-gate the top-K candidates -> re-rank on
# RTL numbers. Cyclotron mis-ranks memory/fusion optimizations (GEMV: cyclotron 1.00x vs RTL
# 1.05x; SMEM blind spot), so the RTL cycle count is the arbiter, not the cyclotron score.
#
#   search_then_rtl_gate.sh <prob_id> <backend: simt|mx> [--gate-only <output_dir>] [--topk N]
#
# Default: runs the cyclotron search (SPENDS $), then RTL-gates. --gate-only skips the search and
# gates an existing run's candidates (NO spend) -- use it to arbitrate a finished search.
set -uo pipefail
AC=/scratch/agustin/projects/autocomp
PROB=$1; BACKEND=$2; shift 2
GATE_ONLY=""; TOPK=3
while [ $# -gt 0 ]; do case "$1" in
  --gate-only) GATE_ONLY=$2; shift 2;;
  --topk) TOPK=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
cd "$AC"

if [ -z "$GATE_ONLY" ]; then
  echo "[ladder] cyclotron search on prob$PROB (spends \$) ..."
  source muon.env
  .venv/bin/python -m autocomp.search.run_search_muon "$PROB" >/dev/null 2>&1 || true
  OUT=$(ls -dt output/*_muon_${PROB}_*cyclotron 2>/dev/null | head -1)
else
  OUT=$GATE_ONLY
fi
[ -d "$OUT" ] || { echo "[ladder] no search output dir found"; exit 1; }
echo "[ladder] gating candidates from: $OUT"

# Collect top-K distinct candidate kernels: the final winner + the best-of-iter candidates.
WORK=$(mktemp -d); i=0
add() { [ -f "$1" ] || return; python3 - "$1" "$WORK/cand_$i.cpp" <<'PY'
import sys, re
raw=open(sys.argv[1]).read()
# clean_code: keep the longest fenced block defining kernel_body, else the whole file
fences=re.findall(r"```(?:[a-zA-Z+]+)?\n(.*?)```", raw, re.DOTALL)
wk=[f for f in fences if "kernel_body" in f]
code=max(wk,key=len).strip() if wk else (max(fences,key=len).strip() if fences else raw)
open(sys.argv[2],"w").write(code)
PY
  i=$((i+1)); }
add "$OUT/best_candidate_so_far.cpp"
for d in $(ls -dt "$OUT"/candidates-iter-* 2>/dev/null | head -"$TOPK"); do add "$d/candidate_0.txt"; done

# RTL-measure baseline + each candidate. simt -> rtl_measure_simt.sh; mx -> rtl_gate_mx.sh.
measure() { # measure <sol.cpp> <tag> ; echoes cycles
  if [ "$BACKEND" = simt ]; then
    bash "$AC/scripts/muon/rtl_measure_simt.sh" "$PROB" "$1" "$2" 2>/dev/null | grep -oE "kernel_cycles=[0-9]+" | grep -oE "[0-9]+"
  else
    SIM=verilator DRAIN=2000 bash "$AC/scripts/muon/rtl_gate_mx.sh" "$PROB" "$1" "$2" 2>/dev/null | grep -oE "kernel_cycles=[0-9]+" | grep -oE "[0-9]+"
  fi
}
BASE=$(measure "$AC/sols/muon/sol${PROB}_baseline.cpp" "ladder_base_${PROB}")
echo "[ladder] RTL baseline=${BASE:-?} cycles"
best_c=$BASE; best_f="baseline"
for f in "$WORK"/cand_*.cpp; do
  [ -s "$f" ] || continue
  c=$(measure "$f" "ladder_$(basename "$f" .cpp)_${PROB}")
  echo "[ladder]   $(basename "$f"): RTL=${c:-fail} cycles $( [ -n "$c" ] && [ -n "$BASE" ] && python3 -c "print(f'({$BASE/$c:.2f}x)')" 2>/dev/null)"
  [ -n "$c" ] && [ "$c" -gt 0 ] && [ "$c" -lt "${best_c:-999999999}" ] && { best_c=$c; best_f=$(basename "$f"); }
done
echo "[ladder] RTL WINNER: $best_f @ ${best_c} cycles ($(python3 -c "print(f'{${BASE:-0}/$best_c:.2f}x vs baseline')" 2>/dev/null))"
rm -rf "$WORK"
