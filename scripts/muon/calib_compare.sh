#!/usr/bin/env bash
# Apples-to-apples calibration table: symbol-based kernel_body span (kernel_utilization.py)
# on BOTH the tapeout-330 RTL trace (fix_tN, if present) and the cyclotron fast-sim trace,
# for a given (KCMP,KDMA). The symbol method (ELF PC ranges) is drain-structure-independent,
# so it is valid on cyclotron (DRAIN=0) and RTL (DRAIN=2000) alike -- unlike the drain
# heuristic in rtl_kernel_cycles.py.
#   calib_compare.sh <KCMP> <KDMA>
set -uo pipefail
cd /scratch/agustin/projects/autocomp
KCMP=${1:-1804}; KDMA=${2:-143}
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron/target/release/cyclotron
KU=scripts/muon/kernel_utilization.py
RKROOT=/scratch/agustin/projects/radiance-kernels/kernels
kcyc() { python3 "$KU" "$1" "$2" "$3" "$4" "$5" --fmt fp8 2>/dev/null | grep -oE "actual cycles      : [0-9]+" | grep -oE "[0-9]+"; }

printf "%-6s %-6s %-12s %-12s %-8s\n" prob tiles RTL_kcyc CYC_kcyc ratio
for spec in "20 1 64" "23 2 128" "22 8 512" "24 16 1024"; do
  set -- $spec; N=$1; T=$2; K=$3
  # RTL (symbol-based) from the fix_tN trace if it exists
  RTRACE="$RKROOT/autocomp_fix_t$N/trace_fix_t$N.sqlite"
  RELF="$RKROOT/autocomp_fix_t$N/kernel.radiance.elf"
  RC=""; [ -f "$RTRACE" ] && [ -f "$RELF" ] && RC=$(kcyc "$RELF" "$RTRACE" 64 64 "$K")
  # cyclotron fast-sim: run, then symbol-based on the fresh trace
  CELF="$RKROOT/autocomp_calib_t$N/kernel.radiance.elf"
  CC=""
  if [ -f "$CELF" ]; then
    rm -f kernel.radiance.sqlite
    CYCLOTRON_MXGEMMINI=1 CYCLOTRON_MXGEMMINI_KCMP=$KCMP CYCLOTRON_MXGEMMINI_KDMA=$KDMA \
      RUST_LOG=error timeout 200 "$CYC" scripts/muon/config_muon.toml \
      --binary-path "$CELF" --timing --gen-trace true --log 0 >/dev/null 2>&1
    [ -f kernel.radiance.sqlite ] && CC=$(kcyc "$CELF" kernel.radiance.sqlite 64 64 "$K")
  fi
  R="-"; [ -n "$RC" ] && [ -n "$CC" ] && [ "$CC" -gt 0 ] && R=$(python3 -c "print(f'{$RC/$CC:.2f}')" 2>/dev/null)
  printf "%-6s %-6s %-12s %-12s %-8s\n" "prob$N" "$T" "${RC:-pending}" "${CC:-?}" "$R"
done