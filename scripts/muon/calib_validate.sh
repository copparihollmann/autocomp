#!/usr/bin/env bash
# Calibration target harness. Runs the 4 reference kernels on cyclotron with the CURRENT config
# and reports cycles (or TIMEOUT/FAIL) + the two rankings we must fix:
#   matmul: naive vs SMEM   -> want SMEM FASTER (RTL ~2.5x; naive/smem ratio target ~2.5)
#   attn:   global vs SMEM  -> want SMEM not-pathological (currently SMEM times out)
# Also prints PASS/FAIL (correctness must hold) per kernel.
# Usage: calib_validate.sh [label] [per-kernel-timeout-sec(default 120)]
set -uo pipefail
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron
CFG=/scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml
K=/scratch/agustin/projects/radiance-kernels/kernels
LABEL=${1:-run}; TO=${2:-120}

run() { # dir -> "cycles|verdict"
  local elf=$K/autocomp_prof_$1/kernel.radiance.elf
  local out; out=$(timeout "$TO" env RUST_LOG=error "$CYC/target/release/cyclotron" "$CFG" \
        --binary-path "$elf" --timing --log 0 2>&1)
  local rc=$?
  if [ $rc -eq 124 ]; then echo "TIMEOUT"; return; fi
  local cyc; cyc=$(echo "$out" | grep -aoE 'finished after [0-9]+ cycles' | grep -oE '[0-9]+' | tail -1)
  local ok="FAIL"
  echo "$out" | grep -aqE 'Error: 0|isa-test passed' && ok="PASS"
  echo "$out" | grep -aq globalOverSubscription && ok="ILLEGAL"
  echo "${cyc:-?}/${ok}"
}

mn=$(run naive); ms=$(run mmsmem); ab=$(run attn); as=$(run atsmem)
mnc=${mn%%/*}; msc=${ms%%/*}; abc=${ab%%/*}; asc=${as%%/*}
mr=$(python3 -c "print(f'{int('$mnc')/int('$msc'):.2f}') if '$mnc'.isdigit() and '$msc'.isdigit() else print('-')" 2>/dev/null)
ar=$(python3 -c "print(f'{int('$abc')/int('$asc'):.2f}') if '$abc'.isdigit() and '$asc'.isdigit() else print('-')" 2>/dev/null)
echo "[$LABEL] matmul naive=$mn smem=$ms (naive/smem=${mr}x, want ~2.5) | attn base=$ab smem=$as (base/smem=${ar}x, want >=1)"
