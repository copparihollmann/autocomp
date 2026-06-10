#!/usr/bin/env bash
# Calibration sweep harness: run the 4 reference kernels on cyclotron with the CURRENT
# config and report cycles + the ranking checks we care about:
#   - naive vs smem (matmul): RTL says smem ~2.5x FASTER; cyclotron currently shows ~equal.
#   - conv / attn baselines: sanity anchors (their search rankings already work; don't break them).
# Usage: sweep_rank.sh [label]
set -uo pipefail
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron
CFG=/scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml
K=/scratch/agustin/projects/radiance-kernels/kernels
LABEL=${1:-run}

run() { # dir
  local elf=$K/autocomp_prof_$1/kernel.radiance.elf
  RUST_LOG=error timeout 200 "$CYC/target/release/cyclotron" "$CFG" \
    --binary-path "$elf" --timing --log 0 2>&1 \
    | grep -oE 'finished after [0-9]+ cycles' | grep -oE '[0-9]+' | tail -1
}

n=$(run naive); s=$(run smem); c=$(run conv); a=$(run attn)
ratio=$(python3 -c "print(f'{${n:-0}/${s:-1}:.2f}')" 2>/dev/null)
echo "[$LABEL] naive=$n smem=$s conv=$c attn=$a | naive/smem=${ratio}x (target ~2.5; >1 = smem wins)"
