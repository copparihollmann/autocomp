#!/usr/bin/env bash
# Calibrate the cyclotron MX-Gemmini latency coefficients against RTL.
#   calib_mx.sh <prob_id> [candidate.cpp]
#
# The co-model charges accelerator work as
#   compute cycles = TI*TJ*TK*(16 fp8 | 32 sub-byte) * K_CMP / 1000
#   dma     cycles = bytes                            * K_DMA / 1000
# and makes the kernel's gemmini_fence() spin on BUSY until that work drains, so the
# coefficients only reach total cycles when the accelerator is ON THE CRITICAL PATH.
#
# IDENTIFIABILITY WARNING. If total cycles are flat across the sweep below, this kernel is
# SIMT-bound (scale staging + C move-out + verify dominate) and CANNOT calibrate anything:
# any coefficient fits. A usable anchor must be accelerator-bound -- large K per loop_ws,
# minimal SIMT work per output element. Fitting to a flat curve produces a coefficient that
# is pure noise, and it will silently mis-rank accelerator-bound candidates later.
#
# Prints the cyclotron total for each (K_CMP, K_DMA); compare against `vcs_gate_mx.sh`.
set -eo pipefail
N=$1
SOL=${2:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
AC=/scratch/agustin/projects/autocomp
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron/target/release/cyclotron
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_calib_t$N

# Build once via the normal MX runner (it stages mxgemm_lib.hpp and forces a rebuild).
bash "$AC/scripts/muon/run_problem_mx.sh" "$N" "$SOL" "calib_t$N" >/dev/null
ELF="$RK/kernel.radiance.elf"

run() { # run <kcmp> <kdma>
  CYCLOTRON_MXGEMMINI=1 CYCLOTRON_MXGEMMINI_KCMP=$1 CYCLOTRON_MXGEMMINI_KDMA=$2 \
    RUST_LOG=error timeout 300 "$CYC" "$AC/scripts/muon/config_muon.toml" \
    --binary-path "$ELF" --timing --log 0 2>&1 |
    grep -oE 'finished after [0-9]+' | grep -oE '[0-9]+'
}

echo "kcmp,kdma,cyclotron_total_cycles"
KC_VALS="451 902 1804 3608 7216"
KD_VALS="71 143 286"
for KC in $KC_VALS; do
  for KD in $KD_VALS; do
    echo "$KC,$KD,$(run $KC $KD)"
  done
done

# Per-parameter identifiability. A coefficient only reaches total cycles when the work it
# prices is on the critical path; if the total never moves as you vary it, ANY value fits
# and a "fit" here is pure noise that will silently mis-rank accelerator-bound candidates.
DEF_KC=1804; DEF_KD=143
KD_LO=$(run $DEF_KC 71); KD_HI=$(run $DEF_KC 286)
KC_LO=$(run 451 $DEF_KD); KC_HI=$(run 3608 $DEF_KD)
echo >&2
[ "$KD_LO" = "$KD_HI" ] && echo "WARNING: K_DMA is UNIDENTIFIABLE here (4x sweep, total unchanged: $KD_LO). The operand DMA is never on the critical path." >&2
[ "$KC_LO" = "$KC_HI" ] && echo "WARNING: K_CMP is UNIDENTIFIABLE here (8x sweep, total unchanged: $KC_LO)." >&2
if [ "$KC_LO" = "$KC_HI" ] || [ "$KD_LO" = "$KD_HI" ]; then
  echo "This kernel is SIMT-bound (scale staging + C move-out dominate). Use an" >&2
  echo "accelerator-bound anchor -- large K per loop_ws, minimal SIMT per output." >&2
fi
