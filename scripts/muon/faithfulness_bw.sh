#!/usr/bin/env bash
# Per-op bandwidth check: for each (prob, baseline, best) report total cycles and speedup at
# bw=6 and bw=64, to confirm the bandwidth setting does not mask compute wins.
set -uo pipefail
GMEM=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron/config/timing/gmem.toml
RP=/scratch/agustin/projects/autocomp/scripts/muon/run_problem.sh
BK=/scratch/agustin/projects/autocomp/muon_search_data/best_kernels
SOLS=/scratch/agustin/projects/autocomp/sols/muon
cp "$GMEM" /tmp/gmem.restore.bak

# prob : best-kernel : human label
ROWS=(
  "1:$BK/muon_1_beam_iters8.cpp:conv_patchembed"
  "5:$BK/muon_5_beam_iters8.cpp:softmax"
  "8:$BK/muon_8_beam_iters8.cpp:layer_norm"
  "9:$BK/muon_9_beam_iters8.cpp:gelu"
)

cyc() { # "PASS cycles=NNN" -> NNN ; else the verdict word
  local s="$1"
  if [[ "$s" == PASS* ]]; then echo "$s" | grep -oE 'cycles=[0-9]+' | cut -d= -f2; else echo "${s%% *}"; fi
}

set_bw() { sed -i "s/^bytes_per_cycle = [0-9]*\$/bytes_per_cycle = $1/" "$GMEM"; }  # only data-level lines are =6/=64? no: see note
# NOTE: gmem has tag/mshr=64 and data=6. We must toggle ONLY the data level. Use the backup approach:
#   bw6  = restore backup (data=6)
#   bw64 = backup with the 3 data '=6' lines -> '=64'
mk_bw64() { sed 's/^bytes_per_cycle = 6$/bytes_per_cycle = 64/' /tmp/gmem.restore.bak > "$GMEM"; }
mk_bw6()  { cp /tmp/gmem.restore.bak "$GMEM"; }

echo "prob,label,bw,base_cyc,best_cyc,speedup"
for r in "${ROWS[@]}"; do
  IFS=':' read -r N BEST LABEL <<< "$r"
  for BW in 6 64; do
    if [ "$BW" = 6 ]; then mk_bw6; else mk_bw64; fi
    b=$(cyc "$(bash "$RP" "$N" "$SOLS/sol${N}_baseline.cpp" "fb_${N}_base" 2>/dev/null | tail -1)")
    k=$(cyc "$(bash "$RP" "$N" "$BEST" "fb_${N}_best" 2>/dev/null | tail -1)")
    sp="-"
    if [[ "$b" =~ ^[0-9]+$ && "$k" =~ ^[0-9]+$ ]]; then sp=$(python3 -c "print(f'{$b/$k:.2f}')"); fi
    echo "$N,$LABEL,bw$BW,$b,$k,${sp}x"
  done
done
mk_bw6
echo "RESTORED_BW6"
