#!/usr/bin/env bash
# Measure kernel-body cycles of a SIMT muon kernel on the (trace-fixed) Verilator RTL.
# Unlike cyclotron, RTL exposes memory-hierarchy wins (SMEM caching, coalescing) — so this is
# where a hand-optimized memory-bound kernel's speedup actually shows.
#   rtl_measure_simt.sh <prob_id> <sol.cpp> <tag>
# Prints: actual kernel-body cycles (symbol-based, drain-independent).
set -uo pipefail
N=$1; SOL=$2; TAG=$3
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
SIMV=/scratch/agustin/projects/chipyard/sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig
KU=/scratch/agustin/projects/autocomp/scripts/muon/kernel_utilization.py
export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}
export RISCV64_TOOLCHAIN_PATH=${RISCV:-/scratch/agustin/projects/chipyard/.conda-env/riscv-tools}
mkdir -p "$RK"
[ -f "$H/data" ] || (cd "$H" && python3 gen_data.py >/dev/null)
# substitute candidate into the flat harness -> kernel.cpp
python3 - "$H/test$N.c" "$SOL" "$RK/kernel.cpp" <<'PY'
import sys; h,s,o=sys.argv[1:4]
t=open(h).read(); c=open(s).read()
a=t.index("// SUBSTITUTE HERE")+len("// SUBSTITUTE HERE"); b=t.index("// SUBSTITUTE END")
open(o,"w").write(t[:a]+"\n"+c+"\n"+t[b:])
PY
cp "$H/data" "$H/Makefile" "$RK/"; [ -f "$H/host.cpp" ] && cp "$H/host.cpp" "$RK/"
rm -f "$RK"/kernel.mu.o "$RK"/kernel.radiance.elf "$RK"/kernel.soc.elf
make -C "$RK" kernel.soc.elf > "$RK/build.log" 2>&1 || { echo "$TAG COMPILE-FAIL"; tail -4 "$RK/build.log"; exit 1; }
TR="$RK/trace_$TAG.sqlite"; rm -f "$TR"
bash -c "ulimit -s unlimited; exec timeout 14400 '$SIMV' +permissive +verbose +max-cycles=100000000 \
      +trace-db='$TR' +loadmem='$RK/kernel.soc.elf' +permissive-off '$RK/kernel.soc.elf'" > "$RK/sim.log" 2>&1 || true
# shape args for kernel_utilization (only affects the 'ideal' col; 'actual' is what we compare)
CYC=$(python3 "$KU" "$RK/kernel.radiance.elf" "$TR" 1 512 512 2>/dev/null | grep -oE "actual cycles      : [0-9]+" | grep -oE "[0-9]+")
echo "$TAG kernel_cycles=${CYC:-?}"
