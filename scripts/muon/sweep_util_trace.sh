#!/usr/bin/env bash
# Build + Verilator-trace one kernel for the utilization sweep (Part C), optionally with a WARM
# double-schedule (Part B.3): the harness runs kernel_body TWICE back-to-back so hw_util_phased.py
# --warm-passes 2 can measure the steady 2nd pass and report the cold pass-1 penalty.
#
#   sweep_util_trace.sh <prob_id> <tag> [sol.cpp]
#   env: WARM=2  inject the double-schedule (default 1 = single pass)
#        DRAIN   write-drain iters (default 2000 -- cycles are drain-independent, read from trace)
#
# Cycle counts are a property of the elaborated tapeout-330 RTL, extracted from the embedded
# cyclotron trace-db (drain-independent), so DRAIN can stay small and the verify verdict is moot.
set -uo pipefail
N=$1; TAG=$2
SOL=${3:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
WARM=${WARM:-1}; DRAIN=${DRAIN:-2000}
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
CHIPYARD=/scratch/agustin/projects/chipyard
SIMV=$CHIPYARD/sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig
export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}
export RISCV64_TOOLCHAIN_PATH=${RISCV:-$CHIPYARD/.conda-env/riscv-tools}
[ -x "$SIMV" ] || { echo "$TAG NO-SIMV"; exit 2; }
mkdir -p "$RK"
[ -f "$H/data" ] || (cd "$H" && python3 gen_data.py >/dev/null 2>&1)

# 1) optional warm double-schedule injection into a temp copy of the harness main
SRC="$H/test$N.cpp"; MAIN="$RK/harness_main.cpp"
python3 - "$SRC" "$MAIN" "$WARM" <<'PY'
import re, sys
src, out, warm = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(src).read().splitlines(keepends=True)
if warm == 2:
    si = next((i for i, l in enumerate(lines) if re.search(r"mu_schedule\(\s*kernel_body", l)), None)
    if si is not None:
        bi = next((j for j in range(si, len(lines)) if re.search(r"mu_barrier\(\s*0", lines[j])), si)
        block = lines[si:bi + 1]                       # schedule .. barrier(+fence) inclusive
        lines = lines[:bi + 1] + ["  // --- WARM 2nd pass (double-schedule) ---\n"] + block + lines[bi + 1:]
open(out, "w").write("".join(lines))
PY

# 2) substitute the candidate solution into the (possibly warm) harness main -> kernel.cpp
python3 - "$MAIN" "$SOL" "$RK/kernel.cpp" <<'PY'
import sys; h, s, o = sys.argv[1:4]
t = open(h).read(); c = open(s).read()
a = t.index("// SUBSTITUTE HERE") + len("// SUBSTITUTE HERE"); b = t.index("// SUBSTITUTE END")
open(o, "w").write(t[:a] + "\n" + c + "\n" + t[b:])
PY

cp "$H/data" "$H/Makefile" "$RK/" 2>/dev/null
[ -f "$H/host.cpp" ] && cp "$H/host.cpp" "$RK/"
cp /scratch/agustin/projects/radiance-kernels/kernels/gemm_mxgemmini/mxgemm_lib.hpp "$RK/" 2>/dev/null
rm -f "$RK"/kernel.mu.o "$RK"/kernel.soc.elf "$RK"/kernel.radiance.elf
make -C "$RK" EXTRA_MU_CFLAGS="-DDRAIN_ITERS=$DRAIN ${EXTRA_CFLAGS:-}" kernel.soc.elf > "$RK/build.log" 2>&1 \
  || { echo "$TAG COMPILE-FAIL"; tail -4 "$RK/build.log"; exit 1; }

TR="$RK/trace_$TAG.sqlite"; rm -f "$TR"
# NOTE: NO +verbose -- it dumps every instruction to stdout (190-260MB logs) and makes concurrent
# sims catastrophically I/O-bound. The trace-db .sqlite (what we parse) is independent of +verbose.
bash -c "ulimit -s unlimited; exec timeout 14400 '$SIMV' +permissive +max-cycles=100000000 \
      +trace-db='$TR' +loadmem='$RK/kernel.soc.elf' +permissive-off '$RK/kernel.soc.elf'" \
      > "$RK/sim.log" 2>&1 || true
[ -s "$TR" ] && echo "$TAG TRACE-OK $TR" || { echo "$TAG SIM-FAIL"; tail -4 "$RK/sim.log"; exit 1; }
