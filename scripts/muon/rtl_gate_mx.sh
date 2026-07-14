#!/usr/bin/env bash
# RTL gate for COMBINED (Muon SIMT + MX-Gemmini) problems, on the TAPEOUT RTL.
#   rtl_gate_mx.sh <prob_id> [candidate.cpp] [tag]
#
# env:
#   SIM=verilator|vcs   which simulator (default verilator -- needs NO licence)
#   DRAIN=<n>           DRAIN_ITERS for the write-drain race (default 200000).
#                       Use a small value (e.g. 2000) for CALIBRATION runs: kernel cycles are
#                       extracted from the trace-db and are provably drain-independent, so the
#                       verify verdict does not matter there and the sim is far faster.
#
# Both simulators run the SAME elaborated RTL (radiance tapeout-330 + gemmini tapeout-260329),
# so they give the SAME cycle counts -- cycle count is a property of the RTL, not the
# simulator. Verilator is preferred because the Synopsys licence servers are not always reachable.
#
# TRACE-DB CYCLES CAVEAT: the trace-db `cycle` field is stamped by a free-running counter in
# the TracerBlackBox (radiance/unittest/Cyclotron.scala) fed over the `cyclotron_trace` DPI.
# Stock tapeout-330 LACKS that counter (predates radiance 6462826+a4656ba), so its trace cycles
# come out as 0/1 garbage in BOTH Verilator and VCS. This build backports those 2 trace-only
# commits; without them, rtl_kernel_cycles.py returns 0. See memory tapeout330-trace-abi-mismatch.
#
# NOTE: Verilator segfaults on this design with the default 8 MB stack. `ulimit -s unlimited`
# is mandatory, not optional.
set -eo pipefail
N=$1
SOL=${2:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
SIM=${SIM:-verilator}
TAG=${3:-rtl_${SIM}_t$N}
DRAIN=${DRAIN:-200000}
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
CHIPYARD=/scratch/agustin/projects/chipyard

case "$SIM" in
  verilator) SIMBIN=$CHIPYARD/sims/verilator/simulator-chipyard.harness-RadianceSingleClusterConfig ;;
  vcs)       SIMBIN=$CHIPYARD/sims/vcs/simv-chipyard.harness-RadianceSingleClusterConfig ;;
  *) echo "SIM must be verilator|vcs"; exit 2 ;;
esac
[ -x "$SIMBIN" ] || { echo "simulator not built: $SIMBIN"; exit 2; }

export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}
export RISCV64_TOOLCHAIN_PATH=${RISCV:-$CHIPYARD/.conda-env/riscv-tools}

mkdir -p "$RK"
[ -f "$H/data" ] || (cd "$H" && python3 gen_data.py >/dev/null)
python3 - "$H/test$N.cpp" "$SOL" "$RK/kernel.cpp" <<'PY'
import sys
h, s, out = sys.argv[1:4]
text = open(h).read(); code = open(s).read()
a = text.index("// SUBSTITUTE HERE") + len("// SUBSTITUTE HERE")
b = text.index("// SUBSTITUTE END")
open(out, "w").write(text[:a] + "\n" + code + "\n" + text[b:])
PY
cp "$H/data" "$H/Makefile" "$H/host.cpp" "$RK/"
cp /scratch/agustin/projects/radiance-kernels/kernels/gemm_mxgemmini/mxgemm_lib.hpp "$RK/"
rm -f "$RK"/kernel.mu.o "$RK"/kernel.soc.elf "$RK"/kernel.radiance.elf
make -C "$RK" EXTRA_MU_CFLAGS="-DDRAIN_ITERS=$DRAIN" kernel.soc.elf > "$RK/build.log" 2>&1 \
  || { echo "COMPILE-FAIL"; tail -5 "$RK/build.log"; exit 1; }

TRACE="$RK/trace_${TAG}.sqlite"
rm -f "$TRACE"
# ulimit -s unlimited: Verilator segfaults on this design with the default 8 MB stack.
OUT=$(bash -c "ulimit -s unlimited; exec timeout 7200 '$SIMBIN' +permissive +verbose \
      +max-cycles=20000000 +trace-db='$TRACE' \
      +loadmem='$RK/kernel.soc.elf' +permissive-off '$RK/kernel.soc.elf'" 2>&1) || true
echo "$OUT" > "$RK/sim_out.log"

# Net kernel cycles (drain-independent) from the embedded cyclotron trace-db.
KC=$(python3 /scratch/agustin/projects/autocomp/scripts/muon/rtl_kernel_cycles.py "$TRACE" 2>/dev/null \
     | grep -oE "net_kernel_cycles\(core\)=[0-9]+" | grep -oE "[0-9]+" || true)

if echo "$OUT" | grep -q '\*\*\* FAILED \*\*\*\|TEST FAILED'; then
  TOHOST=$(echo "$OUT" | grep -oE 'tohost=\s*[0-9]+' | grep -oE '[0-9]+' | tail -1)
  echo "FAIL errors=$(( ${TOHOST:-0} >> 1 )) kernel_cycles=${KC:-?}"
elif echo "$OUT" | grep -q '\*\*\* PASSED \*\*\*'; then
  echo "PASS kernel_cycles=${KC:-?}"
else
  echo "UNKNOWN kernel_cycles=${KC:-?}"; echo "$OUT" | tail -3
fi
