#!/usr/bin/env bash
# RTL gate for COMBINED (Muon SIMT + MX-Gemmini) problems.
#   vcs_gate_mx.sh <prob_id> [candidate.cpp] [tag]
#
# No new RTL config is needed: RadianceSingleClusterConfig ALREADY instantiates the MX
# accelerator (WithRadianceMxGemmini(InCluster(0), dim=16) ++ WithMuonCores(2, ...)), so
# the same simv that gates the SIMT-only problems is the combined SoC. The only deltas vs
# vcs_gate.sh are staging mxgemm_lib.hpp (the MMIO driver the harness includes) and forcing
# a rebuild, since the Makefile has no dependency on `data`/`mxgemm_lib.hpp`.
#
# Builds <k>.soc.elf (host + GPU) and runs the VCS sim. PASS = tohost ecall code 0.
# RTL cycles = $finish simulation time / clock period.
set -eo pipefail
N=$1
SOL=${2:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
TAG=${3:-vcs_mx_t$N}
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
CHIPYARD=/scratch/agustin/projects/chipyard
SIMV=$CHIPYARD/sims/vcs/simv-chipyard.harness-RadianceSingleClusterConfig

export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}
# SOC_DIR defaults to ../../soc inside radiance-kernels
export RISCV64_TOOLCHAIN_PATH=${RISCV:-$CHIPYARD/.conda-env/riscv-tools}

mkdir -p "$RK"
[ -f "$H/data" ] || (cd "$H" && python3 gen_data.py >/dev/null)
python3 - "$H/test$N.cpp" "$SOL" "$RK/kernel.cpp" <<'PY'
import sys
h, s, out = sys.argv[1:4]
code = open(s).read()
text = open(h).read()
a = text.index("// SUBSTITUTE HERE") + len("// SUBSTITUTE HERE")
b = text.index("// SUBSTITUTE END")
open(out, "w").write(text[:a] + "\n" + code + "\n" + text[b:])
PY
cp "$H/data" "$H/Makefile" "$H/host.cpp" "$RK/"
cp /scratch/agustin/projects/radiance-kernels/kernels/gemm_mxgemmini/mxgemm_lib.hpp "$RK/"

# DRAIN_ITERS covers the SIMT C-store write-drain race before verify. 200000 is safe but
# ~92% of sim cycles -- it buries the kernel-cycle signal (and its run-to-run variance
# exceeds the kernel entirely), making the total useless for perf calibration. The race
# only needs to outlast memory latency (~tens of cycles), so a few thousand suffices.
# Override with DRAIN=<n> for calibration runs (e.g. DRAIN=2000); default stays safe.
DRAIN=${DRAIN:-200000}
rm -f "$RK"/kernel.mu.o "$RK"/kernel.soc.elf "$RK"/kernel.radiance.elf
make -C "$RK" EXTRA_MU_CFLAGS="-DDRAIN_ITERS=$DRAIN" kernel.soc.elf > "$RK/build.log" 2>&1 || { echo "COMPILE-FAIL"; tail -5 "$RK/build.log"; exit 1; }

cd "$CHIPYARD/sims/vcs"
# Unique per-tag trace-db so concurrent / repeated runs don't collide on the embedded
# cyclotron trace sqlite (default path is named after the ELF -> kernel.soc.sqlite).
TRACE_DB="$RK/trace_${TAG}.sqlite"
rm -f "$TRACE_DB"
OUT=$(timeout 1800 "$SIMV" +permissive +verbose +max-cycles=10000000 \
  +trace-db="$TRACE_DB" \
  +loadmem="$RK/kernel.soc.elf" +permissive-off "$RK/kernel.soc.elf" 2>&1) || true
echo "$OUT" > "$RK/vcs_out.log"

# Verdict (TestDriver.v): success branch ($finish at line 158, "*** PASSED ***"
# with +verbose) when host tohost==1; failure branch ($fatal, "*** FAILED ***")
# when tohost>=2 or max-cycles timeout. The host returns 0 -> tohost=1 on a clean
# GPU completion, so a silent line-158 $finish with no FAILED is also a PASS.
CYCLES=$(echo "$OUT" | grep -oE 'after[[:space:]]+[0-9]+ simulation cycles' | grep -oE '[0-9]+' | tail -1)
if [ -z "$CYCLES" ]; then
  TIME_PS=$(echo "$OUT" | grep -oE '\$finish at simulation time[[:space:]]+[0-9]+' | grep -oE '[0-9]+' | tail -1)
  CYCLES=$((${TIME_PS:-0} / 2000)) # tile clock 500MHz = 2000ps (was /500, a 4x bug)
fi
if echo "$OUT" | grep -q '\*\*\* FAILED \*\*\*'; then
  TOHOST=$(echo "$OUT" | grep -oE 'tohost = [0-9]+' | grep -oE '[0-9]+' | tail -1)
  echo "FAIL rtl_cycles=${CYCLES} errors=$(( ${TOHOST:-0} >> 1 ))"
elif echo "$OUT" | grep -q '\*\*\* PASSED \*\*\*' || echo "$OUT" | grep -qE 'TestDriver\.v", line 158'; then
  echo "PASS rtl_cycles=${CYCLES}"
else
  echo "UNKNOWN rtl_cycles=${CYCLES}"
  echo "$OUT" | tail -3
fi
