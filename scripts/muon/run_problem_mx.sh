#!/usr/bin/env bash
# Build + run one COMBINED (Muon SIMT + MX-Gemmini) autocomp problem on cyclotron.
#   run_problem_mx.sh <prob_id> [candidate.cpp] [tag]
#
# Differs from run_problem.sh in two ways:
#   1. stages mxgemm_lib.hpp (the MMIO/accelerator driver) into the build dir
#   2. runs cyclotron with CYCLOTRON_MXGEMMINI=1 so the MX accelerator co-model is live.
#      Without it, stores to the Gemmini MMIO block are dead writes and C comes out zero.
#
# Verdict = harness exit code (tohost): 0 = PASS, else error count. The harness compares
# C bit-exactly against a golden from scripts/muon/mx_golden (bit-exact vs real spike).
set -eo pipefail
N=$1
SOL=${2:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
TAG=${3:-t$N}
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron
MXGEMM_LIB=/scratch/agustin/projects/radiance-kernels/kernels/gemm_mxgemmini/mxgemm_lib.hpp
export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}

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
cp "$H/data" "$H/Makefile" "$H/host.cpp" "$RK/" 2>/dev/null || true
cp "$MXGEMM_LIB" "$RK/"

# The Makefile has no dependency on `data`/`mxgemm_lib.hpp`, so a changed golden or driver
# would silently reuse a stale object. Always rebuild.
rm -f "$RK"/kernel.mu.o "$RK"/kernel.radiance.elf
make -C "$RK" kernel.radiance.elf >/dev/null 2>&1 || { echo "COMPILE-FAIL"; exit 1; }

OUT=$(CYCLOTRON_MXGEMMINI=1 RUST_LOG=error timeout 300 "$CYC/target/release/cyclotron" \
  /scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml \
  --binary-path "$RK/kernel.radiance.elf" --timing --log 0 2>&1) || true
CYCLES=$(echo "$OUT" | grep -oE 'finished after [0-9]+ cycles' | grep -oE '[0-9]+' | tail -1)

# Surface accelerator modeling gaps rather than letting them look like kernel bugs.
if echo "$OUT" | grep -q "\[mxgemmini\] WARNING"; then
  echo "$OUT" | grep "\[mxgemmini\] WARNING" | head -2 >&2
fi

if echo "$OUT" | grep -qE "Error: 0$|isa-test passed"; then
  echo "PASS cycles=${CYCLES:-unknown}"
elif echo "$OUT" | grep -q "case="; then
  echo "FAIL errors=$(echo "$OUT" | grep -oE 'case=[0-9]+' | head -1 | cut -d= -f2) cycles=${CYCLES:-unknown}"
else
  echo "SIM-FAIL"
  exit 1
fi
