#!/usr/bin/env bash
# Build + run one muon autocomp problem on cyclotron; report verdict + cycles.
#   run_problem.sh <prob_id> [candidate.cpp] [tag]
# Verdict = harness exit code (tohost): 0 = PASS, N = error count.
set -eo pipefail
N=$1
SOL=${2:-/scratch/agustin/projects/autocomp/sols/muon/sol${N}_baseline.cpp}
TAG=${3:-t$N}
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_$TAG
CYC=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron
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
cp "$H/data" "$H/Makefile" "$RK/"

make -C "$RK" kernel.radiance.elf >/dev/null 2>&1 || { echo "COMPILE-FAIL"; exit 1; }

OUT=$(RUST_LOG=error timeout 300 "$CYC/target/release/cyclotron" /scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml \
  --binary-path "$RK/kernel.radiance.elf" --timing --log 0 2>&1) || true
CYCLES=$(echo "$OUT" | grep -oE 'finished after [0-9]+ cycles' | grep -oE '[0-9]+' | tail -1)

if echo "$OUT" | grep -qE "Error: 0$|isa-test passed"; then
  echo "PASS cycles=${CYCLES:-unknown}"
elif echo "$OUT" | grep -q "case="; then
  echo "FAIL errors=$(echo "$OUT" | grep -oE 'case=[0-9]+' | head -1 | cut -d= -f2) cycles=${CYCLES:-unknown}"
else
  echo "SIM-FAIL"
  exit 1
fi
