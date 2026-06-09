#!/usr/bin/env bash
# Static RTL register-legality pre-check for a Muon kernel candidate.
# The RadianceSingleClusterConfig core has numPhysRegs=256 shared across
# numWarps=8 ALWAYS-active warps -> a kernel may use at most 256/8 = 32 distinct
# architectural registers per warp, else the Rename stage asserts
# globalOverSubscription and the RTL sim $fatals. Cyclotron does NOT model this,
# so check here BEFORE spending a ~13-min VCS run.
#   reg_check.sh <prob_id> <candidate.cpp>
# Exit 0 + "LEGAL" if <= 31 distinct regs; exit 1 + "ILLEGAL" otherwise.
set -u
N=$1
SOL=$2
BUDGET=31   # 31 usable (x0/zero is mapped, free); 32 distinct -> 256 exactly -> overflow
H=/scratch/agustin/projects/autocomp/harnesses/muon/test$N
RK=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_regchk
OBJD=/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump
export LLVM_MUON=${LLVM_MUON:-/scratch2/agustin/radiance-kernels/llvm/llvm-muon}

mkdir -p "$RK"
[ -f "$H/data" ] || (cd "$H" && python3 gen_data.py >/dev/null)
python3 - "$H/test$N.cpp" "$SOL" "$RK/kernel.cpp" <<'PY'
import sys
h, s, out = sys.argv[1:4]
code = open(s).read(); text = open(h).read()
a = text.index("// SUBSTITUTE HERE") + len("// SUBSTITUTE HERE")
b = text.index("// SUBSTITUTE END")
open(out, "w").write(text[:a] + "\n" + code + "\n" + text[b:])
PY
cp "$H/data" "$H/Makefile" "$RK/"
make -C "$RK" kernel.radiance.elf >/dev/null 2>&1 || { echo "COMPILE-FAIL"; exit 1; }

COUNT=$("$OBJD" -d "$RK/kernel.radiance.elf" 2>/dev/null | \
  awk '/<_ZL11kernel_body/{f=1} f&&/[ \t]ret/{print; f=0; next} f{print}' | \
  grep -oE '\b(ra|sp|gp|tp|fp|t[0-9]+|s[0-9]+|a[0-9]+)\b' | sort -u | wc -l)

if [ "$COUNT" -le "$BUDGET" ]; then
  echo "LEGAL regs=$COUNT/$BUDGET (8 warps x $COUNT = $((COUNT*8)) <= 256)"
else
  echo "ILLEGAL regs=$COUNT/$BUDGET (8 warps x $COUNT = $((COUNT*8)) > 256 -> Rename globalOverSubscription)"
  exit 1
fi
