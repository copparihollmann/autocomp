#!/usr/bin/env bash
# Verify the COHERENCE invariant: cyclotron co-model == mx_golden == upstream golden == spike,
# and (when a tapeout simv exists) == RTL. See autocomp/COHERENCE.md.
#
# Run this after ANY version bump (cyclotron, mxgemmini, radiance-kernels, goldens). If it is
# green, the stack is coherent and speedup numbers mean something. If it is red, stop.
#
#   check_coherence.sh            # levels 1-3 (fast, no RTL)
#   check_coherence.sh --rtl      # also level 4 (RTL gate; needs a VCS licence)
set -uo pipefail
AC=/scratch/agustin/projects/autocomp
CYC_DIR=/scratch/agustin/projects/chipyard/generators/radiance/cyclotron
CYC=$CYC_DIR/target/release/cyclotron
PROBS="20 22 23 24"
FAIL=0
ok()   { printf "  \033[32mOK\033[0m   %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAIL=1; }

echo "== version tuple =="
printf "   %-18s %s\n" radiance   "$(git -C /scratch/agustin/projects/chipyard/generators/radiance describe --tags 2>/dev/null)"
printf "   %-18s %s\n" gemmini    "$(git -C /scratch/agustin/projects/chipyard/generators/gemmini describe --tags 2>/dev/null)"
printf "   %-18s %s\n" cyclotron  "$(git -C "$CYC_DIR" rev-parse --short HEAD)"
printf "   %-18s %s\n" mxgemmini  "$(git -C /scratch/agustin/projects/radiance-kernels/lib/mxgemmini rev-parse --short HEAD)"
BR=$(grep -oE '#define BANK_ROWS [0-9]+' /scratch/agustin/projects/radiance-kernels/lib/mxgemmini/include/gemmini_params.h | grep -oE '[0-9]+')
[ "$BR" = "2048" ] && ok "BANK_ROWS=2048 (tapeout 128KiB SMEM)" || bad "BANK_ROWS=$BR -- MUST be 2048; the B operand will be out of range"

echo "== [1] co-model internals vs spike (unit tests) =="
# NB: capture first, then grep. `cargo test | grep -q` under `set -o pipefail` reports failure
# even on a match, because grep -q exits early and SIGPIPEs cargo.
TESTOUT=$(cd "$CYC_DIR" && cargo test --release mxgemmini 2>&1)
NPASS=$(echo "$TESTOUT" | grep -oE "^test result: ok\. [0-9]+ passed" | grep -oE "[0-9]+" | head -1)
if echo "$TESTOUT" | grep -q "^test result: FAILED" || [ -z "${NPASS:-}" ] || [ "${NPASS:-0}" -lt 30 ]; then
  bad "cargo test mxgemmini (passed=${NPASS:-none})"
else ok "cargo test mxgemmini ($NPASS tests)"; fi

echo "== [2] kernels vs golden (end-to-end, bit-exact) =="
for N in $PROBS; do
  R=$(bash "$AC/scripts/muon/run_problem_mx.sh" "$N" 2>&1 | tail -1)
  case "$R" in PASS*) ok "prob$N $R";; *) bad "prob$N $R";; esac
done

echo "== [2b] NEGATIVE CONTROL (co-model OFF must FAIL -- a gate that cannot fail proves nothing) =="
ELF=/scratch/agustin/projects/radiance-kernels/kernels/autocomp_t20/kernel.radiance.elf
if [ -f "$ELF" ]; then
  OUT=$(RUST_LOG=error timeout 300 "$CYC" "$AC/scripts/muon/config_muon.toml" \
        --binary-path "$ELF" --timing --log 0 2>&1)   # note: no CYCLOTRON_MXGEMMINI=1
  if echo "$OUT" | grep -qE "case=[0-9]+"; then ok "co-model disabled -> gate correctly FAILS"
  else bad "co-model disabled but the gate still PASSED -- the check is vacuous!"; fi
else bad "prob20 elf missing (run run_problem_mx.sh 20 first)"; fi

echo "== [3] our mx_golden vs the upstream golden models =="
for H in test20 test23 test22; do
  R=$("$AC/.venv/bin/python" "$AC/scripts/muon/xcheck_upstream_golden.py" "$AC/harnesses/muon/$H" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ bit-exact")
  case "$R" in "") bad "$H xcheck did not run (torch/qtorch installed?)";;
    *) A=${R%%/*}; B=${R#*/}; B=${B%% *}; [ "$A" = "$B" ] && ok "$H $R" || bad "$H $R";; esac
done

if [ "${1:-}" = "--rtl" ]; then
  echo "== [4] RTL gate (tapeout-330 silicon) =="
  for N in $PROBS; do
    R=$(bash "$AC/scripts/muon/vcs_gate_mx.sh" "$N" 2>&1 | tail -1)
    case "$R" in PASS*) ok "prob$N RTL $R";; *) bad "prob$N RTL $R";; esac
  done
else
  echo "== [4] RTL gate: SKIPPED (pass --rtl) =="
fi

echo
[ $FAIL -eq 0 ] && echo -e "\033[32mCOHERENT\033[0m -- the stack agrees; numbers are meaningful." \
                || echo -e "\033[31mINCOHERENT\033[0m -- do NOT trust results until this is green."
exit $FAIL
