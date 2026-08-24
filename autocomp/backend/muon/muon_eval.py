"""Evaluation backend for the radiance Muon GPU using the cyclotron simulator.

Pipeline per candidate:
  substitute kernel into the harness -> build kernel.radiance.elf with LLVM-Muon ->
  run cyclotron (functional + timing) -> verdict from tohost ("isa-test passed" /
  "case=<errors>") + latency from "simulation finished after N cycles".
"""

import os
import pathlib
import re
import shutil
import subprocess
from typing import List

# Opt-in: use the cyclotron FUNCTIONAL run (no --timing) as BOTH the correctness gate
# (tohost=0, bit-exact) AND the fitness (dynamic-issue cycle count). Needed for FUSED
# multi-body megakernels (e.g. the fp4 FFN block) that DEADLOCK under cyclotron --timing
# (the documented warp-reconvergence sync in down_body -> Err(0)); their timing model
# never retires, so the normal timed-latency fitness is unavailable. The functional
# issue-count is a coarser cyclotron pre-filter (ranks SIMT-body instruction-count
# micro-opts); RTL remains the arbiter. Set MUON_FUNC_FITNESS=1 to enable.
FUNC_FITNESS = os.getenv("MUON_FUNC_FITNESS") == "1"

from autocomp.common import HARNESSES_DIR, logger
from autocomp.backend.eval_backend import EvalBackend
from autocomp.search.prob import Prob

LLVM_MUON = pathlib.Path("/scratch2/agustin/radiance-kernels/llvm/llvm-muon")
RADIANCE_KERNELS = pathlib.Path("/scratch/agustin/projects/radiance-kernels/kernels")
# Driver for the MX-Gemmini accelerator (MMIO command stream). Staged into the build dir
# for combined Muon+MX problems; see MX_PROBLEM marker below.
MXGEMM_LIB = RADIANCE_KERNELS / "gemm_mxgemmini" / "mxgemm_lib.hpp"
CYCLOTRON_BIN = pathlib.Path(
    "/scratch/agustin/projects/chipyard/generators/radiance/cyclotron/target/release/cyclotron")
CYCLOTRON_CONFIG = pathlib.Path(
    "/scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml")

COMPILE_TIMEOUT = 120
SIM_TIMEOUT = 180  # wall cap; cyclotron self-terminates at config timeout=10M cyc (~90s, ~3x slowest baseline)
FUNC_TIMEOUT = 60  # functional-only correctness gate (no --timing) — fast; catches wrong/illegal/loops

# RTL physical-register budget: the RadianceSingleClusterConfig core has
# numPhysRegs=256 shared across numWarps=8 always-active warps, so a kernel using N
# distinct architectural registers needs 8*N physical registers. If 8*N > 256
# (N > 32) the RTL Rename stage asserts globalOverSubscription and the kernel is
# UNRUNNABLE on hardware. Cyclotron does not model this, so reject over-budget
# kernels here. Budget 31 (x0/zero is mapped and free).
OBJDUMP = LLVM_MUON / "bin" / "llvm-objdump"
MAX_KERNEL_REGS = 31


def _kernel_reg_count(elf_path) -> int:
    """Distinct architectural registers used in kernel_body (RTL-legality proxy)."""
    try:
        dis = subprocess.run([str(OBJDUMP), "-d", str(elf_path)],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1
    regs, in_body = set(), False
    for line in dis.splitlines():
        if "<_ZL11kernel_body" in line:
            in_body = True
            continue
        if in_body:
            if re.search(r"\bret\b", line):
                break
            regs.update(re.findall(r"\b(?:ra|sp|gp|tp|fp|t\d+|s\d+|a\d+)\b", line))
    return len(regs)

# Per-problem harness overhead (cycles): empty-kernel run = data setup + launch +
# verify loop. Subtracted from total simulation cycles so latency reflects only
# the candidate kernel.
EMPTY_KERNEL = "static inline void kernel_body(void*, uint32_t, uint32_t, uint32_t) {}"


def clean_code(code_str: str) -> str:
    """Extract the kernel from an LLM response.

    Responses may contain several fenced blocks plus prose; keep the longest fenced
    block that defines kernel_body. Fallback: longest fenced block, else the
    largest substring starting at the first occurrence of 'kernel_body'.
    """
    code = code_str.strip()
    fences = re.findall(r"```(?:[a-zA-Z+]+)?\n(.*?)```", code, re.DOTALL)
    with_kernel = [f for f in fences if "kernel_body" in f]
    if with_kernel:
        return max(with_kernel, key=len).strip()
    if fences:
        return max(fences, key=len).strip()
    idx = code.find("kernel_body")
    if idx != -1:
        start = code.rfind("\n", 0, max(code.rfind("void", 0, idx), 0))
        return code[max(start, 0):].strip()
    return code


class MuonEvalBackend(EvalBackend):
    def __init__(self, hw_config=None):
        self.hw_config = hw_config
        self._overhead: dict[int, int] = {}

    def harness_overhead(self, prob: Prob) -> int:
        if prob.prob_id not in self._overhead:
            _, cycles, _ = self._run_one(prob, prob.tests[0], EMPTY_KERNEL, 0,
                                         measure_overhead=True)
            self._overhead[prob.prob_id] = cycles or 0
            logger.info("muon eval: harness overhead for prob %s = %s cycles",
                        prob.prob_id, self._overhead[prob.prob_id])
        return self._overhead[prob.prob_id]

    def __repr__(self):
        return "MuonEvalBackend(cyclotron)"

    def get_backend_specific_rules(self) -> list[str]:
        return [
            "The kernel must define void kernel_body(void*, uint32_t, uint32_t, uint32_t); "
            "the harness calls it via mu_schedule.",
            "Use only the buffers in KernelArgs; do not print from the kernel.",
            "Use mu_exp() for exponentials (matches the golden); never call expf().",
            "HARD LIMIT: kernel_body must use <= 31 distinct registers. The target core "
            "has 256 physical registers shared across 8 always-active warps (32/warp), so "
            "a kernel using >32 distinct registers is rejected (unrunnable on RTL). Avoid "
            "deep #pragma unroll and per-thread register tiling (multiple accumulators); "
            "use one accumulator per thread and increment a single pointer. Shared memory "
            "staging is fine and encouraged; the register blowup comes from unroll/tiling.",
        ]

    def evaluate_code(self, prob: Prob, code_strs: List[str], simulator: str) -> List[dict]:
        stats = [{"correct": True, "test_results": {}} for _ in code_strs]
        for test_i, test in enumerate(prob.tests):
            for code_i, code_str in enumerate(code_strs):
                ok, latency, feedback = self._run_one(
                    prob, test, clean_code(code_str), code_i)
                stats[code_i]["test_results"][test_i] = ok
                stats[code_i]["compiled"] = latency is not None or ok
                if not ok:
                    stats[code_i]["correct"] = False
                    # Carry a per-failure diagnostic so the search can REIMPLEMENT with
                    # context (search.py reimplement_failed reads candidate.stderr) instead
                    # of treating every failure as an opaque score=inf dead-end.
                    if feedback:
                        stats[code_i]["stderr"] = feedback
                if latency is not None and ok:
                    stats[code_i]["latency"] = max(latency - self.harness_overhead(prob), 1)
        return stats

    def _run_one(self, prob, test, code_str, code_i, measure_overhead=False):
        """Returns (correct, latency, feedback).

        Two-phase eval: a cheap functional-only run (no --timing) gates correctness +
        RTL register-legality; the expensive timed run is done ONLY for correct candidates
        (to read cycles). Wrong/illegal/looping candidates never pay the timed-sim cost.
        `feedback` is a specific diagnostic string on failure (else None).

        `measure_overhead=True` (used only for the EMPTY_KERNEL harness-overhead baseline)
        skips the correctness gate — the empty kernel intentionally computes no output and
        would 'fail' verification, but we still want its timed cycle count (data setup +
        launch + verify loop) to subtract from real candidates."""
        harness_dir = HARNESSES_DIR / prob.prob_type / f"test{prob.prob_id}"
        work = RADIANCE_KERNELS / f"autocomp_eval_p{prob.prob_id}_c{code_i}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.cpp").write_text(test.modify_test_code(code_str))
        shutil.copy(harness_dir / "data", work / "data")
        shutil.copy(harness_dir / "Makefile", work / "Makefile")

        # Combined Muon + MX-Gemmini problem: stage the accelerator driver, and run
        # cyclotron with its MX co-model enabled. Without the env flag, stores to the
        # Gemmini MMIO block are dead writes and every candidate silently produces zeros.
        is_mx = (harness_dir / "MX_PROBLEM").exists()
        if is_mx:
            shutil.copy(MXGEMM_LIB, work / "mxgemm_lib.hpp")
            host_cpp = harness_dir / "host.cpp"
            if host_cpp.exists():
                shutil.copy(host_cpp, work / "host.cpp")
            # Stage any problem-specific headers shipped with the harness (e.g. the
            # parameterized-operand mxgemm_lib_param.hpp used by fused multi-matmul
            # blocks, which the stock global-operand mxgemm_lib.hpp cannot drive).
            for hpp in harness_dir.glob("*.hpp"):
                shutil.copy(hpp, work / hpp.name)
        # The Makefile has no dependency on `data`/`mxgemm_lib.hpp`, so a stale object
        # would silently survive a golden or driver change.
        for stale in ("kernel.mu.o", "kernel.radiance.elf"):
            (work / stale).unlink(missing_ok=True)

        sim_env = {"RUST_LOG": "error"}
        if is_mx:
            sim_env["CYCLOTRON_MXGEMMINI"] = "1"

        try:
            res = subprocess.run(
                ["make", "kernel.radiance.elf"], cwd=work, capture_output=True,
                text=True, timeout=COMPILE_TIMEOUT,
                env={"PATH": "/usr/bin:/bin", "LLVM_MUON": str(LLVM_MUON)},
            )
        except subprocess.TimeoutExpired:
            logger.info("muon eval: compile timeout (code %d)", code_i)
            return False, None, "COMPILE TIMEOUT: the kernel took too long to compile."
        if res.returncode != 0:
            logger.info("muon eval: compile error (code %d): %s",
                        code_i, res.stderr[-2000:])
            return False, None, f"COMPILE ERROR:\n{res.stderr[-1500:]}"

        # Register-pressure metric (LOGGED, not a hard reject). The static distinct-
        # register count is NON-MONOTONIC with RTL legality (softmax unroll8 @50 regs
        # RAN on RTL while matmul-2x8 @50 asserted globalOverSubscription) — the real
        # constraint is PEAK simultaneously-live regs, which a static scan can't capture.
        # So we no longer reject on it (that wrongly killed RTL-legal kernels); RTL is
        # the register-legality oracle. Logged only as an advisory signal.
        if code_str != EMPTY_KERNEL:
            nregs = _kernel_reg_count(work / "kernel.radiance.elf")
            if nregs > MAX_KERNEL_REGS:
                logger.info("muon eval: high reg count (code %d): %d distinct regs "
                            "(advisory; not rejected — static count can't predict RTL "
                            "globalOverSubscription)", code_i, nregs)

        elf = str(work / "kernel.radiance.elf")
        base_cmd = [str(CYCLOTRON_BIN), str(CYCLOTRON_CONFIG), "--binary-path", elf, "--log", "0"]

        # Phase 1: functional-only correctness gate (no --timing). Fast (no timing model);
        # produces the same "isa-test passed"/"case=N" verdict and the same RTL register
        # panic. Wrong/illegal/looping candidates are rejected here without the timed cost.
        # (Skipped for the EMPTY_KERNEL overhead baseline, which has no correct output.)
        if not measure_overhead:
            try:
                func = subprocess.run(
                    base_cmd, cwd=work, capture_output=True, text=True,
                    timeout=FUNC_TIMEOUT, env=sim_env,
                )
            except subprocess.TimeoutExpired:
                logger.info("muon eval: functional timeout (code %d)", code_i)
                return False, None, ("FUNCTIONAL TIMEOUT: kernel did not terminate (hit the 10M-cycle "
                                     "cap). Likely an unbounded loop or far too much work per thread.")
            fout = func.stdout + func.stderr
            if "globalOverSubscription" in fout:
                logger.info("muon eval: RTL-ILLEGAL register oversubscription (code %d) — "
                            "rejected (would $fatal on RadianceSingleClusterConfig)", code_i)
                return False, None, ("RTL-ILLEGAL: register oversubscription (>256 distinct regs across "
                                     "8 warps). Cut outputs-per-thread / accumulators / #pragma unroll; "
                                     "use one accumulator and a single incrementing pointer.")
            if "isa-test passed" not in fout:
                mfail = re.search(r"case=(\d+)", fout)
                errs = mfail.group(1) if mfail else "?"
                logger.info("muon eval: FAIL (code %d): %s errors", code_i, errs)
                return False, None, (f"INCORRECT: {errs} lane(s) wrong vs golden. Check the cross-core "
                                     "barrier (mu_barrier(0, total_warps) AFTER the schedule), per-phase "
                                     "barriers (mu_barrier(1, nw)) between SMEM produce/consume, and the "
                                     "thread->output mapping derived from threads_per_threadblock (not a "
                                     "literal warp count). Gold accumulates sequentially (FP order).")
            # FUNCTIONAL-FITNESS mode: the fused megakernel deadlocks under --timing, so use
            # the functional-run cycle count (dynamic issue count) as the fitness and SKIP the
            # timed phase. Correctness already gated above (tohost=0, bit-exact).
            if FUNC_FITNESS:
                mfc = re.findall(r"finished after (\d+) cycles", fout)
                cyc = int(mfc[-1]) if mfc else None
                if cyc is None:
                    return False, None, "FUNC-FITNESS: no cycle count in the functional run."
                return True, cyc, None

        # Phase 2: timed run — AUTHORITATIVE for both correctness and cycles. The functional
        # gate above is only a fast pre-filter; some failures are TIMING-DEPENDENT and pass
        # functionally but fail here — e.g. a cross-core SMEM race (mu_barrier ID 1 = per-core
        # vs ID 0 = cross-core) or a host-verify/write-drain race. Functional SMEM is instantly
        # coherent, so it can't see these; the timed model can. Must re-check correctness here.
        try:
            sim = subprocess.run(
                base_cmd + ["--timing"], cwd=work, capture_output=True, text=True,
                timeout=SIM_TIMEOUT, env=sim_env,
            )
        except subprocess.TimeoutExpired:
            logger.info("muon eval: timed-sim timeout (code %d) — correct but too slow", code_i)
            return False, None, ("TOO SLOW: correct functionally, but the timed sim exceeded the wall "
                                 "cap (far slower than baseline). Likely 16-way SMEM bank-conflict "
                                 "serialization (pad column stride +16 floats) or excessive global "
                                 "memory traffic. Reduce serialization; keep the algorithm.")
        out = sim.stdout + sim.stderr
        if measure_overhead:
            # EMPTY_KERNEL has no correct output; we only want its cycle count (data setup +
            # launch + verify loop) as the per-problem overhead baseline. Skip correctness.
            m = re.findall(r"finished after (\d+) cycles", out)
            return True, (int(m[-1]) if m else None), None
        if "isa-test passed" not in out:
            mfail = re.search(r"case=(\d+)", out)
            errs = mfail.group(1) if mfail else "?"
            logger.info("muon eval: TIMING-FAIL (code %d): %s errors under --timing "
                        "(passed functionally)", code_i, errs)
            return False, None, (f"INCORRECT UNDER TIMING: {errs} lane(s) wrong with the timing model "
                                 "(but correct functionally) => a TIMING-DEPENDENT race. Use "
                                 "mu_barrier(0, total_warps) (ID 0 = CROSS-CORE) — not ID 1 (per-core) "
                                 "— after staging into SMEM that other cores read, and mu_fence_smem() "
                                 "before the cross-core barrier. Ensure all threadblock stores are "
                                 "visible before any read across cores.")
        m = re.findall(r"finished after (\d+) cycles", out)
        cycles = int(m[-1]) if m else None
        if cycles is None:
            return False, None, "TIMED-SIM ERROR: no cycle count produced."
        return True, cycles, None
