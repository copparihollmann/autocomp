"""Evaluation backend for the radiance Muon GPU using the cyclotron simulator.

Pipeline per candidate:
  substitute kernel into the harness -> build kernel.radiance.elf with LLVM-Muon ->
  run cyclotron (functional + timing) -> verdict from tohost ("isa-test passed" /
  "case=<errors>") + latency from "simulation finished after N cycles".
"""

import pathlib
import re
import shutil
import subprocess
from typing import List

from autocomp.common import HARNESSES_DIR, logger
from autocomp.backend.eval_backend import EvalBackend
from autocomp.search.prob import Prob

LLVM_MUON = pathlib.Path("/scratch2/agustin/radiance-kernels/llvm/llvm-muon")
RADIANCE_KERNELS = pathlib.Path("/scratch/agustin/projects/radiance-kernels/kernels")
CYCLOTRON_BIN = pathlib.Path(
    "/scratch/agustin/projects/chipyard/generators/radiance/cyclotron/target/release/cyclotron")
CYCLOTRON_CONFIG = pathlib.Path(
    "/scratch/agustin/projects/autocomp/scripts/muon/config_muon.toml")

COMPILE_TIMEOUT = 120
SIM_TIMEOUT = 700

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
            _, cycles = self._run_one(prob, prob.tests[0], EMPTY_KERNEL, 0)
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
                ok, latency = self._run_one(prob, test, clean_code(code_str), code_i)
                stats[code_i]["test_results"][test_i] = ok
                stats[code_i]["compiled"] = latency is not None or ok
                if not ok:
                    stats[code_i]["correct"] = False
                if latency is not None and ok:
                    stats[code_i]["latency"] = max(latency - self.harness_overhead(prob), 1)
        return stats

    def _run_one(self, prob, test, code_str, code_i):
        """Returns (correct, latency)."""
        harness_dir = HARNESSES_DIR / prob.prob_type / f"test{prob.prob_id}"
        work = RADIANCE_KERNELS / f"autocomp_eval_p{prob.prob_id}_c{code_i}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.cpp").write_text(test.modify_test_code(code_str))
        shutil.copy(harness_dir / "data", work / "data")
        shutil.copy(harness_dir / "Makefile", work / "Makefile")

        try:
            res = subprocess.run(
                ["make", "kernel.radiance.elf"], cwd=work, capture_output=True,
                text=True, timeout=COMPILE_TIMEOUT,
                env={"PATH": "/usr/bin:/bin", "LLVM_MUON": str(LLVM_MUON)},
            )
        except subprocess.TimeoutExpired:
            logger.info("muon eval: compile timeout (code %d)", code_i)
            return False, None
        if res.returncode != 0:
            logger.info("muon eval: compile error (code %d): %s",
                        code_i, res.stderr[-2000:])
            return False, None

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

        try:
            sim = subprocess.run(
                [str(CYCLOTRON_BIN), str(CYCLOTRON_CONFIG),
                 "--binary-path", str(work / "kernel.radiance.elf"),
                 "--timing", "--log", "0"],
                cwd=work, capture_output=True, text=True, timeout=SIM_TIMEOUT,
                env={"RUST_LOG": "error"},
            )
        except subprocess.TimeoutExpired:
            logger.info("muon eval: simulation timeout (code %d)", code_i)
            return False, None

        out = sim.stdout + sim.stderr
        # Cyclotron now models the RTL Rename physical-register pool at runtime
        # (MuonCore::track_register_pressure). An RTL-illegal kernel panics with
        # globalOverSubscription exactly as Rename.scala:123 asserts — reject it as a
        # failed candidate so the search never proposes register-tiled kernels that
        # would $fatal on hardware.
        if "globalOverSubscription" in out:
            logger.info("muon eval: RTL-ILLEGAL register oversubscription (code %d) — "
                        "rejected (would $fatal on RadianceSingleClusterConfig)", code_i)
            return False, None
        passed = "isa-test passed" in out
        cycles = None
        m = re.findall(r"finished after (\d+) cycles", out)
        if m:
            cycles = int(m[-1])
        if not passed:
            mfail = re.search(r"case=(\d+)", out)
            logger.info("muon eval: FAIL (code %d): %s", code_i,
                        f"{mfail.group(1)} errors" if mfail else out[-300:])
        return passed, cycles
