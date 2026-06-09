import os
import re
import subprocess
import pathlib
import multiprocessing
from typing import List
import time
import glob
import shutil

from autocomp.common import logger, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend
from autocomp.hw_config.gemmini_config import GemminiHardwareConfig

FP32_4PE_CHIPYARD_PATH = None
INT8_16PE_CHIPYARD_PATH = "/scratch/agustin/projects/chipyard-mx"  # MX-Gemmini (dim=16) checkout
INT8_32PE_CHIPYARD_PATH = None

# MX-Gemmini mode: the MX spike functional model requires binaries built with
# -DSPIKE_SIM, which gemmini-rocc-tests produces via build_spike.sh into
# build_spike/ (vs build.sh -> build/ for the legacy int8/fp32 flow). Toggle
# with the GEMMINI_MX env var (default on for this checkout).
MX_GEMMINI = os.environ.get("GEMMINI_MX", "1") != "0"
_BUILD_SCRIPT = "./build_spike.sh" if MX_GEMMINI else "./build.sh"
_BUILD_DIR = "build_spike" if MX_GEMMINI else "build"

def clean_code(code_str: str) -> str:
    """
    Takes LLM-generated code, removes the "solution" wrapper function, and cleans up anything that is not runnable C code

    for example:
    '''
    void solution(Kinf, r, K_r) {
        config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
        config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
        ...
        mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
        fence();
    }
    '''

    becomes 

    '''
    config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
    config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
    ...
    mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
    fence();
    '''
    """
    # Remove the function wrapper and return only the body of the function
    if not code_str:
        return ""
    after_void_solution_str = code_str[code_str.find("void solution("):]
    start = after_void_solution_str.find('{') + 1
    end = after_void_solution_str.rfind('}')
    body = after_void_solution_str[start:end]
    return body
    # body = code_str[start:end].split("\n")
    # new_body = []
    # for line in body:
    #     new_body.append(line.strip())
    # return "\n".join(new_body)

def compile_gemmini_code(code_contents: str, gemmini_path: pathlib.Path):
    """
    Compile the provided code to a baremetal binary.
    Returns the path to the compiled binary.
    """
    test_name = "auto_comp_test"
    # if test_name_str:
    #     test_name += "_" + str(test_name_str)
    gemmini_sw_path = gemmini_path / "software" / "gemmini-rocc-tests"
    # copy in file
    with open(gemmini_sw_path / "bareMetalC" / (test_name + ".c"), "w") as f:
        f.write(code_contents)

    # # edit Makefile
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'r') as file:
    #     filedata = file.read()
    # if test_name not in filedata:
    #     filedata = filedata.replace("tests = .*\n", "tests = "+test_name+"\n")
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'w') as file:
    #     filedata = file.write(filedata)
    
    # build software
    p = subprocess.run(["sh", _BUILD_SCRIPT], cwd=gemmini_sw_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        return

    return gemmini_sw_path / _BUILD_DIR / "bareMetalC" / (test_name + "-baremetal")

_HOST_FLOAT_RE = re.compile(
    r"\bfloat\b|\bdouble\b|\bexp_f\b|\bf_exp\b|\bpow2i\b|\bbf16_to_f\b|\bfp8_e4m3_rtz\b|\bf_abs\b|\bexpf\b|\bsqrtf?\b|\blogf?\b",
)
def _count_host_float_ops(code_str: str) -> int:
    """Count host floating-point indicators in a candidate kernel body. The RadianceGemminiOnlyConfig
    host Rocket has NO FPU, so ANY of these traps on real RTL (spike hides it). Used by MX_NOFPU_COST
    to penalize non-deployable (float-using) kernels so the optimizer prefers the float-free integer
    softmax. Strip // and /* */ comments first so commentary about "float" doesn't count."""
    s = re.sub(r"//[^\n]*", "", code_str)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return len(_HOST_FLOAT_RE.findall(s))


def run_spike_mp(code_contents_lst: list[str], gemmini_path: pathlib.Path, timeout: float=120):
    """Evaluate candidate kernels on spike SEQUENTIALLY.

    All candidates share a single bareMetalC/auto_comp_test.c source and a single
    build_spike/.../auto_comp_test-baremetal binary, so they must NOT be built/run
    concurrently — doing so makes them clobber each other's source/binary and
    produces wrong or cross-contaminated results. Each candidate is therefore
    written, built, and run in turn. `timeout` is a PER-CANDIDATE wall-clock bound
    on the spike run, so a kernel that hangs spike is killed and reported as
    "Timeout" without stalling the whole batch.
    """
    results = []
    for code_i, code_contents in enumerate(code_contents_lst):
        return_dict: dict = {}
        run_spike(code_contents, return_dict, gemmini_path, str(code_i), timeout)
        results.append(return_dict.get("retval", "Timeout"))
    return results
    
def run_spike(code_contents: str, return_dict: dict, gemmini_path: pathlib.Path, test_name_str: str, timeout: float):
    test_name = "auto_comp_test"
    # if test_name_str:
    #     test_name += "_" + str(test_name_str)
    gemmini_sw_path = gemmini_path / "software" / "gemmini-rocc-tests"
    # copy in file
    with open(gemmini_sw_path / "bareMetalC" / (test_name + ".c"), "w") as f:
        f.write(code_contents)

    # # edit Makefile
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'r') as file:
    #     filedata = file.read()
    # if test_name not in filedata:
    #     filedata = filedata.replace("tests = .*\n", "tests = "+test_name+"\n")
    # with open(gemmini_sw_path / 'bareMetalC' / 'Makefile', 'w') as file:
    #     filedata = file.write(filedata)
    
    # build software (baremetal only — spike doesn't need the linux/pk variants)
    p = subprocess.run(["sh", _BUILD_SCRIPT, "BAREMETAL_ONLY=1"], cwd=gemmini_sw_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        # return_dict["retval"] = p.stdout.decode()
        return_dict["retval"] = "Compile error"
        return

    try:
        p = subprocess.run(["stdbuf", "-oL", "spike", "--extension=gemmini", f"./{_BUILD_DIR}/bareMetalC/auto_comp_test-baremetal"], cwd=gemmini_sw_path,
                             capture_output=True, text=True, errors="ignore", timeout=timeout)
        spike_output = p.stdout
    except subprocess.TimeoutExpired:
        logger.info("spike ran for more than %s seconds, terminating.", timeout)
        return_dict["retval"] = "Timeout"
        return

    # # run with timeout
    # with subprocess.Popen(["stdbuf", "-oL", "sh", "./scripts/run-spike.sh", test_name], cwd=gemmini_path, 
    #                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore") as process:
    #     try:
    #         spike_output = process.communicate(timeout=timeout)[0]
    #     except subprocess.TimeoutExpired:
    #         os.killpg(process.pid, signal.SIGINT) # send signal to the process group
    #         logger.info("spike ran for more than %d seconds, terminating.", timeout)
    #         return_dict["retval"] = "Timeout"
    #         return

    # start = time.time()
    # while time.time() - start <= timeout:
    #     if p.poll() is not None:
    #         # Process is done, break now.
    #         break
    #     time.sleep(1)  # Just to avoid hogging the CPU
    # else:
    #     # We only enter this if we didn't 'break' above.
    #     logger.info("spike ran for more than %d seconds, terminating.", timeout)
    #     p.kill()
    #     return_dict["retval"] = "Timeout"
    #     return
    # spike_output = p.stdout

    # try:
    #     p = subprocess.run(["sh", "./scripts/run-spike.sh", test_name], cwd=GEMMINI_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    # except subprocess.TimeoutExpired:
    #     return "Timeout"
    # if p.returncode != 0:
    #     return str(p.stdout)

    # Python-side correctness check for harnesses that dump their output and
    # reference a stored gold file ("// GOLD_FILE: <path>"). The harness prints
    # the dump plus "Correct result"; we verify the dump and patch the verdict.
    if "GOLD_BEGIN" in spike_output and "// GOLD_FILE:" in code_contents:
        gold_path = code_contents.split("// GOLD_FILE:", 1)[1].splitlines()[0].strip()
        try:
            golden = pathlib.Path(gold_path).read_text().split()
            dumped = re.findall(r"\b[0-9a-fA-F]{8}\b",
                                spike_output.split("GOLD_BEGIN", 1)[1].split("GOLD_END", 1)[0])
            head = spike_output.split("GOLD_BEGIN", 1)[0]
            tail = spike_output.split("GOLD_END", 1)[1] if "GOLD_END" in spike_output else ""
            logger.info("GOLD check: dumped=%d golden=%d eq=%s tail_has=%s",
                        len(dumped), len(golden), dumped == golden, "Correct result" in tail)
            if dumped == golden and "Correct result" in tail:
                spike_output = head + tail
            else:
                spike_output = head + tail.replace("Correct result", "Incorrect result")
        except Exception as e:
            logger.warning("GOLD check failed: %r", e)
            spike_output = spike_output.replace("Correct result", "Incorrect result")

    # return output
    return_dict["retval"] = spike_output

def run_firesim(code_contents_lst: list[str], gemmini_path: pathlib.Path, firesim_path: pathlib.Path, timeout: float=60):
    results_dir = firesim_path / "deploy" / "results-workload"
    results = []

    # # Check correctness first with spike
    # spike_results = run_spike_mp(code_contents_lst, gemmini_path, timeout=20)

    # Launch FPGA runfarm, if on AWS
    # subprocess.run(["firesim", "launchrunfarm"])
    # target_config = "firesim_rocket_singlecore_gemmini_no_nic_l2_llc4mb_ddr3_v2"
    # Run each code content in sequence
    for i in range(len(code_contents_lst)):
        # if "Correct result" not in spike_results[i]:
        #     logger.info("Skipping FireSim run for code %d due to incorrect spike result.", i)
        #     results.append("")
        #     continue

        # Compile the code
        orig_binary_path = compile_gemmini_code(code_contents_lst[i], gemmini_path)
        if not orig_binary_path:
            logger.info("Failed to compile code.")
            results.append("")
            continue
        firesim_gemmini_workload_path = firesim_path / "deploy" / "workloads" / "gemmini"
        firesim_gemmini_workload_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(orig_binary_path, firesim_gemmini_workload_path / "auto_comp_test-baremetal")

        # Get the current directories under results_dir
        orig_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

        # config_runtime_path = firesim_path / "deploy" / "config_runtime.yaml"
        # with open(config_runtime_path, "r") as f:
        #     config = yaml.safe_load(f)
        # config["workload"]["workload_name"] = "auto_comp_test-baremetal"
        # config["target_config"]["default_hw_config"] = target_config
        # with open(config_runtime_path, "w") as f:
        #     yaml.dump(config, f)
        logger.info("Running `firesim infrasetup`")
        p = subprocess.Popen([str(firesim_path / "deploy" / "firesim"), "infrasetup"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        with p.stdout:
            for line in p.stdout:
                logger.debug(line.decode().strip())
        p.wait()
        logger.info("Running `firesim runworkload`")
        p = subprocess.Popen([str(firesim_path / "deploy" / "firesim"), "runworkload"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # with p.stdout:
        #     for line in p.stdout:
        #         logger.debug(line.decode().strip())
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait()
            logger.warning("Firesim runworkload timed out.")
            logger.info("Running `firesim kill`")
            subprocess.run([str(firesim_path / "deploy" / "firesim"), "kill"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
            results.append("")
            continue
        logger.info("Firesim runworkload finished, terminate now if you need to")
        time.sleep(2)

        # Terminate FPGA runfarm, if on AWS
        # p = subprocess.Popen(["firesim", "terminaterunfarm"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # p.communicate("yes\n".encode())

        # See if there are any new directories under results_dir
        new_dirs = list()
        current_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for dir in current_results_dirs:
            if dir not in orig_results_dirs:
                new_dirs.append(dir)
        if not new_dirs:
            logger.error("No new results directory found after running FireSim.")
            results.append("")
            continue

        relevant_logs = []
        for dir in new_dirs:
            dir_logs = glob.glob(f"{dir}/*/uartlog", recursive=True)
            for log in dir_logs:
                relevant_logs.append(log)
        if len(relevant_logs) > 1:
            logger.warning("Multiple new logs found, using the first one. Logs found: %s", str(relevant_logs))
        if len(relevant_logs) == 0:
            logger.error("No relevant logs found after running FireSim.")
            results.append("")
            continue
        logger.info("Parsing FireSim results from: %s", str(relevant_logs[0]))

        result_string = pathlib.Path(relevant_logs[0]).read_text()
        results.append(result_string)

    return results

def mx_cycle_phase_hint(code_str: str) -> list:
    """Static cycle-phase bottleneck hint for an MX kernel candidate (marker-free).

    autocomp only optimizes accelerator code, so the actionable signal is WHICH phase
    dominates: CPU host staging (memcpy / nested scalar copy loops), accelerator data
    movement (mvin), or compute (loop_ws_spad). On the MX kernels the dominant cost is
    usually host staging, so flag it loudly and point at the fix (strided mvin from DRAM)."""
    import re as _re
    src = code_str
    mvin = len(_re.findall(r'gemmini_(?:extended_)?mvin\w*\s*\(', src))
    compute = len(_re.findall(r'gemmini_(?:loop_ws_spad|compute_\w+|mx_read_smem)\s*\(', src))
    memcpy = len(_re.findall(r'\b(?:memcpy|memmove)\s*\(', src))
    # nested scalar-copy loops that stage data on the host (A_buf[..]=.., O_acc[..]+=.. etc.)
    stage_loops = len(_re.findall(r'\bfor\s*\([^;]*;[^;]*;[^)]*\)\s*(?:\{[^}]*)?[A-Za-z_]\w*\s*\[[^\]]*\]\s*[-+]?=', src))
    hints = []
    if memcpy or stage_loops >= 3:
        hints.append(
            f"CPU/host data staging detected ({memcpy} memcpy, ~{stage_loops} scalar copy loops). "
            "Host staging is typically the dominant cost on this accelerator and autocomp cannot "
            "speed it up directly: mvin tiles STRIDED straight from the source DRAM arrays instead "
            "of copying through a host buffer, and keep partial sums in shared memory (bf16 accumulate) "
            "rather than accumulating on the CPU.")
    if mvin and compute and mvin > 4 * compute:
        hints.append(
            f"Data movement appears to dominate ({mvin} mvin vs {compute} compute launches). "
            "Raise operand reuse: load each operand tile once and reuse across the inner loops, "
            "or fuse tiles so more compute happens per mvin.")
    if not hints:
        hints.append("Accelerator-bound: compute and data-movement look balanced; tune tiling / "
                     "scratchpad occupancy for more overlap.")
    return hints

def parse_spad_acc_utilization(spike_output: str, pe_dim: int, spad_size_kb: int, acc_size_kb: int) -> int:
    """
    Parse the output of the spike simulator to extract the spad utilization.
    """
    # Find the line that contains the spad utilization
    spad_addresses_used = set()
    acc_addresses_used = set()
    for line in spike_output.split("\n"):
        if "GEMMINI: mvin - " in line or "GEMMINI: mvout - " in line:
            if "mvin" in line:
                first_split = line.split("GEMMINI: mvin - ")[1]
            elif "mvout" in line:
                first_split = line.split("GEMMINI: mvout - ")[1]
            col_split = first_split.split(" cols and ")
            cols = int(col_split[0].strip(), 0) # actually just ignore this
            row_split = col_split[1].split(" rows from ")
            try:
                rows = int(row_split[0].strip(), 0)
            except:
                import pdb
                pdb.set_trace()
            addr_split = row_split[1].split(" to addr ")
            first_addr = int(addr_split[0].strip(), 0)
            second_addr = int(addr_split[1].strip(), 0)
            if "mvin" in line:
                spad_acc_addr = second_addr
            elif "mvout" in line:
                spad_acc_addr = first_addr
            accumulator = bool(spad_acc_addr >> 31)
            actual_addr = spad_acc_addr & 0x3FFFFFF
            for j in range(cols):
                for i in range(rows):
                    row_addr = actual_addr + pe_dim * (j // pe_dim * pe_dim) + i
                    if accumulator:
                        acc_addresses_used.add(row_addr)
                    else:
                        spad_addresses_used.add(row_addr)
    num_spad_rows = (spad_size_kb * 1024 / (pe_dim * 1))
    num_acc_rows = (acc_size_kb * 1024 / (pe_dim * 4))
    return len(spad_addresses_used) / num_spad_rows, len(acc_addresses_used) / num_acc_rows

class GemminiEvalBackend(EvalBackend):
    def __init__(self, hw_config: GemminiHardwareConfig):
        self.hw_config = hw_config
        self.pe_dim = hw_config.pe_dim
        self.spad_size_kb = hw_config.spad_size_kb
        self.acc_size_kb = hw_config.acc_size_kb
        pe_dim = self.pe_dim
        if pe_dim == 4:
            if not FP32_4PE_CHIPYARD_PATH:
                raise ValueError("FP32_4PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = pathlib.Path(FP32_4PE_CHIPYARD_PATH).resolve() / "generators" / "gemmini"
            self.firesim_path = pathlib.Path(FP32_4PE_CHIPYARD_PATH).resolve() / "sims" / "firesim"
        elif pe_dim == 16:
            if not INT8_16PE_CHIPYARD_PATH:
                raise ValueError("INT8_16PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = pathlib.Path(INT8_16PE_CHIPYARD_PATH).resolve() / "generators" / "gemmini"
            self.firesim_path = pathlib.Path(INT8_16PE_CHIPYARD_PATH).resolve() / "sims" / "firesim"
        elif pe_dim == 32:
            if not INT8_32PE_CHIPYARD_PATH:
                raise ValueError("INT8_32PE_CHIPYARD_PATH not set at top of hardware_eval.py")
            self.gemmini_path = pathlib.Path(INT8_32PE_CHIPYARD_PATH).resolve() / "generators" / "gemmini"
            self.firesim_path = pathlib.Path(INT8_32PE_CHIPYARD_PATH).resolve() / "sims" / "firesim"
        else:
            raise ValueError("supported Gemmini pe_dims: {4, 16, 32}")
        self.gemmini_sw_path = self.gemmini_path / "software" / "gemmini-rocc-tests"

    def __repr__(self):
        return f"GemminiEvalBackend({self.pe_dim})"

    def get_hw_feedback(self, prob: Prob, code_strs: list[str]) -> list[list[str]]:
        """Per-implementation feedback: spad/acc utilization + a static cycle-phase hint.

        The cycle-phase hint (mx_cycle_phase_hint) reads the candidate's structure to flag
        the dominant bottleneck class on the MX accelerator — CPU host-side data staging vs
        accelerator data movement (mvin) vs compute — since autocomp only optimizes the
        accelerator code and the biggest MX wins come from cutting host staging (the tiled
        GEMM baselines were ~98.8% CPU staging, profiled via mx_pipeline/profile.py)."""
        stats_list = self.get_spad_acc_utilization(prob, code_strs)
        feedback_per_impl = []
        for stats, code_str in zip(stats_list, code_strs):
            spad_cap_used = round(stats['spad_util'] * self.spad_size_kb)
            acc_cap_used = round(stats['acc_util'] * self.acc_size_kb)
            feedback = [
                f"Scratchpad utilization is {spad_cap_used}KB out of {self.spad_size_kb}KB.",
                f"Accumulator utilization is {acc_cap_used}KB out of {self.acc_size_kb}KB."
            ]
            if stats['spad_util'] < 1:
                feedback[0] += " Consider increasing scratchpad utilization to improve performance."
            if stats['acc_util'] < 1:
                feedback[1] += " Consider increasing accumulator utilization to improve performance."
            feedback.extend(mx_cycle_phase_hint(code_str))
            feedback_per_impl.append(feedback)
        return feedback_per_impl

    def evaluate_code_parallel_spike(self, prob: Prob, code_strs: list[str]) -> float:
        return self.evaluate_code(prob, code_strs, "spike")

    def evaluate_code_parallel_firesim(self, prob: Prob, code_strs: list[str]) -> float:
        return self.evaluate_code(prob, code_strs, "firesim")

    def get_spad_acc_utilization(self, prob: Prob, code_strs: list[str]) -> int:
        stats = [{
            "spad_util": 0,
            "acc_util": 0
        } for _ in code_strs]
        clean_code_strs = [clean_code(code_str) for code_str in code_strs]
        for test_i, test in enumerate(prob.tests):
            test_output_per_code_str = run_spike_mp([test.get_test_code([code_str], check_correct=False, repeat_iters=1) for code_str in clean_code_strs], self.gemmini_path, timeout=60)
            for code_i, test_output in enumerate(test_output_per_code_str):
                spad_util, acc_util = parse_spad_acc_utilization(test_output, self.pe_dim, self.spad_size_kb, self.acc_size_kb)
                stats[code_i]["spad_util"] = spad_util
                stats[code_i]["acc_util"] = acc_util
        return stats

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate the code based on the provided optimization metric.
        Returns list of dicts
        [
            {
                "correct": True,
                "test_results": {0: True, 1: True, ...}
            },
            ...
        ]
        """
        # Save stats, all passing at the beginning (flip to False on test failure)
        stats = [{
            "correct": True,
            "test_results": {}
        } for _ in code_strs]
        clean_code_strs = [clean_code(code_str) for code_str in code_strs]
        for test_i, test in enumerate(prob.tests):
            logger.info("Running spike on %d implementations", len(code_strs))
            test_output_per_code_str = run_spike_mp([test.get_test_code([code_str]) for code_str in clean_code_strs], self.gemmini_path, timeout=120)
            num_compile_errors = 0
            num_correct = 0
            num_incorrect = 0
            for code_i, test_output in enumerate(test_output_per_code_str):
                # logger.debug(test_output)
                if test_output == "Compile error" or test_output == "Timeout":
                    logger.debug("Test %d %s for code %d", test_i, test_output, code_i)
                    stats[code_i]["test_results"][test_i] = False
                    stats[code_i]["correct"] = False
                    stats[code_i]["compiled"] = False
                    num_compile_errors += 1
                elif "Correct" in test_output:
                    logger.debug("Test %d Correct result", test_i)
                    stats[code_i]["test_results"][test_i] = True
                    if "compiled" not in stats[code_i]:
                        stats[code_i]["compiled"] = True
                    if simulator == "spike": # Get instruction count from spike
                        if "Generated implementation latency" in test_output:
                            sol_latency = int(test_output.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                            stats[code_i]["latency_spike_raw"] = sol_latency
                            # MX_NOFPU_COST: the host Rocket in RadianceGemminiOnlyConfig has NO FPU
                            # (fpu=None); float ops TRAP on real RTL (verified: fpu_probe FAILED on VCS)
                            # but spike has an FPU and runs them ~1 cyc/op, so spike UNDER-counts float.
                            # Penalize host float so the optimizer prefers float-free (deployable) kernels.
                            if os.environ.get("MX_NOFPU_COST"):
                                nflt = _count_host_float_ops(clean_code_strs[code_i])
                                # float traps on this HW -> a float-using kernel is non-deployable;
                                # use a disqualifying per-occurrence penalty (not a calibrated per-op cost).
                                penalty = int(os.environ.get("MX_FLOAT_PENALTY", "100000000"))
                                sol_latency = sol_latency + penalty * nflt
                            stats[code_i]["latency"] = sol_latency
                    num_correct += 1
                else:
                    logger.debug("Test %d Incorrect result", test_i)
                    stats[code_i]["test_results"][test_i] = False
                    stats[code_i]["correct"] = False
                    if "compiled" not in stats[code_i]:
                        stats[code_i]["compiled"] = True
                    num_incorrect += 1
            logger.info("Test %d: %d compiled (%d correct, %d incorrect), %d compile errors",
                        test_i, num_correct + num_incorrect, num_correct, num_incorrect, num_compile_errors)

        if simulator == "firesim":
            # test = prob.tests[0]
            # binary_paths = [compile_gemmini_code(test.get_test_code([code_str]), self.gemmini_path) for code_str in clean_code_strs]
            # for code_i, binary_path in enumerate(binary_paths):
            #     if binary_path:
            #         logger.debug("Code %d Compile success", code_i)
            #     else:
            #         logger.debug("Code %d Compile error", code_i)
            #         stats[code_i]["correct"] = False
            # Use batched run to speed up evaluation
            working_code_idxs = []
            num_compiled = sum(1 for s in stats if s.get("compiled", True))
            for code_i in range(len(code_strs)):
                if stats[code_i]["correct"]: # Get correctness from spike
                    working_code_idxs.append(code_i)
            logger.info("%d of %d implementations passed spike (%d compiled, %d compile errors)",
                        len(working_code_idxs), len(code_strs), num_compiled, len(code_strs) - num_compiled)
            logger.debug("Working code indices: %s", str(working_code_idxs))
            # logger.info("%d of %d implementations compiled successfully", len(working_code_idxs), len(code_strs))
            if len(working_code_idxs) == 0:
                return stats
            working_code_strs = [clean_code_strs[i] for i in working_code_idxs]
            first_test = prob.tests[0]
            # Batch N at a time
            batch_size = 20
            batch_start_idx = 0
            while batch_start_idx < len(working_code_strs):
                batch_end_idx = min(batch_start_idx + batch_size, len(working_code_strs))
                if self.pe_dim == 4:
                    this_batch_firesim_output = None
                else:
                    this_batch_firesim_output = run_firesim([first_test.get_test_code(working_code_strs[batch_start_idx:batch_end_idx], error_on_incorrect=False, repeat_iters=1)], 
                                                            self.gemmini_path, self.firesim_path, timeout=300)[0]
                if not this_batch_firesim_output:
                    if self.pe_dim != 4:
                        logger.warning("Code hang on batch, running individually")
                    if self.pe_dim == 4:    
                        individual_firesim_outputs = run_firesim([first_test.get_test_code([working_code_strs[idx]], error_on_incorrect=True, repeat_iters=20) for idx in range(batch_start_idx, batch_end_idx)], 
                                                                 self.gemmini_path, self.firesim_path, timeout=180)
                    else:
                        individual_firesim_outputs = run_firesim([first_test.get_test_code([working_code_strs[idx]], error_on_incorrect=False, repeat_iters=1) for idx in range(batch_start_idx, batch_end_idx)], 
                                                                 self.gemmini_path, self.firesim_path, timeout=120)
                    for idx, output in enumerate(individual_firesim_outputs):
                        orig_idx = working_code_idxs[batch_start_idx+idx]
                        if not output:
                            stats[orig_idx]["correct"] = False
                            logger.warning("Code hang on index %s, batch index %s", orig_idx, idx)
                        else:
                            latency_found = False
                            if "Incorrect result" in output:
                                stats[orig_idx]["correct"] = False
                                stats[orig_idx]["test_results"][0] = False
                                continue
                            for line in output.split("\n"):
                                if "Generated implementation latency" in line:
                                    sol_latency = int(line.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                                    if sol_latency == 99999999999:
                                        stats[orig_idx]["correct"] = False
                                        stats[orig_idx]["test_results"][0] = False
                                    else:
                                        stats[orig_idx]["latency"] = sol_latency
                                        stats[orig_idx]["test_results"][0] = True
                                    latency_found = True
                                    break
                            if not latency_found:
                                logger.error("Firesim output did not contain latency for index %s, batch index %s", orig_idx, idx)
                else:
                    cur_working_code_i = 0
                    for line in this_batch_firesim_output.split("\n"):
                        if "Generated implementation latency" in line:
                            sol_latency = int(line.split("Generated implementation latency: ")[-1].split(" cycles")[0])
                            orig_idx = working_code_idxs[batch_start_idx+cur_working_code_i]
                            if sol_latency == 99999999999:
                                stats[orig_idx]["correct"] = False
                                stats[orig_idx]["test_results"][0] = False
                            else:
                                stats[orig_idx]["latency"] = sol_latency
                                stats[orig_idx]["test_results"][0] = True
                            cur_working_code_i += 1
                    if cur_working_code_i != (batch_end_idx - batch_start_idx):
                        raise ValueError("Firesim output did not contain enough latencies.")

                batch_start_idx = batch_end_idx
        logger.debug(stats)
        return stats

if __name__ == "__main__":
    prob = Prob("admm-multifunction", 2)
    files = [SOLS_DIR / "admm-multifunction" / "sol2_5249.c"]
    code_strs = [file.read_text() for file in files]
    stats = GemminiEvalBackend(4).evaluate_code(prob, code_strs, "firesim")
    # stats = GemminiEvalBackend(4).get_spad_acc_utilization(prob, code_strs)
    print(stats)
