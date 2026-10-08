"""
Test harness for Problem 0: FP8 matrix multiply: C = A @ B^T (fp8×fp8 → bf16)

PyTorch reference:
    _result = (ACTIVATION_DATA.float() @ WEIGHT_DATA.float()).to(torch.bfloat16)
    GOLDEN_OUTPUT = torch.cat((_result[:, :16], _result[:, 16:]), dim=0)

Memory layout:
    DRAM_ACTIVATION_BASE = 0x0  — A ((32, 32) torch.float8_e4m3fn, 1024 bytes)
    DRAM_WEIGHT_BASE     = 0x500  — B ((32, 32) torch.float8_e4m3fn, 1024 bytes)
    DRAM_OUTPUT_BASE     = 0xb00  — C ((32, 32) torch.bfloat16)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants (problem-owned) ──
DRAM_ACTIVATION_BASE = 0x0
DRAM_WEIGHT_BASE = 0x500
DRAM_OUTPUT_BASE = 0xb00

# ── Input data (problem-owned) ──
torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)
WEIGHT_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)

# ── Golden output (computed from PyTorch reference) ──
_result = (ACTIVATION_DATA.float() @ WEIGHT_DATA.float()).to(torch.bfloat16)
GOLDEN_OUTPUT = torch.cat((_result[:, :16], _result[:, 16:]), dim=0)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
    (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
    (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
