"""
Test harness for Problem 6: Element-wise multiplication (bf16): C = A * B

PyTorch reference:
    GOLDEN_OUTPUT = (ACTIVATION_DATA.float() * WEIGHT_DATA.float()).to(torch.bfloat16)

Memory layout:
    DRAM_ACTIVATION_BASE = 0x0  — A ((32, 32) torch.bfloat16, 2048 bytes)
    DRAM_WEIGHT_BASE     = 0x900  — B ((32, 32) torch.bfloat16, 2048 bytes)
    DRAM_OUTPUT_BASE     = 0x1300  — C ((32, 32) torch.bfloat16)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants (problem-owned) ──
DRAM_ACTIVATION_BASE = 0x0
DRAM_WEIGHT_BASE = 0x900
DRAM_OUTPUT_BASE = 0x1300

# ── Input data (problem-owned) ──
torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
WEIGHT_DATA = torch.randn(32, 32, dtype=torch.bfloat16)

# ── Golden output (computed from PyTorch reference) ──
GOLDEN_OUTPUT = (ACTIVATION_DATA.float() * WEIGHT_DATA.float()).to(torch.bfloat16)

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
