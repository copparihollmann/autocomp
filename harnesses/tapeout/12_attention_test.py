"""
Test harness for Problem 12: Fused scaled dot-product attention: softmax(Q@K^T/sqrt(d))@V (bf16, single head)

PyTorch reference:
    _Q = ACTIVATION_DATA.float()
    _K = WEIGHT_DATA.float()
    _V = EXTRA_DATA.float()
    _scale = 1.0 / (32 ** 0.5)
    _scores = _Q @ _K.transpose(-2, -1) * _scale
    _attn = torch.nn.functional.softmax(_scores, dim=-1)
    _out = _attn @ _V
    GOLDEN_OUTPUT = torch.cat((_out[:, :16], _out[:, 16:]), dim=0)

Memory layout:
    DRAM_ACTIVATION_BASE = 0x0  — A ((32, 32) torch.bfloat16, 2048 bytes)
    DRAM_WEIGHT_BASE     = 0x900  — B ((32, 32) torch.bfloat16, 2048 bytes)
    DRAM_EXTRA_BASE      = 0x1200  — V ((32, 32) torch.bfloat16, 2048 bytes)
    DRAM_OUTPUT_BASE     = 0x1c00  — C ((32, 32) torch.bfloat16)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants (problem-owned) ──
DRAM_ACTIVATION_BASE = 0x0
DRAM_WEIGHT_BASE = 0x900
DRAM_EXTRA_BASE = 0x1200
DRAM_OUTPUT_BASE = 0x1c00

# ── Input data (problem-owned) ──
torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
WEIGHT_DATA = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
EXTRA_DATA = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1

# ── Golden output (computed from PyTorch reference) ──
_Q = ACTIVATION_DATA.float()
_K = WEIGHT_DATA.float()
_V = EXTRA_DATA.float()
_scale = 1.0 / (32 ** 0.5)
_scores = _Q @ _K.transpose(-2, -1) * _scale
_attn = torch.nn.functional.softmax(_scores, dim=-1)
_out = _attn @ _V
GOLDEN_OUTPUT = torch.cat((_out[:, :16], _out[:, 16:]), dim=0)

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
    (DRAM_EXTRA_BASE, EXTRA_DATA),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
