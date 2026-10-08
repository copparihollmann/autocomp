"""
Test harness for Problem 9: Reduce-sum along last dimension (bf16).

Production flow: reduction feeds broadcast ops (rms_norm, softmax). The natural
output is therefore a (32, 16) broadcast-row-sum tile, and the natural input
layout is split halves (matching what MXU and other NPU ops produce).

Hardware approximation of sum: `vredsum.row.bf16` promotes to fp32 for the
reduction, then casts to bf16 — so the golden is computed pairwise in bf16 to
match that precision.

Memory layout:
    DRAM_ACTIVATION_LEFT  = 0x0000  — A[:, :16]  (32, 16) bf16, 1024 B
    DRAM_ACTIVATION_RIGHT = 0x0400  — A[:, 16:]  (32, 16) bf16, 1024 B
    DRAM_OUTPUT_BASE      = 0x0b00  — row-sum broadcast (32, 16) bf16, 1024 B
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

DRAM_ACTIVATION_LEFT = 0x0000
DRAM_ACTIVATION_RIGHT = 0x0400
DRAM_OUTPUT_BASE = 0x0b00

torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
ACTIVATION_LEFT = ACTIVATION_DATA[:, :16].contiguous()
ACTIVATION_RIGHT = ACTIVATION_DATA[:, 16:].contiguous()

# Hardware: m2 = m_left + m_right (bf16), m3 = vredsum.row(m2) (fp32 then bf16)
_m2 = (ACTIVATION_LEFT.float() + ACTIVATION_RIGHT.float()).to(torch.bfloat16)
_row_sum = _m2.float().sum(dim=1, keepdim=True).to(torch.bfloat16)
GOLDEN_OUTPUT = _row_sum.expand(32, 16).contiguous()

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_LEFT, ACTIVATION_LEFT),
        (DRAM_ACTIVATION_RIGHT, ACTIVATION_RIGHT),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
