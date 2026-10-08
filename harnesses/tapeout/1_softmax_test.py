"""
Test harness for Problem 1: Softmax along last dimension (bf16).

Split-halves input + broadcast-split output matches production layout (what
MXU / other NPU ops produce and consume). Golden computed pairwise in bf16 to
match the hardware's fp32-promoted reduction precision.

Memory layout:
    DRAM 0x0000 — A left  (32, 16) bf16
    DRAM 0x0400 — A right (32, 16) bf16
    DRAM 0x0b00 — softmax(A) left  (golden checked)
    DRAM 0x0f00 — softmax(A) right (also stored, not checked)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

DRAM_A_LEFT = 0x0000
DRAM_A_RIGHT = 0x0400
DRAM_OUTPUT_BASE = 0x0b00

torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
A_LEFT = ACTIVATION_DATA[:, :16].contiguous()
A_RIGHT = ACTIVATION_DATA[:, 16:].contiguous()

# Hardware-matched stepwise bf16 golden (fp32 promote for reductions)
_max_l = A_LEFT.float().max(dim=1, keepdim=True).values.to(torch.bfloat16).expand(32, 16).contiguous()
_max_r = A_RIGHT.float().max(dim=1, keepdim=True).values.to(torch.bfloat16).expand(32, 16).contiguous()
_max = torch.maximum(_max_l, _max_r)
_shift_l = (A_LEFT.float() - _max.float()).to(torch.bfloat16)
_shift_r = (A_RIGHT.float() - _max.float()).to(torch.bfloat16)
_exp_l = torch.exp(_shift_l.float()).to(torch.bfloat16)
_exp_r = torch.exp(_shift_r.float()).to(torch.bfloat16)
_sum_l = _exp_l.float().sum(dim=1, keepdim=True).to(torch.bfloat16).expand(32, 16).contiguous()
_sum_r = _exp_r.float().sum(dim=1, keepdim=True).to(torch.bfloat16).expand(32, 16).contiguous()
_sum = (_sum_l.float() + _sum_r.float()).to(torch.bfloat16)
_inv_sum = (1.0 / _sum.float()).to(torch.bfloat16)
GOLDEN_OUTPUT = (_exp_l.float() * _inv_sum.float()).to(torch.bfloat16)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A_LEFT, A_LEFT),
        (DRAM_A_RIGHT, A_RIGHT),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
