"""
Test harness for Problem 2: RMS normalization (bf16). norm = x * rsqrt(mean(x^2) + eps)

Production flow matches split-halves input (what MXU/other NPU ops produce) and
uses DMA-preloaded constants for 1/N and eps — scalar-store construction of
these constants is wasteful compared to a parallel DMA load.

Memory layout:
    DRAM 0x0000 — x left  (32, 16) bf16
    DRAM 0x0400 — x right (32, 16) bf16
    DRAM 0x0800 — INV_N   (32, 16) bf16 all-1/32
    DRAM 0x0C00 — EPS     (32, 16) bf16 all-1e-6
    DRAM 0x0b00 — out left  (32, 16) bf16
    DRAM 0x0f00 — out right (32, 16) bf16 (unchecked; both halves stored)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

DRAM_X_LEFT = 0x0000
DRAM_X_RIGHT = 0x0400
DRAM_INV_N = 0x0800
DRAM_EPS = 0x0C00
DRAM_OUTPUT_BASE = 0x0b00

torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
X_LEFT = ACTIVATION_DATA[:, :16].contiguous()
X_RIGHT = ACTIVATION_DATA[:, 16:].contiguous()
INV_N_TILE = torch.full((32, 16), 1.0 / 32.0, dtype=torch.bfloat16)
EPS_TILE = torch.full((32, 16), 1e-6, dtype=torch.bfloat16)

# Hardware-matched stepwise bf16 golden
_sq_l = (X_LEFT.float() * X_LEFT.float()).to(torch.bfloat16)
_sq_r = (X_RIGHT.float() * X_RIGHT.float()).to(torch.bfloat16)
_sum_l = _sq_l.float().sum(dim=1, keepdim=True).to(torch.bfloat16).expand(32, 16).contiguous()
_sum_r = _sq_r.float().sum(dim=1, keepdim=True).to(torch.bfloat16).expand(32, 16).contiguous()
_sum = (_sum_l.float() + _sum_r.float()).to(torch.bfloat16)
_mean = (_sum.float() * INV_N_TILE.float()).to(torch.bfloat16)
_var = (_mean.float() + EPS_TILE.float()).to(torch.bfloat16)
_rms = torch.sqrt(_var.float()).to(torch.bfloat16)
_inv_rms = (1.0 / _rms.float()).to(torch.bfloat16)
GOLDEN_OUTPUT = (X_LEFT.float() * _inv_rms.float()).to(torch.bfloat16)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_LEFT, X_LEFT),
        (DRAM_X_RIGHT, X_RIGHT),
        (DRAM_INV_N, INV_N_TILE),
        (DRAM_EPS, EPS_TILE),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
