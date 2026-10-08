"""
Test harness for Problem 31: Fast GELU tanh with pre-loaded constants

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Constants are pre-loaded in DRAM as bf16 tiles (filled with the constant value).
This avoids the expensive scalar-store constant construction that made the
previous gelu kernel 796 cycles. With DMA-loaded constants and vtanh.bf16
hardware instruction, this should be ~200 cycles.

Memory layout:
    DRAM 0x0000 — input x (32x32 bf16, 2048 bytes)
    DRAM 0x0800 — constant tile: 0.5 (32x16 bf16, 1024 bytes)
    DRAM 0x0C00 — constant tile: sqrt(2/pi) = 0.7978846 (32x16 bf16, 1024 bytes)
    DRAM 0x1000 — constant tile: 0.044715 (32x16 bf16, 1024 bytes)
    DRAM 0x1400 — constant tile: 1.0 (32x16 bf16, 1024 bytes)
    DRAM 0x1800 — output (32x32 bf16, 2048 bytes)

Key insight: Use vtanh.bf16 (hardware tanh, ~8 cycle latency) instead of
computing tanh from exp/recip.
"""
from typing import List, Tuple, Any
import math

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout ──
DRAM_INPUT_BASE = 0x0000
DRAM_HALF_BASE = 0x0800
DRAM_SQRT2PI_BASE = 0x0C00
DRAM_COEFF_BASE = 0x1000
DRAM_ONE_BASE = 0x1400
DRAM_OUTPUT_BASE = 0x1800

# ── Input data ──
torch.manual_seed(42)
INPUT_DATA = torch.randn(32, 32, dtype=torch.bfloat16)

# Pre-filled constant tiles (32x16 bf16 = 1024 bytes, one MRF register)
HALF_TILE = torch.full((32, 16), 0.5, dtype=torch.bfloat16)
SQRT2PI_TILE = torch.full((32, 16), math.sqrt(2.0 / math.pi), dtype=torch.bfloat16)
COEFF_TILE = torch.full((32, 16), 0.044715, dtype=torch.bfloat16)
ONE_TILE = torch.full((32, 16), 1.0, dtype=torch.bfloat16)

# ── Golden output ──
_x = INPUT_DATA.float()
GOLDEN_OUTPUT = torch.nn.functional.gelu(_x, approximate='tanh').to(torch.bfloat16)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_HALF_BASE, HALF_TILE),
        (DRAM_SQRT2PI_BASE, SQRT2PI_TILE),
        (DRAM_COEFF_BASE, COEFF_TILE),
        (DRAM_ONE_BASE, ONE_TILE),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
