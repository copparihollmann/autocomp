"""
Test harness for Problem 11: Requantization bf16 → fp8 (vpack.bf16.fp8 on NPU)

In production, requant consumes bf16 tiles produced by the MXU. The MXU writes
its 32×32 bf16 output as TWO mrf registers (left 16 cols in m_low, right 16 in
m_high) — this is exactly the layout vpack.bf16.fp8 consumes. The harness
mirrors that: the input is laid out as two (32×16) bf16 halves in DRAM so a
simple DMA+vload produces the correct mrf split.

PyTorch reference:
    GOLDEN_OUTPUT = ACTIVATION_DATA.to(torch.float8_e4m3fn)

Memory layout:
    DRAM_ACTIVATION_LEFT  = 0x0000  — A[:, :16] ((32, 16) bf16, 1024 bytes)
    DRAM_ACTIVATION_RIGHT = 0x0400  — A[:, 16:] ((32, 16) bf16, 1024 bytes)
    DRAM_OUTPUT_BASE      = 0x0b00  — C ((32, 32) fp8, 1024 bytes)

Key insight for the kernel: vpack reads ERF[es1] as a scale multiplier. Default
ERF is 0, which zeros the output — use `seli rd=k, imm=1` to write unit scale
into ERF[k] before vpack.
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants (problem-owned) ──
DRAM_ACTIVATION_LEFT = 0x0000
DRAM_ACTIVATION_RIGHT = 0x0400
DRAM_OUTPUT_BASE = 0x0b00

# ── Input data (problem-owned) ──
torch.manual_seed(42)
ACTIVATION_DATA = torch.randn(32, 32, dtype=torch.bfloat16) * 0.5
ACTIVATION_LEFT = ACTIVATION_DATA[:, :16].contiguous()
ACTIVATION_RIGHT = ACTIVATION_DATA[:, 16:].contiguous()

# ── Golden output (computed from PyTorch reference) ──
GOLDEN_OUTPUT = ACTIVATION_DATA.to(torch.float8_e4m3fn)

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
