"""
Test harness for Problem 30: Tiled Matmul 2x1 with DMA Pipelining

Computes C = A @ B where A is 64x32 (2 tiles tall) and B is 32x32 (1 tile).
The result C is 64x32 bf16, computed as two 32x32 fp8 matmul tiles.

The LLM must generate a program that:
1. Loads tile 0 of A (32x32 fp8) and B (32x32 fp8) from DRAM
2. Computes tile 0 matmul while loading tile 1 of A (DMA pipelining)
3. Computes tile 1 matmul
4. Stores both result tiles to DRAM

Key optimization: overlap DMA load of tile 1 with MXU compute of tile 0.
The DMA unit and MXU are separate execution units that run in parallel.

Memory layout:
    DRAM 0x0000 — A tile 0 (32x32 fp8, 1024 bytes)
    DRAM 0x0400 — A tile 1 (32x32 fp8, 1024 bytes)
    DRAM 0x0800 — B weight (32x32 fp8, 1024 bytes)
    DRAM 0x0C00 — C tile 0 output (32x16 bf16 × 2 halves, 2048 bytes)
    DRAM 0x1400 — C tile 1 output (32x16 bf16 × 2 halves, 2048 bytes)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout ──
DRAM_A0_BASE = 0x0000
DRAM_A1_BASE = 0x0400
DRAM_B_BASE = 0x0800
DRAM_C0_BASE = 0x0C00
DRAM_C1_BASE = 0x1400

# ── Input data ──
torch.manual_seed(42)
A0_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)
A1_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)
B_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)

# ── Golden output ──
_C0 = (A0_DATA.float() @ B_DATA.float()).to(torch.bfloat16)
_C1 = (A1_DATA.float() @ B_DATA.float()).to(torch.bfloat16)
# NPU MRF layout: 32x32 bf16 → 64x16 (two halves stacked)
_C0_mrf = torch.cat((_C0[:, :16], _C0[:, 16:]), dim=0)
_C1_mrf = torch.cat((_C1[:, :16], _C1[:, 16:]), dim=0)
GOLDEN_C0 = _C0_mrf
GOLDEN_C1 = _C1_mrf

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A0_BASE, A0_DATA),
        (DRAM_A1_BASE, A1_DATA),
        (DRAM_B_BASE, B_DATA),
    ]
    # Golden: tile 0 output at DRAM_C0_BASE, tile 1 at DRAM_C1_BASE
    # We check tile 0 (the harness checks one golden result)
    golden_result: Tuple[int, torch.Tensor] = (DRAM_C0_BASE, GOLDEN_C0)
