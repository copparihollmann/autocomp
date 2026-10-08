"""
Test harness for Problem 20: Gemma MLP Full Layer (fp8/bf16)

This is a LAYER-LEVEL problem. The LLM must generate a complete MLP layer
program including DMA pipelining and tiling.

Layer computation (PyTorch reference):
    x_norm = rms_norm(x)
    gate = x_norm @ gate_proj_weight.T   (fp8 matmul)
    up   = x_norm @ up_proj_weight.T     (fp8 matmul)
    gate_activated = silu(gate)
    gated = gate_activated * up
    down = gated @ down_proj_weight.T    (fp8 matmul)
    output = x + down                    (residual add)

Memory layout (all tensors are 32x32 tiles):
    DRAM 0x0000 — gate_proj weight (32x32 fp8, 1024 bytes)
    DRAM 0x0400 — up_proj weight (32x32 fp8, 1024 bytes)
    DRAM 0x0800 — down_proj weight (32x32 fp8, 1024 bytes)
    DRAM 0x0C00 — input activation (32x32 fp8, 1024 bytes)
    DRAM 0x1000 — output (32x32 bf16, 2048 bytes)

Performance target: Pipeline DMA loads with MXU computation.
Use multiple DMA channels to overlap weight loading with compute.
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants ──
DRAM_GATE_WEIGHT_BASE = 0x0000
DRAM_UP_WEIGHT_BASE = 0x0400
DRAM_DOWN_WEIGHT_BASE = 0x0800
DRAM_ACTIVATION_BASE = 0x0C00
DRAM_OUTPUT_BASE = 0x1000

# ── Input data ──
torch.manual_seed(42)
GATE_WEIGHT = torch.randn(32, 32).to(torch.float8_e4m3fn)
UP_WEIGHT = torch.randn(32, 32).to(torch.float8_e4m3fn)
DOWN_WEIGHT = torch.randn(32, 32).to(torch.float8_e4m3fn)
ACTIVATION_DATA = torch.randn(32, 32).to(torch.float8_e4m3fn)

# ── Golden output (full MLP layer in PyTorch) ──
_x = ACTIVATION_DATA.float()
# RMS norm (simplified: just normalize, no learned weight)
_var = _x.pow(2).mean(-1, keepdim=True)
_x_norm = _x * torch.rsqrt(_var + 1e-6)
# Gate and up projections
_gate = _x_norm @ GATE_WEIGHT.float().T
_up = _x_norm @ UP_WEIGHT.float().T
# SiLU activation on gate
_gate_activated = _gate * torch.sigmoid(_gate)
# Gated multiplication
_gated = _gate_activated * _up
# Down projection
_down = _gated @ DOWN_WEIGHT.float().T
# Residual add
_output = _x + _down
_output_bf16 = _output.to(torch.bfloat16)
# Reshape for NPU MRF layout (32x32 bf16 → 64x16 as two m-registers)
GOLDEN_OUTPUT = torch.cat((_output_bf16[:, :16], _output_bf16[:, 16:]), dim=0)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_GATE_WEIGHT_BASE, GATE_WEIGHT),
        (DRAM_UP_WEIGHT_BASE, UP_WEIGHT),
        (DRAM_DOWN_WEIGHT_BASE, DOWN_WEIGHT),
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)
