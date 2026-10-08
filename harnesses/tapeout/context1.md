# Working NPU Kernel Examples

## 00_matmul
```python
def test() -> list:
    """
    Computes C = A @ B^T (fp8 x fp8 -> bf16) on the NPU.
    - A is a 32x32 fp8 matrix (activation)
    - B is a 32x32 fp8 matrix (weights)
    - C is a 32x32 bf16 matrix (output)
    """
    # Define symbolic names for registers for clarity
    # Scalar registers
    DRAM_A_ADDR = 1
    DRAM_B_ADDR = 2
    DRAM_C_ADDR = 3
    VMEM_A_ADDR = 4
    VMEM_B_ADDR = 5
    VMEM_C_ADDR = 6
    SIZE_A_B = 7
    SIZE_C = 8
    VMEM_C_HALF_ADDR = 9  # For the second half of the output matrix
    # Tensor registers
    M_A = 0
    M_B = 1
    M_C_PAIR = 2  # m2 and m3 (must be even)

    # Memory and size constants
    dram_a_base = 0x0
    dram_b_base = 0x500
    dram_c_base = 0xb00
    vmem_a_base = 0x2000
    fp8_tile_size = 1024
    bf16_tile_size = 2048

    prog = [
        # 1. Initialization: Setup scalar registers with addresses and sizes
        Instruction("addi", ScalarArgs(rd=DRAM_A_ADDR, rs1=0, imm=dram_a_base)),
        Instruction("addi", ScalarArgs(rd=DRAM_B_ADDR, rs1=0, imm=dram_b_base)),
        Instruction("addi", ScalarArgs(rd=DRAM_C_ADDR, rs1=0, imm=dram_c_base)),

        # VMEM addresses are > 2047, so they require a lui/addi sequence.
        Instruction("lui", ScalarArgs(rd=VMEM_A_ADDR, imm=(vmem_a_base >> 12))),
        Instruction("addi", ScalarArgs(rd=VMEM_A_ADDR, rs1=VMEM_A_ADDR, imm=(vmem_a_base & 0xFFF))),
        Instruction("addi", ScalarArgs(rd=VMEM_B_ADDR, rs1=VMEM_A_ADDR, imm=fp8_tile_size)),
        Instruction("addi", ScalarArgs(rd=VMEM_C_ADDR, rs1=VMEM_B_ADDR, imm=fp8_tile_size)),

        Instruction("addi", ScalarArgs(rd=SIZE_A_B, rs1=0, imm=fp8_tile_size)),
        Instruction("addi", ScalarArgs(rd=SIZE_C, rs1=0, imm=bf16_tile_size)),

        # 2. Data Ingress (DRAM -> VMEM)
        # Configure shared DMA base address to 0. One call is sufficient for all channels.
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),

        # Start DMA transfers in parallel on channels 0 and 1.
        # dma.load: rd=VMEM_addr_reg, rs1=DRAM_addr_reg, rs2=size_reg
        Instruction("dma.load.ch<N>", DmaArgs(rd=VMEM_A_ADDR, rs1=DRAM_A_ADDR, rs2=SIZE_A_B, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=VMEM_B_ADDR, rs1=DRAM_B_ADDR, rs2=SIZE_A_B, channel=1)),

        # Wait for both DMAs to complete before using the data.
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),

        # 3. Computation (MXU)
        # Load data from VMEM into tensor registers (blocking operations).
        Instruction("vload", VectorArgs(vd=M_A, rs1=VMEM_A_ADDR, imm12=0)),
        Instruction("vload", VectorArgs(vd=M_B, rs1=VMEM_B_ADDR, imm12=0)),

        # Push weights from m1 into MXU0's weight slot (blocking).
        Instruction("vmatpush.weight.fp8.mxu0", VectorArgs(vd=0, vs1=M_B, vs2=0)),

        # Issue matrix multiply: A(m0) @ B(in slot) -> MXU0 accumulator.
        # This is non-blocking with a 32-cycle latency.
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=M_A, vs2=0)),

        # Stall the instruction pipeline to cover the matmul latency.
        # delay(30) + 1 cycle for delay issue = 31 cycles. Matmul issues 1 cycle before.
        # Total latency covered is 32 cycles.
        Instruction("delay", ScalarArgs(imm=30)),

        # Pop the 32x32 BF16 result from the accumulator into tensor registers m2/m3 (blocking).
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=M_C_PAIR, vs1=0, vs2=0)),

        # 4. Result Retrieval (MXU to VMEM)
        # Store the two 1024-byte half-tiles back to a contiguous VMEM region.
        Instruction("vstore", VectorArgs(vd=M_C_PAIR, rs1=VMEM_C_ADDR, imm12=0)),
        Instruction("addi", ScalarArgs(rd=VMEM_C_HALF_ADDR, rs1=VMEM_C_ADDR, imm=fp8_tile_size)),
        Instruction("vstore", VectorArgs(vd=M_C_PAIR + 1, rs1=VMEM_C_HALF_ADDR, imm12=0)),

        # 5. Data Egress (VMEM -> DRAM)
        # DMA the final result from VMEM to DRAM.
        # dma.store: rd=DRAM_addr_reg, rs1=VMEM_addr_reg, rs2=size_reg
        Instruction("dma.store.ch<N>", DmaArgs(rd=DRAM_C_ADDR, rs1=VMEM_C_ADDR, rs2=SIZE_C, channel=0)),
        # Wait for the DMA store to complete.
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),

        # 6. Termination
        # Halt the processor.
        Instruction("ecall", ScalarArgs(imm=0)),
    ]
    return prog
```

## 05_elementwise_add
```python
def test() -> List[Instruction[Any]]:
    """
    Implements element-wise addition C = A + B for 32x32 bf16 matrices.
    """

    def load_imm(reg_idx: int, value: int) -> List[Instruction[Any]]:
        """Helper to load a 32-bit immediate into a scalar register."""
        if -2048 <= value < 2048:
            return [Instruction("addi", ScalarArgs(rd=reg_idx, rs1=0, imm=value))]

        # Standard LUI/ADDI sequence for RISC-V
        lower_bits = value & 0xFFF
        upper_bits = (value >> 12)
        if lower_bits & 0x800:  # If the lower 12 bits would be sign-extended as negative
            upper_bits += 1

        instructions = [Instruction("lui", ScalarArgs(rd=reg_idx, imm=upper_bits))]
        # Only add the ADDI if the lower bits are non-zero, avoiding a redundant `addi rd, rd, 0`
        if (value - (upper_bits << 12)) != 0:
             instructions.append(Instruction("addi", ScalarArgs(rd=reg_idx, rs1=reg_idx, imm=lower_bits)))
        return instructions

    # --- Register and Memory Allocation ---
    # Scalar registers for addresses and sizes
    DRAM_A_ADDR_REG = 1
    DRAM_B_ADDR_REG = 2
    DRAM_C_ADDR_REG = 3
    VMEM_A_ADDR_REG = 4
    VMEM_B_ADDR_REG = 5
    VMEM_C_ADDR_REG = 6
    TRANSFER_SIZE_REG = 7

    # Memory addresses from test harness
    DRAM_A_ADDR = 0x0
    DRAM_B_ADDR = 0x900
    DRAM_C_ADDR = 0x1300
    # VMEM staging addresses
    VMEM_A_ADDR = 0x2000
    VMEM_B_ADDR = 0x2800
    VMEM_C_ADDR = 0x3000
    TRANSFER_SIZE = 2048  # 32*32*2 bytes for a bf16 matrix

    # Tensor registers for matrices (2 regs per matrix)
    A_MREG_BASE = 0
    B_MREG_BASE = 2
    C_MREG_BASE = 4

    # --- Program ---
    program: List[Instruction[Any]] = []

    # 1. Setup: Load constants into scalar registers
    program.extend(load_imm(DRAM_A_ADDR_REG, DRAM_A_ADDR))
    program.extend(load_imm(DRAM_B_ADDR_REG, DRAM_B_ADDR))
    program.extend(load_imm(DRAM_C_ADDR_REG, DRAM_C_ADDR))
    program.extend(load_imm(VMEM_A_ADDR_REG, VMEM_A_ADDR))
    program.extend(load_imm(VMEM_B_ADDR_REG, VMEM_B_ADDR))
    program.extend(load_imm(VMEM_C_ADDR_REG, VMEM_C_ADDR))
    program.extend(load_imm(TRANSFER_SIZE_REG, TRANSFER_SIZE))

    # 2. Data Ingress: DMA A and B from DRAM to VMEM concurrently
    # Configure DMA once (sets base address to 0 for all channels)
    program.append(Instruction("dma.config.ch<N>", DmaArgs(rd=0, rs1=0, channel=0)))
    # Start DMA for matrix A (DRAM -> VMEM) on channel 0
    program.append(Instruction("dma.load.ch<N>", DmaArgs(rd=VMEM_A_ADDR_REG, rs1=DRAM_A_ADDR_REG, rs2=TRANSFER_SIZE_REG, channel=0)))
    # Start DMA for matrix B (DRAM -> VMEM) on channel 1
    program.append(Instruction("dma.load.ch<N>", DmaArgs(rd=VMEM_B_ADDR_REG, rs1=DRAM_B_ADDR_REG, rs2=TRANSFER_SIZE_REG, channel=1)))

    # 3. Computation: Load from VMEM, perform addition
    # Wait for A to arrive, then load it into tensor registers
    program.append(Instruction("dma.wait.ch<N>", DmaArgs(rd=0, rs1=0, channel=0)))
    # A 32x32 bf16 tile is 2048 bytes; a vload moves 1024 bytes. Need two vloads.
    # imm12 is scaled by 32 (<< 5), so an offset of 1024 bytes is imm12=32.
    program.append(Instruction("vload", VectorArgs(vd=A_MREG_BASE, rs1=VMEM_A_ADDR_REG, imm12=0)))
    program.append(Instruction("vload", VectorArgs(vd=A_MREG_BASE + 1, rs1=VMEM_A_ADDR_REG, imm12=32)))

    # Wait for B to arrive, then load it
    program.append(Instruction("dma.wait.ch<N>", DmaArgs(rd=0, rs1=0, channel=1)))
    program.append(Instruction("vload", VectorArgs(vd=B_MREG_BASE, rs1=VMEM_B_ADDR_REG, imm12=0)))
    program.append(Instruction("vload", VectorArgs(vd=B_MREG_BASE + 1, rs1=VMEM_B_ADDR_REG, imm12=32)))

    # Element-wise add. VPU operates on 32x16 half-tiles (one m-register)
    program.append(Instruction("vadd.bf16", VectorArgs(vd=C_MREG_BASE, vs1=A_MREG_BASE, vs2=B_MREG_BASE)))
    program.append(Instruction("vadd.bf16", VectorArgs(vd=C_MREG_BASE + 1, vs1=A_MREG_BASE + 1, vs2=B_MREG_BASE + 1)))

    # 4. Data Egress: Store result C to VMEM, then DMA to DRAM
    program.append(Instruction("vstore", VectorArgs(vd=C_MREG_BASE, rs1=VMEM_C_ADDR_REG, imm12=0)))
    program.append(Instruction("vstore", VectorArgs(vd=C_MREG_BASE + 1, rs1=VMEM_C_ADDR_REG, imm12=32)))

    # DMA result C from VMEM to DRAM, reusing channel 0
    program.append(Instruction("dma.store.ch<N>", DmaArgs(rd=DRAM_C_ADDR_REG, rs1=VMEM_C_ADDR_REG, rs2=TRANSFER_SIZE_REG, channel=0)))
    # Wait for the final transfer to complete before finishing
    program.append(Instruction("dma.wait.ch<N>", DmaArgs(rd=0, rs1=0, channel=0)))

    return program
```

## 03_silu
```python
from typing import List, Any
from npu_model.software import Instruction
from npu_model.isa import (
    DmaArgs,
    ScalarArgs,
    VectorArgs,
)

def test() -> List[Instruction[Any]]:
    """
    Implements the SiLU activation function: f(x) = x * sigmoid(x)
    where sigmoid(x) = 1 / (1 + exp(-x))
    
    Processes one 32x32 bf16 tile.
    """
    
    # Memory and size constants
    VMEM_BUFFER_BASE = 0x2000
    DRAM_OUTPUT_BASE = 0xb00
    TILE_SIZE_BYTES = 2048
    HALF_TILE_SIZE_BYTES = 1024
    
    # The vload/vstore immediate is a 32-byte chunk index.
    HALF_TILE_VMEM_OFFSET_IMM = HALF_TILE_SIZE_BYTES // 32

    instructions = []

    # 1. Setup scalar registers with addresses and sizes
    instructions.extend([
        # x1 = VMEM base address for the activation/output buffer
        Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=VMEM_BUFFER_BASE)),
        # x3 = DRAM output base address
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=DRAM_OUTPUT_BASE)),
        # x4 = Transfer size for a full tile
        Instruction("addi", ScalarArgs(rd=4, rs1=0, imm=TILE_SIZE_BYTES)),
    ])

    # 2. DMA input activation from DRAM to VMEM
    instructions.extend([
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        # Load from DRAM base 0 (rs1=0) to VMEM base (rd=1)
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=0, rs2=4, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ])

    # --- Process first half-tile (left 32x16) ---

    # Load first half from VMEM into tensor register m0
    instructions.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))

    # Generate constants: 0.0 in m4, 1.0 in m3
    # m4 = m0 - m0. Latency: 2 cycles.
    instructions.extend([
        Instruction("vsub.bf16", VectorArgs(vd=4, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    # m3 = exp(m4) = exp(0.0) = 1.0. Latency: 8 cycles.
    instructions.extend([
        Instruction("vexp.bf16", VectorArgs(vd=3, vs1=4, vs2=0)),
        Instruction("delay", ScalarArgs(imm=8)),
    ])

    # Compute sigmoid(m0) -> m8
    # m5 = m4 - m0 = -x. Latency: 2 cycles.
    instructions.extend([
        Instruction("vsub.bf16", VectorArgs(vd=5, vs1=4, vs2=0)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    # m6 = exp(m5) = exp(-x). Latency: 8 cycles.
    instructions.extend([
        Instruction("vexp.bf16", VectorArgs(vd=6, vs1=5, vs2=0)),
        Instruction("delay", ScalarArgs(imm=8)),
    ])
    # m7 = m3 + m6 = 1.0 + exp(-x). Latency: 2 cycles.
    instructions.extend([
        Instruction("vadd.bf16", VectorArgs(vd=7, vs1=3, vs2=6)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    # m8 = recip(m7) = sigmoid(x). Latency: 8 cycles.
    instructions.extend([
        Instruction("vrecip.bf16", VectorArgs(vd=8, vs1=7, vs2=0)),
        Instruction("delay", ScalarArgs(imm=8)),
    ])
    
    # Compute silu(m0) = m0 * m8 -> m9. Latency: 2 cycles.
    instructions.extend([
        Instruction("vmul.bf16", VectorArgs(vd=9, vs1=0, vs2=8)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    
    # Store result of first half from m9 to VMEM
    instructions.append(Instruction("vstore", VectorArgs(vd=9, rs1=1, imm12=0)))

    # --- Process second half-tile (right 32x16) ---
    
    # Load second half from VMEM into m1
    instructions.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=HALF_TILE_VMEM_OFFSET_IMM)))

    # Compute sigmoid(m1) using existing constants in m3 (1.0) and m4 (0.0)
    # m5 = m4 - m1 = -x. Latency: 2 cycles.
    instructions.extend([
        Instruction("vsub.bf16", VectorArgs(vd=5, vs1=4, vs2=1)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    # m6 = exp(m5) = exp(-x). Latency: 8 cycles.
    instructions.extend([
        Instruction("vexp.bf16", VectorArgs(vd=6, vs1=5, vs2=0)),
        Instruction("delay", ScalarArgs(imm=8)),
    ])
    # m7 = m3 + m6 = 1.0 + exp(-x). Latency: 2 cycles.
    instructions.extend([
        Instruction("vadd.bf16", VectorArgs(vd=7, vs1=3, vs2=6)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])
    # m8 = recip(m7) = sigmoid(x). Latency: 8 cycles.
    instructions.extend([
        Instruction("vrecip.bf16", VectorArgs(vd=8, vs1=7, vs2=0)),
        Instruction("delay", ScalarArgs(imm=8)),
    ])

    # Compute silu(m1) = m1 * m8 -> m9. Latency: 2 cycles.
    instructions.extend([
        Instruction("vmul.bf16", VectorArgs(vd=9, vs1=1, vs2=8)),
        Instruction("delay", ScalarArgs(imm=2)),
    ])

    # Store result of second half from m9 to VMEM
    instructions.append(Instruction("vstore", VectorArgs(vd=9, rs1=1, imm12=HALF_TILE_VMEM_OFFSET_IMM)))

    # --- DMA output result from VMEM to DRAM ---
    instructions.extend([
        # Store from VMEM base (rs1=1) to DRAM output (rd=3)
        Instruction("dma.store.ch<N>", DmaArgs(rd=3, rs1=1, rs2=4, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ])

    return instructions
```

