## Radiance Muon GPU — Hardware Architecture Summary

### Overview and Programming Model

Radiance Muon is a RISC-V SIMT GPU core implementing RV32IM + Zfinx (integer-encoded floating-point). The execution unit is a **threadblock** spanning 2 cores × 4 warps × 16 lanes = **128 threads total**. Each warp is 16 threads wide (`VLEN=16`). The programming interface is `kernel_body(void* raw_arg, uint32_t tid_in_threadblock, uint32_t threads_per_threadblock, uint32_t threadblock_id)`, launched via `mu_schedule(kernel_body, &args, NUM_WARPS=4)`. Thread indexing: `warp_id = tid / 16`, `lane_id = tid % 16`. Warps execute in SIMT fashion; divergent branches serialize across the warp. The ISA supports up to 256 architectural registers per thread (compiler currently limits to 128), enabling aggressive register tiling and ILP. Dynamic warp occupancy scales concurrent warps based on register usage, trading thread-level parallelism for register space.

Synchronization uses `mu_barrier(barrier_id, num_warps)` where `num_warps` is the total warp count across both cores (8 warps for a full threadblock). Barriers must not be placed inside warp-divergent branches. `mu_fence()` / `mu_fence_smem()` enforce memory ordering. Exponentials use `mu_exp(float)` (never `expf`/`exp`).

---

### Memory Hierarchy

**Shared Memory (SMEM):** 128 KiB per cluster, shared across both cores. Base address `0x0000_0000`. Banked 4×16 with 1R1W dual-port per bank; aggregate bandwidth **256 B/cycle** (read + write simultaneously). Addressed via `store_shared` / `load32_shared` / `load16_shared` intrinsics or `__local` pointers. SMEM is the primary staging area for inter-warp data sharing and reuse; it is the highest-bandwidth on-chip memory and should be used aggressively for operand reuse (e.g., tiling A/B matrices in matmul, staging softmax intermediates).

**L0 Instruction Cache (L0i):** 4 KiB per core, direct-mapped, 64 B lines, 8 B/cycle bandwidth, 3-cycle read latency.

**L0 Data Cache (L0d):** 16 KiB per core, direct-mapped, write-through, 64 B lines, 64 B/cycle bandwidth, 3-cycle read latency.

**L1 Cache:** 64 KiB per cluster (unified), 4-way set-associative, write-through, 32 B lines, 32 B/cycle bandwidth. Target hit latency ~8 cycles. Shared across both cores; core accesses are serialized (RR arbitration). Non-blocking with MSHRs.

**L2 Cache:** 512 KiB SoC-level, 8-way, 128 B lines, 32 B/cycle bandwidth (~20-cycle hit latency). Coherent; atomics handled at L2 only.

**DRAM (GPU):** 2 GB, ~100-cycle latency (target), effectively the slowest tier. GPU addresses are transparently remapped (33rd bit appended at L2 egress).

**Memory access pattern guidance:** Coalesced sequential per-lane accesses maximize L0D/L1 utilization. The coalescer merges per-lane global requests into coalesced transactions; stride-1 (consecutive 4-byte words across lanes) achieves maximum coalescing. SMEM staging eliminates redundant global loads for data reused across warps or iterations.

---

### Compute Units

**Integer (INT32) SIMT:** 16 INT PEs per core, one per lane. Throughput: **1 INT32 op/thread/cycle**, 1 warp-instruction/cycle peak IPC. Full 16-wide warp executes in a single cycle.

**Floating-Point (FP32) SIMT:** 8 FP32 PEs per core (half the lane count). Throughput: **0.5 FP32 op/thread/cycle** — a warp-wide FP32 instruction takes 2 cycles. FP16/BF16 throughput: **1 op/thread/cycle** (2× bit-scaling from FP32).

**Special functions:** `mu_exp(float)` for fp32 exponential; `mu_fexp(_Float16)` / `mu_fnexp(_Float16)` for fp16 exp via dedicated instructions. No hardware tensor/matrix instructions in SIMT (mxgemmini is a separate accelerator not available to SIMT kernels).

**Register file:** Physical RF is 16 KiB per core. With 128 regs/thread × 4 B × 16 lanes = 8 KiB per warp; at maximum register usage, 2 warps fit per core. With 32 regs/thread, up to 8 warps fit. More registers per thread → fewer concurrent warps but more ILP; fewer registers → more warps for latency hiding via TLP.

**Operand bandwidth:** RF supports 3 read operands (rs1/rs2/rs3) per instruction; minimum RF bandwidth is 192 B/cycle for conflict-free operation.

---

### Key Optimization Constraints

**FP32 is half-throughput:** FP32 compute is the primary bottleneck at 0.5 FLOP/thread/cycle. Maximizing FP32 utilization requires hiding the 2-cycle FP32 latency with independent instructions (ILP via loop unrolling, register tiling) or warp-level TLP.

**ILP via large register tiles:** With up to 128 architectural registers, kernels should accumulate large per-thread output tiles (e.g., 4–8 FP32 accumulators per thread in matmul) and unroll inner loops to expose independent FP32 operations and hide FU latency.

**SMEM is the bandwidth workhorse:** At 256 B/cycle aggregate, SMEM far exceeds L1 (32 B/cycle) and L0D (64 B/cycle). All reused data (matrix tiles, softmax row statistics, attention intermediates) should be staged in SMEM. SMEM bandwidth provisions 64 B/cycle for SIMT (one word per lane), matching L0D.

**Barrier cost:** `mu_barrier` synchronizes all 8 warps across both cores. Minimize barrier frequency by maximizing work between synchronization points (larger tiles, more unrolling per phase).

**Coalesced global access:** Global loads/stores should be stride-1 across lanes (lane `i` accesses address `base + i*4`) to maximize coalescing. Non-coalesced accesses generate multiple transactions and saturate the L1/L2 bandwidth budget.

**Warp occupancy vs. register pressure:** Kernels with high register usage automatically reduce warp count, reducing TLP. Compensate with ILP (unrolled loops, multiple accumulators). Kernels with low register usage benefit from high warp count to hide memory latency.

**No divergence in barriers:** Never place `mu_barrier` inside conditionals guarded by `tid`-dependent conditions without an explicit `else { asm volatile("nop"); }` workaround, to prevent compiler branch duplication causing deadlocks.