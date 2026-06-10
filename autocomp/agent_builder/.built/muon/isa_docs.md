## Instruction Formats

### R-type

### R-type
R-type instructions will include current RISC-V R-type instructions, as well as R4-type instructions used by Vortex and floating point instructions. In particular, the `funct2` field in R4 will be delegated to the last two bits of `funct7`, with the upper 5 bits set to 0. It has this format:
```
63  60   59   58    52 51 44 43 36 35 28 27 20 19    17 16 9 8     7 6      0
[pred] [resv] [funct7] [rs4] [rs3] [rs2] [rs1] [funct3] [rd] [opext] [opcode]
  4b     1b      7b      8b    8b    8b    8b     3b     8b     2b      7b
```
#### Subtypes
* R5 has `rs1` through `rs4`, as well as `rd`
	* Assembly: `opcode.variant rd, rs1, rs2, rs3, rs4 @ pred`
* R4 has `rs1` through `rs3`, as well as `rd`
	* R4Frm used for floating points, with `frm` as `funct3`
	* Assembly: `opcode.variant rd, rs1, rs2, rs3 @ pred`
* R3 has `rs1` and `rs2`, as well as `rd` (this is what base RISC-V R-type is)
	* R3Atomic for atomic, with `aq`, `rl`, `funct5` as `funct7` and `.global`, `.shared` address space qualifier encoded in opext.
	* R3Frm for floating point
	* Assembly: `opcode.variant rd, rs1, rs2 @ pred`

**Special case for floating point operations.** The floating point rounding mode `frm` will remain located at `funct3`.

---

### I3-type

### I3-type
I3-type instructions will have up to 2 source registers and up to 1 destination register. This leaves 24 bits for immediate, which can be used directly or split across the two source registers as offsets.
```
63  60 59         36 35 28 27 20 19    17 16 9 8     7 6      0
[pred] [ imm[23:0] ] [rs2] [rs1] [funct3] [rd] [opext] [opcode]
  4b        24b        8b    8b     3b     8b     2b      7b
```
#### Subtypes
* I3 has `rs1`, `rs2`, `rd`, and a 24-bit immediate
	* Assembly: `opcode.variant rd, rs1, rs2, imm @ pred` 
* II3 has `rs1`, `rs2`, `rd`, and two 12-bit immediates
	* Assembly: `opcode.variant rd, imm1(rs1), imm2(rs2) @ pred` 

---

### S/SB-type

### S/SB-type
S/SB-type instructions has 2 source registers. The upper 8 bits of the 32-bit immediate is encoded in rd.
```
63  60 59         36 35 28 27 20 19    17 16           9 8     7 6      0
[pred] [ imm[23:0] ] [rs2] [rs1] [funct3] [ imm[31:24] ] [opext] [opcode]
  4b        24b        8b    8b     3b          8b          2b      7b
```
* S and SB both have `rs1`, `rs2`, and a 32-bit immediate
	* S Assembly: `opcode.variant rs2, imm(rs1) @ pred`
	* SB Assembly: `opcode.variant rs1, rs2, label @ pred`

**Special case for SB instructions.** Base RISC-V branch PC offset has implicit bit 0. For Muon, bits 0 to 2 are explicitly encoded as 0s in the machine code (bit 36 to 38), leaving 29 remaining effective bits. The assembly behavior remains the same (immediate is specified in number of bytes), but the specified value gets rounded down to a multiple of 8.

**Address Space Qualifier for Stores** We extend RISC-V `sb/sh/sw` with address space qualifiers `.global`, `.shared`, encoded using opext field as 00 and 01. 

---

### I2-type

### I2-type
I1-type instructions will have 1 source register and 1 destination register. This leaves the full 32bits for immediate. This instruction will replace the `I` and `UJ` type instructions in the original RISC-V specification, and will render `U` type instructions unnecessary.
```
63  60 59         36 35          28 27 20 19    17 16 9 8     7 6      0
[pred] [ imm[23:0] ] [ imm[31:24] ] [rs1] [funct3] [rd] [opext] [opcode]
  4b        24b            8b         8b     3b     8b     2b      7b
```
#### Subtypes
* I2 has `rs1`, `rd`, and a 32-bit immediate (base RISC-V I-type)
	* Assembly: `opcode.variant rd, rs1, imm @ pred`
* U and J both have `rd` and a 32-bit immediate.
	* U should no longer be necessary but is preserved for compatibility.
		* `auipc` will be unnecessary since branch offsets can be 32-bit;
		* `lui` will be unnecessary since the immediate for `addi` can be 32-bit;
	* U Assembly: `opcode.variant rd, imm @ pred`
	* J Assembly: `jal rd, label @ pred`

**Special case for J instruction.** Similar to branches, implicit bit 0 is now explicit zeros for bits 0 to 2.
**Special case for shift-immediate instructions.** Shift amount remains `imm[6:0]`. Shift opcode will still occupy the same bits in the immediate (`[11:7]`); however it will no longer overlap with where `funct7` is in R-types. 
**Special case for CSR instructions.** The CSR source/dest used to be encoded in the imm12 field; it's now 32-bits (nice and wide, as it should be). The CSR immediate used to be encoded in the 5-bit rs1 address; it will still occupy rs1 in Muon but will expand to use 8 bits.
**Address Space Qualifier for Loads** We extend RISC-V `lb/lh/lw` with address space qualifiers `.global`, `.shared`, encoded using opext field as 00 and 01. 

## Control Flow and SIMT Divergence

### vx_split

#### `vx_split`

```
vx_split  rd, rs1, rs2
```

Used jointly with `vx_join` to serialize execution of divergent branches.

* `rs1`: Per-lane condition value.  `vx_split` will set the current tmask as
high for lanes that has non-zero `rs1` values, and push the tmask set high for
lanes with zero `rs1` values to the IPDOM stack.  Original tmask is preserved,
i.e. lanes not active before entering `vx_split` will not be made active in
either tmask.
* If `rs2` is set, use the `vx_split_n` variant, i.e. modify the current tmask
to lanes with zero `rs1` values, and push the non-zero `rs1` tmask to the
stack.
* `rd` is unused.  This *deviates from Vortex* where `rd` is set to 1 if the
condition is divergent, e.g. the value of `rs1` disagrees between
currently-enabled lanes.

Note that `vx_split` pushes two tmasks to the IPDOM stack, in this order:
(1) the original tmask that restores the state before entering the instruction,
and (2) the diverged tmask that will be executed later.  When pushing (2), the
PC of the subsequent instruction after `vx_split` will also be pushed as part
of the stack entry. When pushing (1), no valid PC will be pushed to the stack.

These entries will be consumed by two later executions of `vx_join`, where the
first one pops the diverged tmask + PC and initiates the pipeline to traverse
the "else-path". The second one will pop the restored tmask, but since PC entry
is invalid, execution PC is *not* altered, allowing the program to proceed to
the branch exit.

Note that for non-divergent branches, we skip pushing the (2) entry to the
IPDOM stack.  Since the (1) entry does not alter PC and therefore prevents
the second execution of `vx_join`, communicating the result of branch
divergence from `vx_split` to `vx_join` via `rd` (so that `vx_join` can
conditionally avoid popping from stack) becomes unnecessary.


---

### vx_join


```
vx_join  rs1
```

Pops `(tmask, optional PC)` entry from the IPDOM stack and modifies the
architectural state accordingly.  Note it may not alter the PC depending on
whether the corresponding `vx_split` pushed a valid PC or not.

`rs1` is unused, which *deviates from Vortex* where a zero `rs1` value will
indicate that the corresponding `vx_split` was non-divergent.
Instead, we skip pushing the non-divergent tmask/PC to the IPDOM stack,
eliminating the need for conditional stack-popping in `vx_join`.  See
`vx_split` for details.


---

### vx_tmc

#### `vx_tmc`

```
vx_tmc  rs1
```

* Set tmask to the value of `rs1` of the "leader" lane, i.e. the left-most
active lane.


---

### vx_pred

#### `vx_pred`

```
vx_pred  rd, rs1, rs2
```

* `rs1`: Per-lane condition value.  `vx_pred` will set the current tmask
as high for lanes that has non-zero `rs1` values.  Original tmask is
preserved, i.e. lanes not active before entering `vx_pred` will not be made
active in the tmask.
* If the *address* of `rd` is non-zero, use the `vx_pred_n` variant, i.e. tmask
is set for lanes with zero `rs1` values.  Nothing is written back to `rd`.
* If no lanes have a non-zero `rs1` value (zero for `vx_pred_n`), set tmask
to the value of `rs2` of the "leader" lane, i.e. the left-most active lane.

---

### vx_wspawn

#### `vx_wspawn`

```
vx_spawn rs1, rs2
```

Activates warps `[0, rs1)`, except the warp that executed this instruction, and
sets their `PC` to `rs2`.  Sets tmask of newly active warps to all 1's.
The values of `rs1` and `rs2` are taken from the "leader" lane.

Note: `vx_wspawn` eventually be superceded by command processor's scheduling
capabilities.


---

### vx_bar


```
vx_bar rs1, rs2
```

Waits for `rs2` warps in a single cluster to reach barrier with id given by `rs1`. 
The values of `rs1` and `rs2` are taken from "leader" lane. TODO: how many barriers? 16?

Note: `vx_bar` will eventually be superceded by neutrino / command processor barrier mechanism

---

### mu_barrier

/** NOTE about barriers: Placing barriers around thread-divergent branches
 *  may cause bugs.  The compiler might decide to duplicate mu_barrier() into
 *  both paths of a warp-divergent branch, which will cause the barrier to
 *  execute twice via SIMT serialization, and cause potential deadlocks.
 *  mu_barrier() doesn't check for participating tmasks.
 *
 *  We wrap mu_barrier() with convergent/noduplicate/noinline, but that doesn't
 *  seem to be sufficient.
 *
 *  This seems to happen the most around single-thread-guarded code, e.g.:
 *
 *    if (tid == 0) {
 *        // do something
 *    }
 *    mu_barrier(...);
 *
 *  A workaround that _may_ work is to put an explicit else clause with a
 *  nop in it:
 *
 *    if (tid == 0) {
 *        // do something
 *    } else {
 *        asm volatile("nop");
 *    }
 *    mu_barrier(...);
 *
 *  Another workaround is to use -Os for the optimization, which keeps
 *  the compiler from branch-duplicating to save code size.
 *
 *  None of these workarounds are fundamental, and we need proper compiler
 *  support to reason about warp-convergence.  TODO.
 */
__attribute__((convergent))
static void mu_barrier(unsigned barried_id, unsigned num_warps) {
    asm volatile ("vx_bar %0, %1" :: "r"(barried_id), "r"(num_warps) : "memory");
}

### mu_barrier

- Barriers: `mu_barrier(id, num_warps)` synchronizes all warps of the threadblock
  (8 warps total when NUM_WARPS=4 since two cores).

## Memory Operations

### fence.{i,d,s}

#### `fence.{i,d,s}`
`fence.i` flushes the LSU global memory queues and the L0i cache.

`fence.d` flushes the LSU global memory queues and the L0d cache.

`fence.s` flushes the LSU shared memory queues. 

All three instructions are blocking, and essentially have `seqcst` semantics.

---

### mu_fence

__attribute__((convergent))
inline void mu_fence() {
    asm volatile ("fence" ::: "memory");
}

---

### mu_fence_smem

__attribute__((convergent))
inline void mu_fence_smem() {
    asm volatile ("fence.s" ::: "memory");
}

---

### store_shared

inline void store_shared(uint32_t base, uint32_t offset, uint32_t data) {
    asm volatile("sw.shared %2, %1(%0)" :: "r"(base), "I"(offset), "r"(data)
                 : "memory");
}

---

### store64_shared

inline void store64_shared(uint32_t base, uint32_t offset, uint64_t data) {
    uint32_t lo = static_cast<uint32_t>(data);
    uint32_t hi = static_cast<uint32_t>(data >> 32);
    store_shared(base, offset,     lo);
    store_shared(base, offset + 4, hi);
}

---

### store_shared_from_global

inline void store_shared_from_global(uint32_t dst, uint32_t src) {
    uint32_t data;
    asm volatile("lw.global %0, 0(%1)" : "=r"(data) : "r"(src) : "memory");
    asm volatile("sw.shared %1, 0(%0)" :: "r"(dst), "r"(data) : "memory");
}

---

### load16_shared (uint32_t address)

inline uint16_t load16_shared(uint32_t address) {
    uint16_t data;
    asm volatile("lh.shared %0, %1(%2)" : "=r"(data) : "I"(0), "r"(address)
                 : "memory");
    return data;
}

---

### load32_shared

inline uint32_t load32_shared(uint32_t address) {
    uint32_t data;
    asm volatile("lw.shared %0, %1(%2)" : "=r"(data) : "I"(0), "r"(address)
                 : "memory");
    return data;
}

---

### load16_shared (const T *address)

template <typename T>
inline std::remove_cv_t<T> load16_shared(const T *address) {
    // need bit_cast to re-interpret uint16_t bits as _Float16
    using U = std::remove_cv_t<T>;
    static_assert(sizeof(U) == sizeof(uint16_t), "load16_shared<T*> expects 16-bit T");
    uint16_t bits = load16_shared(reinterpret_cast<uint32_t>(address));
    return __builtin_bit_cast(U, bits);
}

## Data Types and Constants

### as_bf16

static inline _Float16 as_bf16(uint16_t bits) {
  return __builtin_bit_cast(_Float16, bits);
}

---

### bf16x2

struct bf16x2 { _Float16 lo, hi; };

---

### unpack_bf16x2

static inline bf16x2 unpack_bf16x2(uint32_t packed) {
  return { as_bf16((uint16_t)packed), as_bf16((uint16_t)(packed >> 16)) };
}

---

### pack_bf16x2

static inline uint32_t pack_bf16x2(_Float16 lo, _Float16 hi) {
  return (uint32_t)__builtin_bit_cast(uint16_t, lo)
       | ((uint32_t)__builtin_bit_cast(uint16_t, hi) << 16);
}

---

### ONE_BF16_BITS / NEG_INF_BF16_BITS / as_bf16 usage note

// You need to use __builtin_bit_cast(_Float16, ONE_BF16_BITS) for the compiler to correctly emit it.
// use as_bf16 to quickly convert
#define ONE_BF16_BITS ((uint16_t)0x3f80)
#define NEG_INF_BF16_BITS ((uint16_t) 0xFF80)

## Math and Arithmetic

### mu_fexp

inline _Float16 mu_fexp(_Float16 arg) {
    _Float16 output;
    asm volatile("fexp.h %0, %1" : "=r"(output) : "r"(arg));
    return output;
}

---

### mu_fnexp

inline _Float16 mu_fnexp(_Float16 arg) {
    _Float16 output;
    asm volatile("fnexp.h %0, %1" : "=r"(output) : "r"(arg));
    return output;
}

---

### mu_exp

- For exponentials use the harness-provided `mu_exp(float)`; never expf/exp.
  Sigmoid: `1/(1 + mu_exp(-x))`.

## Hardware Configuration and Intrinsics

### mu_num_threads

// This compiles to CSR reads which stalls the pipeline. Use sparingly & cache.
inline int mu_num_threads() {
    return vx_num_threads();
}

---

### MU_NUM_THREADS / MU_NUM_WARPS / MU_NUM_CORES / MU_BLOCK_SIZE hardware config macros

// This hard-codes hardware config into kernel, but this allows efficient
// compile-time unrolling and constant propagation.
#define MU_NUM_THREADS 16
#define MU_NUM_WARPS 8
#define MU_NUM_CORES 2
#define MU_NUM_MAX_WARPS 8
#define MU_NUM_CLUSTERS 1
#define MU_BLOCK_NUM_WARPS(n) (MU_NUM_CORES * (n))
#define MU_BLOCK_SIZE(n) (MU_BLOCK_NUM_WARPS(n) * MU_NUM_THREADS)
#define MU_DOUBLE_BLOCK_SIZE(n) (MU_BLOCK_SIZE(n) * 2)

## Kernel Interface and Harness

### kernel_body

Every generated kernel must define exactly:

```cpp
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
);
```

The harness launches it with `mu_schedule(kernel_body, &kernel_args, NUM_WARPS)`:
one threadblock spanning MU_NUM_CORES=2 cores x NUM_WARPS=4 warps x 16 lanes = 128
threads. `tid_in_threadblock` is 0..127; warp id = tid / 16; lane id = tid % 16.

---

### KernelArgs

The kernel receives a `KernelArgs*` (problem-specific struct defined in the harness)
with `__global float*` buffers and shape fields. Use only these buffers.