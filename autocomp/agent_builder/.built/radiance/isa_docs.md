## Data Types and Constants

### GemmDatatype

enum class GemmDatatype : uint8_t {
    FP8,
    FP6,
    FP4,
};

---

### GemmConfig

struct GemmConfig {
    uint32_t TILE_M = 128;
    uint32_t TILE_N = 128;
    uint32_t TILE_K = 256;
    GemmDatatype DATATYPE = GemmDatatype::FP8;
    // quantize output to fp4/fp6/fp8
    bool QUANT_OUTPUT = false;

    constexpr bool IS_FP8() const { return DATATYPE == GemmDatatype::FP8; }
    constexpr uint32_t PE_M() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_N() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_K() const { return 16; }
    constexpr uint32_t PE_TILES_I() const { return TILE_M / PE_M(); }
    constexpr uint32_t PE_TILES_J() const { return TILE_N / PE_N(); }
    constexpr uint32_t PE_TILES_K() const { return TILE_K / PE_K(); }
    // TODO: TILE_N not differentiated
    constexpr uint32_t SCALE_FACTORS_PER_TILE() const { return TILE_M * TILE_K / 32; }
    constexpr uint32_t VALUES_PER_BYTE() const { return (IS_FP8() ? 1 : 2); }
    // Size of each C element *after column-packing*.
    constexpr uint32_t OUT_ELEM_SIZE() const {
        // C FP4/FP6 elem-packing is along the M dimension, not N
        return (QUANT_OUTPUT ? sizeof(uint8_t) : sizeof(uint16_t));
    }
    constexpr uint32_t TILE_M_QUANT() const {
        return (QUANT_OUTPUT ? TILE_M / VALUES_PER_BYTE() : TILE_M);
    }
    constexpr uint32_t TILE_N_QUANT() const {
        // packing of N-dimension is already reflected in OUT_ELEM_SIZE()
        return TILE_N;
    }
    constexpr bool USE_LUT() const { return DATATYPE == GemmDatatype::FP6; }
};

### GemmConfig

`copy_smem_to_gmem_simt`, vectorized 32-bit stores). `GemmConfig` levers: TILE_M/TILE_N/
TILE_K (TILE_K multiple of 32), DATATYPE (FP8/FP6/FP4), QUANT_OUTPUT.

---

### bf16x2

struct bf16x2 { _Float16 lo, hi; };

---

### as_bf16

static inline _Float16 as_bf16(uint16_t bits) {
  return __builtin_bit_cast(_Float16, bits);
}

---

### MX_NUM_WARPS

## Register constraint

`mxgemm_lib` is heavily inlined. `NUM_WARPS` is already 8 in `VX_config.h`; launching 8
warps' worth of live registers blows the 256-entry physical register file (RTL Rename
assert / cyclotron `globalOverSubscription`). Use `MX_NUM_WARPS = 2` — do NOT reuse the
name `NUM_WARPS`.

---

### RadianceSingleClusterConfig

## Config

`RadianceSingleClusterConfig` = `WithRadianceMxGemmini(InCluster(0), dim=16)` +
`WithMuonCores(2)`. One threadblock = 2 cores × 2 warps (`MX_NUM_WARPS`, see registers) ×
16 lanes. DIM = 16 (systolic array is 16×16). SMEM = BANK_NUM(4) × BANK_ROWS(2048) rows ×
16 B/row = 128 KiB. A scratchpad *row address* r maps to SMEM **byte** `r * DIM`.

---

### Tiling and DIM constants

DIM = 16 (the systolic array / tile dimension). Tiling counts are
`tiles_I = M/DIM`, `tiles_J = N/DIM`, `tiles_K = K/DIM`. `BANK_NUM`/`BANK_ROWS`
come from `gemmini_params.h`.

---

### Supported MX formats

## Supported MX formats

| Format | Enc | Notes |
|--------|-----|-------|
| FP4 (E2M1) | 4-bit | 2 elements packed (dual) |
| FP6 E2M3 | 6-bit | single |
| FP6 E3M2 | 6-bit | dual |
| FP8 E4M3 | 8-bit | single — the most common input format |
| FP8 E5M2 | 8-bit | dual |

Each MX block shares one `fpe8m0` (power-of-two) scale factor. Scale factors are
grouped: the golden headers expose `A_scales_row[GK][M]` and `B_scales_col[GK][N]`
where `GK` is the number of scale groups along K.


---

### KernelArgs

The kernel receives a `KernelArgs*` (problem-specific struct defined in the harness)
with `__global float*` buffers and shape fields. Use only these buffers.

## Control Flow and Synchronization

### gemmini_flush

#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

---

### gemmini_fence

#undef gemmini_fence
inline void gemmini_fence() {
    while (load32_shared(GEMMINI_BUSY_ADDR) != 0) {
        asm volatile("nop");
    }
}

### gemmini_fence

`gemmini_fence()` spins on BUSY until the array drains (blocking). `gemmini_fence_ready()`

### gemmini_fence

#define gemmini_fence() asm volatile("fence")

---

### gemmini_fence_ready

`gemmini_fence()` spins on BUSY until the array drains (blocking). `gemmini_fence_ready()`
waits only until the unit can accept the *next* command — use it to overlap SIMT work
(scale staging, previous tile's C move-out, next-tile DMA) underneath accelerator compute.

### gemmini_fence_ready

// spin-lock until MMIO interface is ready to accept new commands, instead of
// waiting for all previous commands to complete
inline void gemmini_fence_ready() {
    while (load32_shared(GEMMINI_READY_ADDR) == 0) {
        asm volatile("nop");
    }
}

---

### gemmini_fence_waitcount

inline void gemmini_fence_waitcount(const int n) {
    while (load32_shared(GEMMINI_OCCUPANCY_ADDR) > n) {
        asm volatile("nop");
    }
}

---

### vx_tmc

#### `vx_tmc`

```
vx_tmc  rs1
```

* Set tmask to the value of `rs1` of the "leader" lane, i.e. the left-most
active lane.


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

---

### mu_num_threads

inline int mu_num_threads() {
    return vx_num_threads();
}

---

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

## MMIO and RoCC Command Interface

### MMIO command protocol / GEMMINI_CTRL

## MMIO command protocol

`GEMMINI_CTRL = 0x00084000` in the GPU-local **shared** space (see `mxgemmini_mmio.h`).
To issue a RoCC command, a SIMT thread does three shared stores:

```
store rs1 -> GEMMINI_CTRL + 0x10        (RS1)
store rs2 -> GEMMINI_CTRL + 0x18        (RS2)
store inst-> GEMMINI_CTRL + 0x00        (INST word triggers the command)
   inst = 0x7B | (funct << 25)
```

Only the INST-word store fires the command; RS1/RS2 just latch. Status registers (read to
poll / fence):

```
READY     @ +0x08   (accepts next command)
BUSY      @ +0x20   (array still draining)
OCCUPANCY @ +0x28   (queue depth)
```

`gemmini_fence()` spins on BUSY until the array drains (blocking). `gemmini_fence_ready()`
waits only until the unit can accept the *next* command — use it to overlap SIMT work
(scale staging, previous tile's C move-out, next-tile DMA) underneath accelerator compute.

---

### RoCC funct table

## RoCC funct table (the ones the MX datapath uses)

| funct | name | purpose |
|------:|------|---------|
| 0  | CONFIG_EX          | set A/B/C formats (fp8/fp6/fp4/full), uselut |
| 7  | FLUSH              | reset accelerator state between matmuls (clears the C accumulator) |
| 8  | LOOP_WS            | run the loop FSM (DMA and/or compute per the `skips` bits) |
| 9  | LOOP_WS_CONFIG_BOUNDS | PE-tile loop bounds I/J/K |
| 10 | LOOP_WS_CONFIG_ADDRS_AB | A/B DRAM base addresses |
| 12 | LOOP_WS_CONFIG_STRIDES_AB | A/B DRAM row strides |
| 24 | LOOP_WS_CONFIG_SPAD_AB | A/B scratchpad (SMEM) tile addresses |
| 25 | LOOP_WS_CONFIG_SPAD_C  | C scratchpad address (rs2[63:32]) |
| 26 | MXQUANT_CONFIG_MVOUT | output requant + **scale double-buffer selects** |
| 27 | MX_LOAD_SCALES     | (DRAM path only; the Radiance kernel stages scales via SIMT stores instead) |
| 28 | MX_READ_SMEM       | drain the bf16 C accumulator from SMEM to DRAM |
| 29 | MX_LOAD_LUT        | FP6 dequant LUTs |

`skips` field of LOOP_WS rs2 (bits, `loop_matmul_skips`): lda, ldb, ldd, ex, stc. The
Radiance kernel issues LOOP_WS **twice per K-tile**: first **load-only** (`skip_ex|skip_stc`)
so the accelerator DMAs A/B tiles DRAM→scratchpad, then **compute-only**
(`skip_lda|skip_ldb`) to run the systolic array and bf16-accumulate C in SMEM. The
`ex_accumulate` bit (set for every K-tile after the first) makes compute ADD into the
existing C rather than overwrite — this is how K is reduced across multiple K-tiles.

---

### ROCC_INSTRUCTION_RS1_RS2

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET, gemmini_arg_to_u64(rs1)); \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET, gemmini_arg_to_u64(rs2)); \
    store_shared  (GEMMINI_CTRL, GEMMINI_INST_OFFSET, (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25)); \
}

---

### gemmini_arg_to_u64

template <typename T>
inline uint64_t gemmini_arg_to_u64(T value) {
    if constexpr (std::is_pointer_v<T>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
    } else {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T>,
                      "Gemmini argument must be integral/enum or pointer");
        return static_cast<uint64_t>(value);
    }
}

---

### gemmini_status

#define gemmini_status() ({uint32_t status; asm volatile ("csrr %0, 0xacc" : "=r" (status)); status;})

## Accelerator Configuration

### gemmini_config_ld

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

---

### gemmini_extended_config_ld

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

---

### gemmini_extended2_config_ld

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(stride, scale, shrunk, 0)

---

### gemmini_extended3_config_ld

#define gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

---

### gemmini_extended4_config_ld

#define gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

---

### gemmini_extended5_config_ld

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

---

### gemmini_config_st

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

---

### gemmini_extended_config_st

#define gemmini_extended_config_st(stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

---

### gemmini_extended2_config_st

#define gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

---

### gemmini_extended3_config_ex

// RS1: [63:32] acc_scale | [31:16] a_stride | [15:14] out_mx_fmt | [13:12] wgt_mx_fmt | [11:10] act_mx_fmt | [9] b_transpose | [8] a_transpose | [7] set_only_strides | [6] spacer | [5] uselut | [4:3] activation | [2] dataflow | [1:0] cmd_type
// RS2: [63:48] c_stride | [47:32] relu6_shift | [31:0] in_shift
#define gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides, act_mx_fmt, wgt_mx_fmt, out_mx_fmt, uselut) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
        ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | \
        ((uint64_t)(A_stride) << 16) | \
        ((uint64_t)(out_mx_fmt) << 14) | \
        ((uint64_t)(wgt_mx_fmt) << 12) | \
        ((uint64_t)(act_mx_fmt) << 10) | \
        ((uint64_t)(B_transpose) << 9) | \
        ((uint64_t)(A_transpose) << 8) | \
        ((uint64_t)(set_only_strides) << 7) | \
        ((uint64_t)(uselut) << 5) | \
        ((uint64_t)(sys_act) << 3) | \
        ((uint64_t)(dataflow) << 2) | \
        CONFIG_EX, \
        ((uint64_t)(C_stride) << 48) | (uint64_t)(uint32_t)(sys_shift), \
        k_CONFIG)

### gemmini_extended3_config_ex

`gemmini_extended3_config_ex` now packs four MX format fields into
previously-unused bits of `rs1`; the macro signature changed to take
`(act_fmt, wgt_fmt, out_fmt, uselut)` as the last four arguments. The
spike handler reads:

| `rs1` bits | Field | Meaning |
|------------|-------|---------|
| `[5]`     | `uselut`  | When 1, the MX kernel takes the FP6 LUT path. |
| `[11:10]` | `act_fmt` | A-side / activation format: 0 = FP8 E4M3, 1 = FP6 E3M2 (LUT-indexed), 2 = FP4 E2M1. Selects the matmul kernel. |
| `[13:12]` | `wgt_fmt` | B-side / weight format (same encoding). |
| `[15:14]` | `out_fmt` | Output format: 0 = FP8 E4M3 packed, 1 = FP6 (4-bit LUT indices), 2 = FP4 E2M1 packed, 3 = BF16. Selects whether a requant post-pass runs and which projection it uses. |

All other config-EX bits (dataflow, sys_act, strides, transposes, sys
shifts) keep their pre-existing meanings.

### gemmini_extended3_config_ex

1. **Configure execution** — set dataflow + the MX format fields:
   ```c
   gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
       /*C_stride*/1, /*A_stride*/1, /*A_transpose*/0, /*B_transpose*/0,
       /*set_only_strides*/false,
       /*act_mx_fmt*/0, /*wgt_mx_fmt*/0, /*out_mx_fmt*/3, /*uselut*/0);
   ```
   `act_mx_fmt`/`wgt_mx_fmt`/`out_mx_fmt` are 2-bit fields (config_ex RS1 bits
   [11:10],[13:12],[15:14]). For fp8 the example uses act=0,wgt=0,out=3 (BF16
   output). **For fp6/fp4 use the values shown in the corresponding
   `matmul_tiled_fp6_*.c` / `matmul_tiled_fp4_*.c` example — do not guess the
   encoding.** Set `uselut=1` and load a LUT (below) when the format requires it.

---

### CONFIG_EX MX format fields

### CONFIG_EX changes (funct 0, CONFIG_EX subtype, `rs1[1:0] == 0b00`)

`gemmini_extended3_config_ex` now packs four MX format fields into
previously-unused bits of `rs1`; the macro signature changed to take
`(act_fmt, wgt_fmt, out_fmt, uselut)` as the last four arguments. The
spike handler reads:

| `rs1` bits | Field | Meaning |
|------------|-------|---------|
| `[5]`     | `uselut`  | When 1, the MX kernel takes the FP6 LUT path. |
| `[11:10]` | `act_fmt` | A-side / activation format: 0 = FP8 E4M3, 1 = FP6 E3M2 (LUT-indexed), 2 = FP4 E2M1. Selects the matmul kernel. |
| `[13:12]` | `wgt_fmt` | B-side / weight format (same encoding). |
| `[15:14]` | `out_fmt` | Output format: 0 = FP8 E4M3 packed, 1 = FP6 (4-bit LUT indices), 2 = FP4 E2M1 packed, 3 = BF16. Selects whether a requant post-pass runs and which projection it uses. |

All other config-EX bits (dataflow, sys_act, strides, transposes, sys
shifts) keep their pre-existing meanings.

---

### gemmini_state_t MX fields

### State added to `gemmini_state_t`

`mx_act_fmt`, `mx_wgt_fmt`, `mx_out_fmt`, `mx_use_lut`,
`mx_lut_update_granularity`, `mx_scale_dram`, `mx_tiles_{I,J,K}`,
`mx_scale_{act,wgt}_sel`, `mx_loop_{a,b,c}_spad`, `mx_loop_skips`,
`mx_loop_spad_marker`, plus the buffers `mx_scale_a_mem`,
`mx_scale_b_mem`, `mx_smem`, and per-format LUTs `mx_lut_{a,b,c}` (each
16 codes × max LUTs). All cleared on `reset()`.


---

### gemmini_mxquant_config_mvout

#define gemmini_mxquant_config_mvout(dram_addr, i_bound, j_bound, k_bound, scale_act_sel, scale_w_sel, lut_update_granularity) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
   ((uint64_t)(scale_w_sel) << 61) | ((uint64_t)(scale_act_sel) << 60) |  ((uint64_t)(k_bound) << 51) | ((uint64_t)(j_bound) << 42) | ((uint64_t)(i_bound) << 33) | (uint64_t)(dram_addr), \
    (uint64_t)(lut_update_granularity) & 0xFFFF, CONFIG_SCALE_MEM)

### gemmini_mxquant_config_mvout

5. **Configure the requantized mvout** — programs the requantizer + output scale
   memory and bounds:
   ```c
   gemmini_config_st(OUT_COLS * sizeof(out_t));
   gemmini_mxquant_config_mvout((uint64_t)scale_factors,
       /*i_bound*/tiles_I, /*j_bound*/tiles_J, /*k_bound*/tiles_K,
       /*scale_act_sel*/0, /*scale_w_sel*/0, /*lut_update_granularity*/1);
   ```

### gemmini_mxquant_config_mvout

(`GEMMINI_SF_MEM_BUFFER_OFFSET`), and the accelerator is told which buffer to read via the
`scale_act_sel`/`scale_wgt_sel` bits (rs1[60]/rs1[61]) of `gemmini_mxquant_config_mvout`.

---

### MXQUANT_CONFIG_MVOUT

| 26 | `MXQUANT_CONFIG_MVOUT`    | `[32:0]` = `scale_dram` (low 33 bits), `[33:42]` = `tiles_I`, `[42:51]` = `tiles_J`, `[51:60]` = `tiles_K`, `[60]` = `scale_act_sel`, `[61]` = `scale_wgt_sel` | `[15:0]` = `lut_update_granularity` (G) | Configures the requant write-back. Per-row, per-N-group e8m0 scale codes are stored at `scale_dram + m*N_blocks + bi`. |

---

### loop_matmul_skips

`skips` field of LOOP_WS rs2 (bits, `loop_matmul_skips`): lda, ldb, ldd, ex, stc. The
Radiance kernel issues LOOP_WS **twice per K-tile**: first **load-only** (`skip_ex|skip_stc`)
so the accelerator DMAs A/B tiles DRAM→scratchpad, then **compute-only**
(`skip_lda|skip_ldb`) to run the systolic array and bf16-accumulate C in SMEM. The
`ex_accumulate` bit (set for every K-tile after the first) makes compute ADD into the
existing C rather than overwrite — this is how K is reduced across multiple K-tiles.

### loop_matmul_skips

#define loop_matmul_skips(skip_lda, skip_ldb, skip_ldd, skip_ex, skip_stc) \
  (((skip_lda) | ((skip_ldb) << 1) | ((skip_ldd) << 2) | ((skip_ex) << 3) | ((skip_stc) << 4)) << 3)

---

### gemmini_loop_ws_spad skips parameter

- **Using `gemmini_loop_ws_spad`** (a fused hardware loop) vs hand-rolled
  preload/compute sequences.
- **Skips / flags** in `loop_ws_spad` (last arg is a skip bitmask, e.g. 0x38).
- Avoiding redundant scale/LUT reloads across tiles.

---

### LOOP_WS_CONFIG_SPAD_AB

| 24 | `LOOP_WS_CONFIG_SPAD_AB`  | A spad base address | B spad end address | Sets `mx_loop_a_spad`, `mx_loop_b_spad`. Also marks the next `LOOP_WS` as the MX (`mx_loop_ws_spad`) variant. |

---

### LOOP_WS_CONFIG_SPAD_C

| 25 | `LOOP_WS_CONFIG_SPAD_C`   | C spad base address | – | Sets `mx_loop_c_spad`. |

---

### gemmini_loop_ws_config_bounds

#define gemmini_loop_ws_config_bounds(I, J, K, pad_I, pad_J, pad_K) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
  }

## Shared Memory Access

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

### load16_shared

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

### load16_shared<T*>

template <typename T>
inline std::remove_cv_t<T> load16_shared(const T *address) {
    // need bit_cast to re-interpret uint16_t bits as _Float16
    using U = std::remove_cv_t<T>;
    static_assert(sizeof(U) == sizeof(uint16_t), "load16_shared<T*> expects 16-bit T");
    uint16_t bits = load16_shared(reinterpret_cast<uint32_t>(address));
    return __builtin_bit_cast(U, bits);
}

## DMA and Data Movement

### gemmini_mvin

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

---

### gemmini_extended_mvin

4. **MVIN tiles** of A and B into the scratchpad. Tiles are DIM×DIM (DIM=16):
   ```c
   gemmini_config_ld(MATMUL_M * sizeof(elem_t));
   gemmini_extended_mvin(dram_ptr, spad_addr, DIM, DIM);
   ```
   Lay A tiles at `a_base + (i*tiles_K + k)*DIM`, B tiles at
   `b_base + (j*tiles_K + k)*DIM` where `b_base = BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM`.

### gemmini_extended_mvin

#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

---

### gemmini_extended_mvin2

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

---

### gemmini_extended_mvin3

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

---

### gemmini_block_mvin

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

---

### gemmini_mvout

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

---

### gemmini_extended_mvout

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

---

### gemmini_mvout_spad

#define gemmini_mvout_spad(dst_addr, src_addr) \
  gemmini_extended_mvout_spad(dst_addr, 1, src_addr, DIM, DIM)

---

### gemmini_extended_mvout_spad

#define gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

---

### copy_smem_to_gmem_simt

/** Move tensor data from SMEM->GMEM using SIMT threads.
 *  Assumes row-major, packed layout (row stride == dim_col) for both src and dest.
 *  TODO: De-dup with FlashAttention */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_smem_to_gmem_simt(const __shared uint8_t *src_smem,
                                   uint8_t *dest_gmem,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock) {
    asm volatile("copy_smem_to_gmem_simt_start_%=:" ::);

    // Thread mapping: All warps in a threadblock cooperatively copies a
    // contiguous chunk of the same size as the threadblock per every "wave".

    // Vectorize to 32-bit words for better throughput.
    auto *src_smem_vec = reinterpret_cast<const __shared uint32_t *>(src_smem);
    auto *dest_gmem_vec = reinterpret_cast<uint32_t *>(dest_gmem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint32_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint32_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock)*i + tid_in_threadblock;
        const auto smem_addr = src_smem_vec + index;
        auto gmem_addr = dest_gmem_vec + index;
        *gmem_addr = *smem_addr;
    }

    asm volatile ("copy_smem_to_gmem_simt_end_%=:" :: );
}

### copy_smem_to_gmem_simt

`SPAD_DEST=256` i.e. SMEM byte 4096) → `gemmini_fence` → SIMT move-out of C (SMEM→GMEM,
`copy_smem_to_gmem_simt`, vectorized 32-bit stores). `GemmConfig` levers: TILE_M/TILE_N/
TILE_K (TILE_K multiple of 32), DATATYPE (FP8/FP6/FP4), QUANT_OUTPUT.

---

### copy_gmem_to_smem_simt_bf16

/** Copy tensor data from GMEM->SMEM using SIMT threads.
 *  Does 16-bit writes in order to comply with requantizer memory interface. */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_gmem_to_smem_simt_bf16(
    const uint8_t *src_gmem, __shared uint8_t *dest_smem,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock) {
    asm volatile("copy_gmem_to_smem_simt_bf16_start_%=:" ::);

    // TODO: dedup with copy_smem_to_gmem_simt

    // Vectorize to 16-bit words
    auto *src_gmem_vec = reinterpret_cast<const uint16_t *>(src_gmem);
    auto *dest_smem_vec = reinterpret_cast<__shared uint16_t *>(dest_smem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint16_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint16_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock)*i + tid_in_threadblock;
        const auto src_addr = src_gmem_vec + index;
        auto dst_addr = dest_smem_vec + index;
        *dst_addr = *src_addr;
    }

    asm volatile("copy_gmem_to_smem_simt_bf16_end_%=:" ::);
}

---

### copy_C_smem_to_gmem_dma_sync

/** Move C result tensor from SMEM->GMEM using Gemmini DMA.
 *  `src_spad_addr` is in scratchpad row address.
 *  This call blocks and synchronizes with the completion of the DMA. */
template <GemmConfig C>
static void copy_C_smem_to_gmem_dma_sync(const uint32_t src_spad_addr,
                                         uint8_t *dest_gmem,
                                         const uint32_t dim_n,
                                         const uint32_t tid_in_threadblock) {
    asm volatile("copy_smem_to_gmem_dma_sync_start_%=:" ::);

    if (tid_in_threadblock == 0) {
        for (int i = 0; i < C.PE_TILES_I(); i++) {
#pragma unroll 32
            for (int j = 0; j < 2 * C.PE_TILES_J(); j++) {
                const uint32_t tile_spad_addr =
                    src_spad_addr + (i * 2 * C.PE_TILES_J() + j) * DIM;
                // row-major layout
                // TODO: DRAM stride is wrong for re-quantized output
                uint8_t *dram_ptr =
                    dest_gmem + (i * 2 * DIM * dim_n + j * DIM) * C.OUT_ELEM_SIZE();
                gemmini_mvout(rad_device_to_host_address(
                                  reinterpret_cast<uint32_t>(dram_ptr)),
                              tile_spad_addr);
            }
        }

        gemmini_fence();
    }

    asm volatile("copy_smem_to_gmem_dma_sync_end_%=:" ::);
}

---

### copy_accmem_to_gmem_dma_sync

/** Move tensor data from AccMEM->GMEM using Gemmini DMA.
 *  This call blocks and synchronizes with the completion of the DMA. */
template <GemmConfig C>
static void copy_accmem_to_gmem_dma_sync(uint8_t *dest_gmem,
                                         const uint32_t dim_n,
                                         const uint32_t tid_in_threadblock) {
    asm volatile("copy_accmem_to_gmem_dma_sync_start_%=:" ::);

    if (tid_in_threadblock == 0) {
        for (int i = 0; i < C.PE_TILES_I(); i++) {
#pragma unroll 32
            // need 4 because 4 fit in accmem row
            for (int j = 0; j < C.PE_TILES_J() / 4; j++) {
                const uint32_t tile_acc_addr =
                    GEMMINI_ACC_ADDR + (i * C.PE_TILES_J() / 4 + j) * DIM;
                // row-major layout
                // TODO: DRAM stride is wrong for re-quantized output
                uint8_t *dram_ptr =
                    dest_gmem +
                    (i * DIM * dim_n + j * DIM * 2 /*is this right?*/) *
                        C.OUT_ELEM_SIZE();
                gemmini_mvout(rad_device_to_host_address(
                                  reinterpret_cast<uint32_t>(dram_ptr)),
                              tile_acc_addr);
            }
        }

        gemmini_fence();
    }

    asm volatile("copy_accmem_to_gmem_dma_sync_end_%=:" ::);
}

---

### MVIN strategy and batch constraints

- **MVIN strategy**: block-mvin batching (load `batch` DIM-tiles per `gemmini_extended_mvin`,
  `cols = batch*DIM`) and strides (`gemmini_config_ld`). Validated envelope: batch=2 (cols=32) is the
  sweet spot; keep `cols <= 48` (batch <= 3) — the RTL `num_cols` field is only 6 bits, so `cols=64`
  truncates to 0 and the load is ILLEGAL. **Do NOT use `mvin2`/`mvin3` (port != 0) for `loop_ws_spad`
  inputs**: the RTL routes non-zero ports to a scratchpad region the loop_ws read ports cannot see, so
  the kernel is spike-correct but RTL-wrong. Use single-port `gemmini_extended_mvin` for loop_ws A/B.

## MX Scale and LUT Management

### gemmini_mx_load_scales

2. **Load scale factors** into scale-factor memory (sel 0 = activations/A,
   sel 1 = weights/B):
   ```c
   gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
   gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
   ```
   (RTL builds instead MMIO-write the scale-factor memory at GEMMINI_SF_MEM;
   spike uses the intrinsic. Optimize for the spike path.)

### gemmini_mx_load_scales

#define gemmini_mx_load_scales(dram_addr, len, sel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(sel) << 32) | ((uint64_t)(len) & 0xFFFFFFFFu), \
    k_MX_LOAD_SCALES)

---

### gemmini_mx_load_lut

// Load num_luts × (16 6-bit FP6 codes) from DRAM (3 LE uint32 per LUT).
// sel: 0 = B (weight), 1 = A (activation), 2 = C (output)
#define gemmini_mx_load_lut(dram_addr, num_luts, sel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(sel) << 32) | ((uint64_t)(num_luts) & 0xFFFFFFFFu), \
    k_MX_LOAD_LUT)

### gemmini_mx_load_lut

3. **(Optional) Load a quant LUT** with `gemmini_mx_load_lut(dram_addr, num_luts, sel)`
   for formats/requantization that need it (see `matmul_data_mx_lut_hw.h` users).

---

### gemmini_mx_read_smem

#define gemmini_mx_read_smem(dram_addr, smem_off_bf16, num_bf16) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(dram_addr), \
    ((uint64_t)(num_bf16) << 32) | ((uint64_t)(smem_off_bf16) & 0xFFFFFFFFu), \
    k_MX_READ_SMEM)

### gemmini_mx_read_smem

7. **Read the result** out of shared memory (BF16, packed 4 per uint64 word):
   ```c
   gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST*16, MATMUL_M*MATMUL_N);
   gemmini_fence();
   ```

---

### MX_LOAD_SCALES

| 27 | `MX_LOAD_SCALES`          | DRAM byte address of scale block | `[32]` = `sel` (0 = A-row scales, 1 = B-col scales), `[31:0]` = `len` (bytes) | Streams `len` e8m0 codes from DRAM into `mx_scale_a_mem` / `mx_scale_b_mem`. |

---

### MX_LOAD_LUT

| 29 | `MX_LOAD_LUT`             | DRAM byte address | `[33:32]` = `sel` (0 = B, 1 = A, 2 = C), `[31:0]` = `num_luts` | Loads `num_luts × (3 LE uint32 = 96 bits)` from DRAM and unpacks 16 × 6-bit FP6 E3M2 codes per LUT into `mx_lut_a / b / c`. |

---

### MX_READ_SMEM

| 28 | `MX_READ_SMEM`            | DRAM byte address | `[63:32]` = `num_u16`, `[31:0]` = `smem_word_offset` | Copies `num_u16` 16-bit words from the MX shared memory (`mx_smem`) to DRAM. Used to drain BF16 outputs or packed FP4 / FP6 codes after the matmul. |

---

### Scale-factor staging and double-buffering

## Scale-factor staging + double-buffering (subtle, easy to get wrong)

Per-32-K-block e8m0 scales are staged into SMEM by SIMT stores (NOT DMA'd):
`SF_MEM_B = 0x88000`, `SF_MEM_A = 0x8A000`. For software-pipelined K-tiles the kernel
**double-buffers** scales: odd K-tiles are staged at `SF_MEM_* + 0x800`
(`GEMMINI_SF_MEM_BUFFER_OFFSET`), and the accelerator is told which buffer to read via the
`scale_act_sel`/`scale_wgt_sel` bits (rs1[60]/rs1[61]) of `gemmini_mxquant_config_mvout`.
FP6 LUTs likewise stage into SMEM at LUT0=0x84080 (weights/B), LUT1=0x84380 (activations/A),
LUT2=0x84680 (output).

---

### MX Intrinsics Reference

## Key MX intrinsics (from gemmini.h)

- `gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides, act_mx_fmt, wgt_mx_fmt, out_mx_fmt, uselut)`
- `gemmini_mx_load_scales(dram_addr, len, sel)` — sel 0=A, 1=B
- `gemmini_mx_load_lut(dram_addr, num_luts, sel)`
- `gemmini_mxquant_config_mvout(dram_addr, i_bound, j_bound, k_bound, scale_act_sel, scale_w_sel, lut_update_granularity)`
- `gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips)`
- `gemmini_mx_read_smem(dram_addr, smem_off_bf16, num_bf16)`
- Standard Gemmini intrinsics also apply: `gemmini_extended_mvin/mvin2/mvin3`,
  `gemmini_extended_mvout`, `gemmini_preload`, `gemmini_compute_preloaded/accumulated`,
  `gemmini_config_ld/st`, `gemmini_fence`, `gemmini_flush`.

## Matmul Execution and Tiling

### gemmini_loop_ws

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

---

### gemmini_loop_ws_spad

#define gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_SPAD_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((uint64_t)(C) << 32) | 0x200U | (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

### gemmini_loop_ws_spad

6. **Compute** the weight-stationary tiled matmul over the scratchpad:
   ```c
   gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0,0,0,
       a_base, BANK_NUM*BANK_ROWS, 0, SPAD_DEST,
       false,false, false,false,false, NO_ACTIVATION, 0,0, false, 0x38);
   ```
   Results land in shared memory at `SPAD_DEST`, requantized to BF16.

---

### gemmini_loop_ws_spad_without_config_bounds

#define gemmini_loop_ws_spad_without_config_bounds(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_SPAD_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((uint64_t)(C) << 32) | 0x200U | (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

---

### gemmini_loop_ws_spad_without_config_address_and_bounds

#define gemmini_loop_ws_spad_without_config_address_and_bounds(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd, skips) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((uint64_t)(C) << 32) | 0x200U | (skips) | ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

---

### LOOP_WS (MX variant)

The MX matmul itself reuses `LOOP_WS` (funct 8). When the preceding
instruction was `LOOP_WS_CONFIG_SPAD_AB`, the spike handler dispatches
to `mx_loop_ws_spad` instead of the legacy CISC loop; `rs1` is unused
and in `rs2` the upper 32 bits are the C-side spad word offset, while
`rs2[5:0]` is the existing skip-mask byte.

---

### sp_tiled_matmul_full_spad_ws

#define sp_tiled_matmul_full_spad_ws(A_sp_addr_start, B_sp_addr_start, D_sp_addr_start, C_dst_sp_addr_start,\
  I, J, K, pad_I, pad_J, pad_K, a_transpose, b_transpose, full_C, low_D, acc, act, skips) \
  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A_sp_addr_start, (B_sp_addr_start) + (K) * (J) * DIM, NULL, \
  C_dst_sp_addr_start, a_transpose, b_transpose, full_C, low_D, acc, act, 0, 0, false, skips)

---

### calculate_spad_addr

/** Calculate scratchpad row address for A if `is_b == false` or B if `is_b == true`. */
template <bool is_b>
static inline uint32_t calculate_spad_addr(const uint32_t tile_k) {
    constexpr auto SMEM_SIZE_ROWS = BANK_NUM * BANK_ROWS;
    constexpr auto SMEM_QUARTER_ROWS = SMEM_SIZE_ROWS / 4;
    static_assert(SMEM_QUARTER_ROWS != 0);
    constexpr auto A_SPAD_ADDR_EVEN = 0;
    constexpr auto A_SPAD_ADDR_ODD = SMEM_QUARTER_ROWS;
    // B spad address is counted from the end (SMEM_SIZE_ROWS)
    // TODO: might want to swap even and odd (do bank 0-2, 1-3 instead of 0-3, 1-2)
    constexpr auto B_SPAD_ADDR_EVEN = SMEM_SIZE_ROWS;
    constexpr auto B_SPAD_ADDR_ODD = SMEM_SIZE_ROWS - SMEM_QUARTER_ROWS;

    const uint32_t odd_k = (tile_k & 1);
    const uint32_t a_spad_addr = odd_k ? A_SPAD_ADDR_ODD : A_SPAD_ADDR_EVEN;
    const uint32_t b_spad_addr = odd_k ? B_SPAD_ADDR_ODD : B_SPAD_ADDR_EVEN;

    if constexpr (is_b) {
        return b_spad_addr;
    } else {
        return a_spad_addr;
    }
}

---

### matmul_tile_async

/** Asynchronously kick off loop FSM matmul compute operation in MxGemmini.
 *  Move out accumulator data to SMEM if `acc_move_out` is true. */
template <GemmConfig C>
static inline void matmul_tile_async(const uint32_t tile_k, const bool acc_move_out) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/skip_stc);

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = calculate_spad_addr<true>(tile_k);

    const bool first_k = tile_k == 0;

    // TODO: support skipping move-out to SMEM
    // TODO(perf): !first_k creates a branch
    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,                // pad_I=0, pad_J=0, pad_K=0
        a_spad_addr_start,      // A scratchpad address in rows (grows upward)
        b_spad_addr_end,        // B scratchpad address in rows (grows downward)
        0,                      // D (bias) - none
        SPAD_DEST,              // C scratchpad address in rows
        false, false,           // A_transpose, B_transpose
        false, false, !first_k, // full_C, low_D, ex_accumulate
                                // only start in-mem accumulation after first k
        NO_ACTIVATION,          // activation
        0, 0,                   // a_spad_id, b_spad_id
        false,                  // is_resadd
        skips_compute);         // skips

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

---

### mxgemm_single_output_tile

/** Do matmul on a single TILE_M * TILE_N output tile, accumulating over the
 *  full GEMM_K. */
template <GemmConfig C, bool barrier_tile = false>
void mxgemm_single_output_tile(const uint32_t dim_m, const uint32_t dim_n,
                               const uint32_t dim_k,
                               const uint32_t tid_in_threadblock,
                               const uint32_t threads_per_threadblock) {
    asm volatile ("mxgemm_single_output_tile_start_%=:" :: );

    constexpr auto barrier_id = 2;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;

    if (tid_in_threadblock != 0) {
        return;
    }

    configure_mxgemmini<C>(dim_m, dim_n, dim_k);

    // -----------------
    // Initiate pipeline
    // -----------------
    //
    int tile_k = 0;
    // TODO: change 0's for multiple SMEM tiles
    copy_gmem_to_smem_async<C>(dim_m, dim_n, dim_k, 0, 0, tile_k);

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(calculate_scale_factor_smem_addr<false>(tile_k),
                       calculate_scale_factor_gmem_addr<C, false>(
                           &A_scales_row[0][0], tile_k, dim_m, dim_n),
                       C.SCALE_FACTORS_PER_TILE());
    load_scale_factors(calculate_scale_factor_smem_addr<true>(tile_k),
                       calculate_scale_factor_gmem_addr<C, true>(
                           &B_scales_col[0][0], tile_k, dim_m, dim_n),
                       C.SCALE_FACTORS_PER_TILE());

    // LUT is shared across the entire K, and thus loaded once per one SMEM
    // output tile
    load_lut<C>();

    // fence scale factor and LUT writes
    mu_fence_smem();

    // wait for GMEM->SMEM copy
    gemmini_fence();

    if constexpr (barrier_tile) {
        mu_barrier(barrier_id, warps_per_threadblock);
    }

    // ------------------------------
    // Main software-pipelined K-loop
    // ------------------------------
    //
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    // Potential software-pipelining loop structures:
    //                 ┌───┐   ┌───┐
    //             ┌───────────┐
    //         ┌───────┐
    // Loop 1: M0->M1->C0->M0->C1->M1->C0
    //         ┌───┐   ┌───┐   ┌───┐
    // Loop 2: M0->C0->M1->C1->M0->C0->...
    //
    for (; (tile_k * C.TILE_K) < dim_k; tile_k++) {
        const auto odd_k = (tile_k & 1);
        const auto odd_next_k = !odd_k;
        const auto last_k = ((tile_k + 1) * C.TILE_K) >= dim_k;

        // configure scalefac->PE double-buffer read; inst: 0x3420b07b
        // done for (tile_k) compute; we do this before (tile_k + 1) DMA,
        // since this may get serialized with the DMA instruction
        gemmini_mxquant_config_mvout(
            // TODO: dummy move-out space for the scale factor
            rad_device_to_host_address(
                reinterpret_cast<uint32_t>(&C_scale_factors[0])),
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
            odd_k, // A double-buffer toggle
            odd_k, // B double-buffer toggle
            QUANT_LUT_UPDATE_GRANULARITY);

        // GMEM->SMEM DMA for the next tile_k
        // TODO: This results in an unnecessary move-in at the last K tile
        if constexpr (!DISABLE_MOVE_IN_AFTER_FIRST_K) {
            if (!last_k) {
                copy_gmem_to_smem_async<C>(dim_m, dim_n, dim_k, 0 /*FIXME*/,
                                           0 /*FIXME*/, tile_k + 1);
            }
        }

        // asynchrously kick off matmul for this tile_k
        // gemmini_fence_ready();
        matmul_tile_async<C>(tile_k, last_k);

        // update scale factors for the next tile_k
        // make sure to place this between tile_async and fence to hide latency
        if constexpr (!DISABLE_SCALE_FACTOR_UPDATE) {
            load_scale_factors(
                calculate_scale_factor_smem_addr<false>(tile_k + 1),
                calculate_scale_factor_gmem_addr<C, false>(
                    &A_scales_row[0][0], tile_k + 1, dim_m, dim_n),
                C.SCALE_FACTORS_PER_TILE());
            load_scale_factors(
                calculate_scale_factor_smem_addr<true>(tile_k + 1),
                calculate_scale_factor_gmem_addr<C, true>(
                    &B_scales_col[0][0], tile_k + 1, dim_m, dim_n),
                C.SCALE_FACTORS_PER_TILE());

            // fence scale factor and LUT writes before next Gemmini compute
            mu_fence_smem();
        }

        gemmini_fence();

        if constexpr (barrier_tile) {
            mu_barrier(barrier_id, warps_per_threadblock);
        }
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );

    asm volatile ("mxgemm_single_output_tile_end_%=:" :: );
}

---

### mxgemm (mxgemm_lib.hpp)

## The driver: `mxgemm_lib.hpp`

`mxgemm<GemmConfig>(...)` computes a single TILE_M×TILE_N output tile and loops K internally
(software-pipelined: DMA next tile / compute this tile / stage next scales / fence). Per
output tile it: `configure_mxgemmini` (CONFIG_EX + bounds) → stage e8m0 scales into SMEM →
LOOP_WS load-only (DMA A/B) → LOOP_WS compute (systolic, accumulate C in SMEM at
`SPAD_DEST=256` i.e. SMEM byte 4096) → `gemmini_fence` → SIMT move-out of C (SMEM→GMEM,
`copy_smem_to_gmem_simt`, vectorized 32-bit stores). `GemmConfig` levers: TILE_M/TILE_N/
TILE_K (TILE_K multiple of 32), DATATYPE (FP8/FP6/FP4), QUANT_OUTPUT.

---

### mxgemm

/** Do a full GEMM and store the result C tensor at `C_gmem` GMEM address. */
template <GemmConfig C>
static void
mxgemm(const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
       uint8_t *C_gmem, const uint32_t tid_in_threadblock,
       const uint32_t threads_per_threadblock, const uint32_t threadblock_id) {
    mxgemm_single_output_tile<C>(dim_m, dim_n, dim_k, tid_in_threadblock,
                                 threads_per_threadblock);

    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(1, warps_per_threadblock);

    // Move-out C from SMEM to GMEM
    if constexpr (!DISABLE_GMEM_MOVE_OUT) {
        auto C_smem =
            reinterpret_cast<const __shared uint8_t *>(SPAD_DEST * DIM);
        if constexpr (SIMT_GMEM_MOVE_OUT) {
            copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(
                C_smem, C_gmem, tid_in_threadblock, threads_per_threadblock);
        } else {
            // copy_accmem_to_gmem_dma_sync<C>(C_gmem, dim_n, tid_in_threadblock);
            copy_C_smem_to_gmem_dma_sync<C>(SPAD_DEST, C_gmem, dim_n,
                                            tid_in_threadblock);

            mu_barrier(2, warps_per_threadblock);

            // we do not trace DMA move; do an additional bogus SIMT copy to
            // generate verifiable trace
            auto trace_gmem = reinterpret_cast<uint8_t *>(0x60000000);
            copy_gmem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(C_gmem, trace_gmem,
                                                      tid_in_threadblock,
                                                      threads_per_threadblock);
        }
    }
}

## Math and Data Utilities

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

### mu_exp

- For exponentials use the harness-provided `mu_exp(float)`; never expf/exp.
  Sigmoid: `1/(1 + mu_exp(-x))`.