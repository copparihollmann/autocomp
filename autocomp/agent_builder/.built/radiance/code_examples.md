## example_mxgemm_lib.hpp

SUMMARY: This document covers a GPU-attached MX-Gemmini accelerator kernel library for microscaling GEMM (fp8/fp6/fp4), demonstrating MMIO-driven systolic array control, software-pipelined K-tile double-buffering, scale factor/LUT staging into shared memory, warp-specialized DMA vs SIMT move-out, and the full `mxgemm` orchestration API.

```cpp
#include <stdint.h>
#include <radiance.h>
#include <mu_intrinsics.h>
#include "include/gemmini.h"
#include "mxgemmini_mmio.h"

// ---------------------------------------------------------------------------
// Tiling / format configuration
// ---------------------------------------------------------------------------

enum class GemmDatatype : uint8_t { FP8, FP6, FP4 };

struct GemmConfig {
    uint32_t TILE_M = 128;
    uint32_t TILE_N = 128;
    uint32_t TILE_K = 256;
    GemmDatatype DATATYPE = GemmDatatype::FP8;
    bool QUANT_OUTPUT = false;

    constexpr bool IS_FP8() const { return DATATYPE == GemmDatatype::FP8; }
    constexpr uint32_t PE_M() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_N() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_K() const { return 16; }
    constexpr uint32_t PE_TILES_I() const { return TILE_M / PE_M(); }
    constexpr uint32_t PE_TILES_J() const { return TILE_N / PE_N(); }
    constexpr uint32_t PE_TILES_K() const { return TILE_K / PE_K(); }
    constexpr uint32_t SCALE_FACTORS_PER_TILE() const { return TILE_M * TILE_K / 32; }
    constexpr uint32_t VALUES_PER_BYTE() const { return (IS_FP8() ? 1 : 2); }
    constexpr uint32_t OUT_ELEM_SIZE() const {
        return (QUANT_OUTPUT ? sizeof(uint8_t) : sizeof(uint16_t));
    }
    constexpr uint32_t TILE_M_QUANT() const {
        return (QUANT_OUTPUT ? TILE_M / VALUES_PER_BYTE() : TILE_M);
    }
    constexpr uint32_t TILE_N_QUANT() const { return TILE_N; }
    constexpr bool USE_LUT() const { return DATATYPE == GemmDatatype::FP6; }
};

// ---------------------------------------------------------------------------
// Gemmini / MMIO constants
// ---------------------------------------------------------------------------

constexpr auto GEMMINI_FORMAT_FP8  = 0;
constexpr auto GEMMINI_FORMAT_FP6  = 1;
constexpr auto GEMMINI_FORMAT_FP4  = 2;
constexpr auto GEMMINI_FORMAT_FULL = 3;
constexpr auto QUANT_LUT_UPDATE_GRANULARITY = 1;
constexpr auto GEMMINI_ACC_ADDR = (1u << (ADDR_LEN - 1));
constexpr auto SPAD_DEST = 256;

constexpr bool GEMMINI_DMA                    = true;
constexpr bool ZERO_STRIDE_K                  = false;
constexpr bool DISABLE_MOVE_IN_AFTER_FIRST_K  = false;
constexpr bool DISABLE_SCALE_FACTOR_UPDATE    = false;
constexpr bool DISABLE_GMEM_MOVE_OUT          = false;
constexpr bool SIMT_GMEM_MOVE_OUT             = true;

static uint32_t C_scale_factors[128 * 128 / 32] __attribute__((aligned(32))) = {0};

// ---------------------------------------------------------------------------
// MxGemmini one-time configuration
// ---------------------------------------------------------------------------

template <GemmConfig C>
static inline void configure_mxgemmini(const uint32_t dim_m,
                                       const uint32_t dim_n,
                                       const uint32_t dim_k) {
    static_assert(C.TILE_M == C.TILE_N,
                  "currently only supports square SMEM tile dimensions");
    static_assert(C.TILE_K >= 32 && (C.TILE_K % 32) == 0,
                  "tile K dimension is not a multiple of block size (32)");

    gemmini_flush(0);

    constexpr auto GEMMINI_FORMAT =
        C.DATATYPE == GemmDatatype::FP8 ? GEMMINI_FORMAT_FP8 :
        C.DATATYPE == GemmDatatype::FP6 ? GEMMINI_FORMAT_FP6 :
                                          GEMMINI_FORMAT_FP4;
    constexpr auto GEMMINI_FORMAT_OUT =
        C.QUANT_OUTPUT ? GEMMINI_FORMAT : GEMMINI_FORMAT_FULL;

    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY,
        0, 0, ACC_SCALE_IDENTITY,
        1, 1,
        0, 0,
        false,
        GEMMINI_FORMAT,
        GEMMINI_FORMAT,
        GEMMINI_FORMAT_OUT,
        C.USE_LUT()
    );

    // GMEM move-in strides for A (row-major, byte-packed) and B
    gemmini_extended3_config_ld(dim_k * sizeof(uint8_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE(),
                                MVIN_SCALE_IDENTITY, false, 1);

    // GMEM move-out stride for C
    gemmini_config_st(dim_n * C.OUT_ELEM_SIZE());

    // Configure scale-factor double-buffer addresses and PE read pointers
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0,  // A double-buffer toggle
        0,  // B double-buffer toggle
        QUANT_LUT_UPDATE_GRANULARITY);

    // Configure loop FSM bounds (must be issued twice for the two internal FSMs)
    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, 0);
    gemmini_fence();

    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, 0);
    gemmini_fence();
}

// ---------------------------------------------------------------------------
// Scratchpad address helpers (double-buffered A/B)
// ---------------------------------------------------------------------------

/** Return scratchpad row address for A (is_b=false) or B (is_b=true).
 *  A grows upward from 0; B grows downward from SMEM_SIZE_ROWS. */
template <bool is_b>
static inline uint32_t calculate_spad_addr(const uint32_t tile_k) {
    constexpr auto SMEM_SIZE_ROWS   = BANK_NUM * BANK_ROWS;
    constexpr auto SMEM_QUARTER_ROWS = SMEM_SIZE_ROWS / 4;
    static_assert(SMEM_QUARTER_ROWS != 0);

    constexpr auto A_SPAD_ADDR_EVEN = 0;
    constexpr auto A_SPAD_ADDR_ODD  = SMEM_QUARTER_ROWS;
    constexpr auto B_SPAD_ADDR_EVEN = SMEM_SIZE_ROWS;
    constexpr auto B_SPAD_ADDR_ODD  = SMEM_SIZE_ROWS - SMEM_QUARTER_ROWS;

    const uint32_t odd_k       = (tile_k & 1);
    const uint32_t a_spad_addr = odd_k ? A_SPAD_ADDR_ODD  : A_SPAD_ADDR_EVEN;
    const uint32_t b_spad_addr = odd_k ? B_SPAD_ADDR_ODD  : B_SPAD_ADDR_EVEN;

    if constexpr (is_b) return b_spad_addr;
    else                return a_spad_addr;
}

/** Return SMEM pointer to the scale-factor double-buffer slot for tile_k.
 *  Odd tiles live at SF_MEM + GEMMINI_SF_MEM_BUFFER_OFFSET. */
template <bool is_b>
static inline __shared uint32_t *
calculate_scale_factor_smem_addr(const uint32_t tile_k) {
    const uint32_t odd_k       = (tile_k & 1);
    const uint32_t dbuf_offset = odd_k ? GEMMINI_SF_MEM_BUFFER_OFFSET : 0;
    auto a_sf_addr = reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A + dbuf_offset);
    auto b_sf_addr = reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_B + dbuf_offset);

    if constexpr (is_b) return b_sf_addr;
    else                return a_sf_addr;
}

/** Return GMEM pointer to the e8m0 scale block for tile_k. */
template <GemmConfig C, bool is_b>
static inline const uint8_t *
calculate_scale_factor_gmem_addr(const uint8_t *scales_base_addr,
                                 const uint32_t tile_k, const uint32_t dim_m,
                                 const uint32_t dim_n) {
    const auto dim_mn    = is_b ? dim_n : dim_m;
    const auto scales_addr =
        scales_base_addr +
        (!ZERO_STRIDE_K ? (tile_k * C.TILE_K * dim_mn / 32) * sizeof(uint8_t)
                        : 0);
    return scales_addr;
}

// ---------------------------------------------------------------------------
// Scale-factor staging: GMEM -> SMEM (SIMT, ILP-unrolled 32-bit stores)
// ---------------------------------------------------------------------------

static void __attribute__((noinline))
load_scale_factors(volatile __shared uint32_t *sf_mem,
                   const uint8_t *scale_factors,
                   const int n) {
    auto word_scale_factors = reinterpret_cast<const uint32_t *>(scale_factors);

    constexpr auto ILP = 8;
    uint32_t unrolled[ILP];
    #pragma unroll 4
    for (size_t i = 0; i < n / 4; i += ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++)
            unrolled[j] = word_scale_factors[i + j];
        for (int j = 0; j < ILP; j++)
            sf_mem[i + j] = unrolled[j];
    }
}

// ---------------------------------------------------------------------------
// FP6 LUT staging: GMEM -> SMEM
// ---------------------------------------------------------------------------

template <GemmConfig C>
static inline void load_lut() {
    asm volatile ("load_lut_start_%=:" :: );

    if constexpr (C.USE_LUT()) {
        for (size_t i = 0; i < (C.TILE_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT0_ADDR) + 3 * i;
            dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT1_ADDR) + 3 * i;
            dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT2_ADDR) + 3 * i;
            dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
        }
    }

    asm volatile ("load_lut_end_%=:" :: );
}

// ---------------------------------------------------------------------------
// GMEM -> SMEM DMA (Gemmini loop FSM) or SIMT fallback
// ---------------------------------------------------------------------------

template <GemmConfig C>
static inline void copy_gmem_to_smem_async(
    const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
    const uint32_t tile_i, const uint32_t tile_j, const uint32_t tile_k) {
    asm volatile ("copy_gmem_to_smem_async_start_%=:" :: );

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end   = calculate_spad_addr<true>(tile_k);

    if constexpr (GEMMINI_DMA) {
        const uint32_t A_tile_start =
            reinterpret_cast<uint32_t>(A_in) +
            (!ZERO_STRIDE_K ? C.TILE_K * tile_k : 0);
        const uint32_t B_tile_start =
            reinterpret_cast<uint32_t>(B_in) +
            (!ZERO_STRIDE_K ? dim_n * C.TILE_K * tile_k / C.VALUES_PER_BYTE() : 0);

        // Set GMEM base addresses for A and B in the loop FSM
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            rad_device_to_host_address(A_tile_start),
            rad_device_to_host_address(B_tile_start),
            k_LOOP_WS_CONFIG_ADDRS_AB)

        // Set GMEM row strides for A and B
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            (uint64_t)(dim_k * sizeof(uint8_t)),
            (uint64_t)(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE()),
            k_LOOP_WS_CONFIG_STRIDES_AB)

        // Issue DMA move-in only (skip execute + store-C)
        constexpr uint32_t skips_mvin =
            loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                              /*skip_ex=*/1, /*skip_stc=*/1);
        constexpr auto DONTCARE = 0;
        gemmini_loop_ws_spad(
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
            0, 0, 0,
            a_spad_addr_start,
            b_spad_addr_end,
            0, DONTCARE,
            false, false,
            false, false, false,
            NO_ACTIVATION,
            0, 0,
            false,
            skips_mvin);

    } else {
        gemmini_config_ld(dim_k * sizeof(uint8_t));

        for (int i = 0; i < C.PE_TILES_I(); i++) {
            for (int k = 0; k < C.PE_TILES_K(); k++) {
                const uint8_t *dram_ptr = ((uint8_t*)A_in) + i * DIM * dim_k + k * DIM;
                const uint32_t sp_addr  = a_spad_addr_start + (i * C.PE_TILES_K() + k) * DIM;
                gemmini_extended_mvin(
                    rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)),
                    sp_addr, DIM, DIM);
            }
        }

        gemmini_config_ld(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE());

        for (int k = 0; k < C.PE_TILES_K(); k++) {
            for (int j = 0; j < C.PE_TILES_J(); j++) {
                const uint8_t *dram_ptr =
                    ((uint8_t *)B_in) + k * DIM * dim_n / C.VALUES_PER_BYTE() + j * DIM;
                const uint32_t b_spad_addr_start =
                    b_spad_addr_end - C.PE_TILES_K() * C.PE_TILES_J() * DIM;
                const uint32_t sp_addr =
                    b_spad_addr_start + (k * C.PE_TILES_J() + j) * DIM;
                gemmini_extended_mvin(
                    rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)),
                    sp_addr, DIM, DIM);
            }
        }
    }

    asm volatile ("copy_gmem_to_smem_async_end_%=:" :: );
}

// ---------------------------------------------------------------------------
// SMEM -> GMEM move-out: SIMT (32-bit vectorized)
// ---------------------------------------------------------------------------

template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_smem_to_gmem_simt(const __shared uint8_t *src_smem,
                                   uint8_t *dest_gmem,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock) {
    asm volatile("copy_smem_to_gmem_simt_start_%=:" ::);

    auto *src_smem_vec  = reinterpret_cast<const __shared uint32_t *>(src_smem);
    auto *dest_gmem_vec = reinterpret_cast<uint32_t *>(dest_gmem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint32_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint32_t)
                      / threads_per_threadblock;

    #pragma unroll 32
    for (int i = 0; i < iter; i++) {
        const auto index    = threads_per_threadblock * i + tid_in_threadblock;
        const auto smem_addr = src_smem_vec  + index;
        auto       gmem_addr = dest_gmem_vec + index;
        *gmem_addr = *smem_addr;
    }

    asm volatile ("copy_smem_to_gmem_simt_end_%=:" :: );
}

// ---------------------------------------------------------------------------
// GMEM -> SMEM move-in: SIMT, 16-bit writes (requantizer interface)
// ---------------------------------------------------------------------------

template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_gmem_to_smem_simt_bf16(
    const uint8_t *src_gmem, __shared uint8_t *dest_smem,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock) {
    asm volatile("copy_gmem_to_smem_simt_bf16_start_%=:" ::);

    auto *src_gmem_vec  = reinterpret_cast<const uint16_t *>(src_gmem);
    auto *dest_smem_vec = reinterpret_cast<__shared uint16_t *>(dest_smem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint16_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint16_t)
                      / threads_per_threadblock;

    #pragma unroll 32
    for (int i = 0; i < iter; i++) {
        const auto index    = threads_per_threadblock * i + tid_in_threadblock;
        const auto src_addr = src_gmem_vec  + index;
        auto       dst_addr = dest_smem_vec + index;
        *dst_addr = *src_addr;
    }

    asm volatile("copy_gmem_to_smem_simt_bf16_end_%=:" ::);
}

// ---------------------------------------------------------------------------
// SMEM -> GMEM move-out: Gemmini DMA (blocking, tid==0 only)
// ---------------------------------------------------------------------------

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
                uint8_t *dram_ptr =
                    dest_gmem + (i * 2 * DIM * dim_n + j * DIM) * C.OUT_ELEM_SIZE();
                gemmini_mvout(
                    rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)),
                    tile_spad_addr);
            }
        }
        gemmini_fence();
    }

    asm volatile("copy_smem_to_gmem_dma_sync_end_%=:" ::);
}

// ---------------------------------------------------------------------------
// Asynchronous matmul tile dispatch to MxGemmini loop FSM
// ---------------------------------------------------------------------------

/** Kick off loop FSM compute for tile_k.
 *  If acc_move_out==true, also issues the store-C (STC) step. */
template <GemmConfig C>
static inline void matmul_tile_async(const uint32_t tile_k,
                                     const bool acc_move_out) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
        loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                          /*skip_ex=*/0, /*skip_stc=*/skip_stc);

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end   = calculate_spad_addr<true>(tile_k);
    const bool first_k = (tile_k == 0);

    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, 0, 0,
        a_spad_addr_start,
        b_spad_addr_end,
        0,          // D (bias) – none
        SPAD_DEST,  // C destination scratchpad address
        false, false,
        false, false, !first_k,  // ex_accumulate=true after first K tile
        NO_ACTIVATION,
        0, 0,
        false,
        skips_compute);

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

// ---------------------------------------------------------------------------
// Software-pipelined K-loop: single TILE_M x TILE_N output tile
// ---------------------------------------------------------------------------

/** Compute one TILE_M×TILE_N output tile, accumulating over the full GEMM_K.
 *  Only tid==0 drives the accelerator; other threads return immediately. */
template <GemmConfig C, bool barrier_tile = false>
void mxgemm_single_output_tile(const uint32_t dim_m, const uint32_t dim_n,
                               const uint32_t dim_k,
                               const uint32_t tid_in_threadblock,
                               const uint32_t threads_per_threadblock) {
    asm volatile ("mxgemm_single_output_tile_start_%=:" :: );

    constexpr auto barrier_id = 2;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;

    if (tid_in_threadblock != 0) return;

    configure_mxgemmini<C>(dim_m, dim_n, dim_k);

    // --- Pipeline prologue: prefetch tile_k=0 ---
    int tile_k = 0;
    copy_gmem_to_smem_async<C>(dim_m, dim_n, dim_k, 0, 0, tile_k);

    load_scale_factors(
        calculate_scale_factor_smem_addr<false>(tile_k),
        calculate_scale_factor_gmem_addr<C, false>(&A_scales_row[0][0], tile_k, dim_m, dim_n),
        C.SCALE_FACTORS_PER_TILE());
    load_scale_factors(
        calculate_scale_factor_smem_addr<true>(tile_k),
        calculate_scale_factor_gmem_addr<C, true>(&B_scales_col[0][0], tile_k, dim_m, dim_n),
        C.SCALE_FACTORS_PER_TILE());

    // LUT is invariant across K; load once per output tile
    load_lut<C>();

    mu_fence_smem();   // fence scale-factor + LUT writes before Gemmini reads
    gemmini_fence();   // wait for GMEM->SMEM DMA

    if constexpr (barrier_tile)
        mu_barrier(barrier_id, warps_per_threadblock);

    // --- Main software-pipelined K-loop ---
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    for (; (tile_k * C.TILE_K) < dim_k; tile_k++) {
        const auto odd_k      = (tile_k & 1);
        const auto last_k     = ((tile_k + 1) * C.TILE_K) >= dim_k;

        // Point scale-factor PE read to the current double-buffer slot
        gemmini_mxquant_config_mvout(
            rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
            odd_k,  // A double-buffer toggle (rs1[60])
            odd_k,  // B double-buffer toggle (rs1[61])
            QUANT_LUT_UPDATE_GRANULARITY);

        // Prefetch next K tile while current tile computes
        if constexpr (!DISABLE_MOVE_IN_AFTER_FIRST_K) {
            if (!last_k)
                copy_gmem_to_smem_async<C>(dim_m, dim_n, dim_k, 0, 0, tile_k + 1);
        }

        // Dispatch compute (and optional STC on last tile)
        matmul_tile_async<C>(tile_k, last_k);

        // Stage scale factors for the next tile while accelerator computes
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

            mu_fence_smem();  // fence before next Gemmini compute reads scales
        }

        gemmini_fence();  // wait for current tile compute + optional STC

        if constexpr (barrier_tile)
            mu_barrier(barrier_id, warps_per_threadblock);
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );
    asm volatile ("mxgemm_single_output_tile_end_%=:" :: );
}

// ---------------------------------------------------------------------------
// Top-level GEMM: compute + C move-out
// ---------------------------------------------------------------------------

/** Full GEMM: compute one output tile then move C from SMEM to GMEM. */
template <GemmConfig C>
static void
mxgemm(const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
       uint8_t *C_gmem, const uint32_t tid_in_threadblock,
       const uint32_t threads_per_threadblock, const uint32_t threadblock_id) {

    mxgemm_single_output_tile<C>(dim_m, dim_n, dim_k,
                                 tid_in_threadblock, threads_per_threadblock);

    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(1, warps_per_threadblock);  // sync before C move-out

    if constexpr (!DISABLE_GMEM_MOVE_OUT) {
        auto C_smem = reinterpret_cast<const __shared uint8_t *>(SPAD_DEST * DIM);

        if constexpr (SIMT_GMEM_MOVE_OUT) {
            // All warps cooperatively stream C out via uniform-strided 32-bit loads
            copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(
                C_smem, C_gmem, tid_in_threadblock, threads_per_threadblock);
        } else {
            // tid==0 drives Gemmini DMA move-out
            copy_C_smem_to_gmem_dma_sync<C>(SPAD_DEST, C_gmem, dim_n,
                                            tid_in_threadblock);

            mu_barrier(2, warps_per_threadblock);

            // Bogus SIMT GMEM->GMEM copy to generate a verifiable memory trace
            auto trace_gmem = reinterpret_cast<uint8_t *>(0x60000000);
            copy_gmem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(C_gmem, trace_gmem,
                                                      tid_in_threadblock,
                                                      threads_per_threadblock);
        }
    }
}
```

## matmul_tiled_fp6_128x128.c

SUMMARY: Demonstrates MMIO-driven MX-Gemmini accelerator control from SIMT/RISC-V, covering LUT loading, scale factor staging, tiled mvin/mvout, loop_ws_spad dispatch with skip-flags, and BF16 result readback from shared scratchpad memory.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"

#define TILE 16
#define VALUES_PER_BYTE 2
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4

// MMIO base addresses
#define GEMMINI_CTRL      0x40084000
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

// LUT regions (weight=0x80, activation=0x380, output=0x680)
#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x380)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x680)

// Scale-factor SRAM regions
#define GEMMINI_SF_MEM   0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B  GEMMINI_SF_MEM

#define SMEM             0x40000000

// FP6 format selector (0=fp8, 1=fp6, 2=fp4)
#define GEMMINI_FORMAT 1

// Override fence to poll MMIO BUSY register
#undef gemmini_fence
#define gemmini_fence() \
  { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

// Override RoCC instruction issue to write MMIO command block
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
  *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
  *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
  *((volatile uint32_t*) GEMMINI_INST_ADDR) =                                 \
      (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
```

```c
// Write e8m0 scale factors into accelerator SRAM, 8 bytes at a time
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}
```

```c
// Configure Gemmini for FP6 weight-stationary mode with LUT dequant
void gemmini_configure(int USE_LUT) {
  gemmini_flush(0);

  // CONFIG_EX: set format, strides, activation, WS mode, LUT enable
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)       // A stride
    | (GEMMINI_FORMAT << 14)      // C format (fp6)
    | (GEMMINI_FORMAT << 12)      // B format
    | (GEMMINI_FORMAT << 10)      // A format
    | (0 << 9)                    // B transpose = false
    | (0 << 8)                    // A transpose = false
    | ((false) << 7)              // set_only_strides = false
    | ((USE_LUT) << 4)            // enable LUT dequant
    | ((0) << 3)                  // activation = none
    | ((WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
    ((uint64_t)(1) << 48)         // C stride
    | (0),
    k_CONFIG);

  gemmini_extended_config_st(DIM * sizeof(uint8_t), NO_ACTIVATION, 1);
}
```

```c
// Stage LUT tables into accelerator MMIO LUT banks
// B_lut / A_lut / C_lut: uint32_t [N_groups][3], packed by lut_mapping_demo.py
void stage_luts(int matmul_m, int matmul_n) {
  // LUT0: weight (B) groups
  for (size_t i = 0; i < (matmul_n >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i;
    dst[0] = B_lut[i][0];
    dst[1] = B_lut[i][1];
    dst[2] = B_lut[i][2];
  }
  // LUT1: activation (A) groups
  for (size_t i = 0; i < (matmul_m >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i;
    dst[0] = A_lut[i][0];
    dst[1] = A_lut[i][1];
    dst[2] = A_lut[i][2];
  }
  // LUT2: output (C) requant groups
  for (size_t i = 0; i < (matmul_m >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i;
    dst[0] = C_lut[i][0];
    dst[1] = C_lut[i][1];
    dst[2] = C_lut[i][2];
  }
}
```

```c
// Stage e8m0 scale factors for A (row-wise) and B (col-wise) into SF SRAM
void stage_scale_factors(int matmul_m, int matmul_n, int matmul_gk) {
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                     (uint8_t *) &A_scales_row, matmul_m * matmul_gk);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                     (uint8_t *) &B_scales_col, matmul_n * matmul_gk);
  gemmini_fence();
}
```

```c
// Tiled mvin of A (packed fp6, 2 values/byte) into scratchpad
// Layout: A_in_hw[MATMUL_M/2][MATMUL_K], tiles_I m-tiles x tiles_K k-tiles
// Each tile = DIM hw-rows x DIM bytes
void mvin_A(int tiles_I, int tiles_K, int matmul_k, uint32_t a_base) {
  gemmini_config_ld(matmul_k * sizeof(uint8_t));  // stride = full packed row width
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw
                          + i * DIM * matmul_k
                          + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }
}
```

```c
// Tiled mvin of B (packed fp6, 2 values/byte) into scratchpad
// Layout: B_in[MATMUL_K][MATMUL_N/2], tiles_K k-tiles x tiles_J n-tiles
// Each tile = K_TILE rows x DIM bytes
void mvin_B(int tiles_K, int tiles_J, int matmul_n, uint32_t b_base) {
  gemmini_config_ld((matmul_n / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in
                          + k * K_TILE * (matmul_n / VALUES_PER_BYTE)
                          + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }
}
```

```c
// Issue tiled WS matmul entirely from scratchpad, skipping all DRAM loads/stores
// skip_flags = 0x38: bits[3]=ldA, [4]=ldB, [5]=ldD all set → skip DRAM mvins
// Accumulator result lands at SPAD_DEST in shared scratchpad
void dispatch_loop_ws_spad(int tiles_I, int tiles_J, int tiles_K,
                            uint32_t a_base, uint32_t b_end,
                            int spad_dest) {
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0,
                               ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);

  gemmini_mxquant_config_mvout(
      (uint64_t)/*scale_factors ptr*/ 0,
      tiles_I, tiles_J, tiles_K,
      0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,          // pad_I, pad_J, pad_K
      a_base,           // A scratchpad base address
      b_end,            // B scratchpad end address (8192)
      0,                // D (bias) — none
      spad_dest,        // C accumulator destination in scratchpad
      false, false,     // A_transpose, B_transpose
      false, false, false, // full_C, low_D, ex_accumulate
      NO_ACTIVATION,
      0, 0,             // a_spad_id, b_spad_id (double-buffer select)
      false,            // is_resadd
      0x38);            // skip ldA|ldB|ldD; execute compute only
  //
  // Skip-flag encoding (rs1 bits):
  //   bit 3 (0x08) = ldA  — skip loading A from DRAM
  //   bit 4 (0x10) = ldB  — skip loading B from DRAM
  //   bit 5 (0x20) = ldD  — skip loading D/bias
  //   bit 6 (0x40) = ex   — skip compute
  //   bit 7 (0x80) = st   — skip acc→spad store
}
```

```c
// Read BF16 results back from scratchpad SMEM and verify against golden
// C_hw[MATMUL_M][OUT_COLS] where OUT_COLS = MATMUL_N / BF16_PER_WORD
// smem layout: SPAD_DEST * 2 uint64_t words offset from SMEM base
void readback_and_verify(uint64_t C_hw[][/*OUT_COLS*/4],
                         int matmul_m, int out_cols,
                         int spad_dest) {
  // Each scratchpad row = 2 uint64_t words (128-bit); SPAD_DEST in 64-bit units
  uint64_t *smem_start_addr = ((uint64_t *)SMEM) + spad_dest * 2;

  for (int i = 0; i < matmul_m; i++) {
    for (int j = 0; j < out_cols; j++) {
      C_hw[i][j] = *(smem_start_addr + (i * out_cols + j));
    }
  }
  gemmini_fence();

  int errors = 0;
  for (int i = 0; i < matmul_m; i++) {
    for (int j = 0; j < out_cols; j++) {
      uint64_t got = C_hw[i][j];

      // Pack 4 consecutive bf16 golden values into expected uint64_t
      uint64_t exp =
          ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
          ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
          ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
          ((uint64_t)C_out_bf16[i][j*4 + 0]);

      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            errors++;
          }
        }
      }
    }
  }
}
```

## matmul_ws_mx_generic.c

SUMMARY: Demonstrates MMIO-driven MX-Gemmini accelerator control from SIMT/CPU, covering LUT programming, scale factor loading, scratchpad address management, and weight-stationary tiled matrix multiply with fp6/fp8/fp4 microscaling formats using loop_ws_spad with skip masks.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"

#define TILE 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define ADDR_LEN 32

// MMIO register map
#define GEMMINI_CTRL      0x40084000
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

// LUT memory regions (weight=64 rows x 96b, activation, output)
#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)   // weight LUT
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x380)  // activation LUT
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x680)  // output LUT

// Scale factor memory
#define GEMMINI_SF_MEM   0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B  GEMMINI_SF_MEM

#define SMEM 0x40000000
#define BF16_PER_WORD 4

// fp6 format selector (0=fp8, 1=fp6, 2=fp4)
#define GEMMINI_FORMAT 1

// Fence: spin on BUSY register
#undef gemmini_fence
#define gemmini_fence() \
  { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

// MMIO RoCC instruction issue
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
  *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
  *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
  *((volatile uint32_t*) GEMMINI_INST_ADDR) =                                 \
      (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
```

```c
// Configure Gemmini execution mode: weight-stationary, fp6, LUT-enabled
void gemmini_config_ex_mx(int use_lut, int format) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)   // A stride
    | (format << 14)          // C format
    | (format << 12)          // B format
    | (format << 10)          // A format
    | (0 << 9)                // B transpose
    | (0 << 8)                // A transpose
    | ((false) << 7)          // set only strides
    | ((use_lut) << 4)
    | ((0) << 3)              // activation function
    | ((WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
    ((uint64_t)(1) << 48)     // C stride
    | (0),
    k_CONFIG);
}
```

```c
// Write pre-packed HW-layout LUTs (uint32_t [N_groups][3]) to accelerator LUT banks
// B_lut -> LUT0 (weight), A_lut -> LUT1 (activation), C_lut -> LUT2 (output)
void program_luts(int matmul_m, int matmul_n) {
  for (size_t i = 0; i < (matmul_n >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT0_ADDR) + 3 * i;
    dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
  }
  for (size_t i = 0; i < (matmul_m >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT1_ADDR) + 3 * i;
    dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
  }
  for (size_t i = 0; i < (matmul_m >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
    volatile uint32_t *dst = ((volatile uint32_t *) GEMMINI_LUT2_ADDR) + 3 * i;
    dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
  }
}
```

```c
// Load e8m0 scale factors into accelerator SF memory as 64-bit words
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *) scale_factors;
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}
```

```c
// Tiled mvin A (row-major packed fp6, stride = full K width) into scratchpad
// Layout: A_in_hw[MATMUL_M/2][MATMUL_K], tiles_I m-tiles x tiles_K k-tiles
void mvin_A_tiled(int tiles_I, int tiles_K, uint32_t a_base,
                  int matmul_k) {
  gemmini_config_ld(matmul_k * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw
                          + i * DIM * matmul_k
                          + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }
}
```

```c
// Tiled mvin B (packed fp6, stride = MATMUL_N/2) into scratchpad
// Layout: B_in[MATMUL_K][MATMUL_N/2], tiles_K k-tiles x tiles_J n-tiles
void mvin_B_tiled(int tiles_K, int tiles_J, uint32_t b_base,
                  int matmul_n) {
  gemmini_config_ld((matmul_n / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in
                          + k * K_TILE * (matmul_n / VALUES_PER_BYTE)
                          + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * K_TILE;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }
}
```

```c
// Issue weight-stationary loop over scratchpad tiles, skipping DRAM ld/st
// skip_mask bits: 0x08=ldA, 0x10=ldB, 0x20=ldD, 0x40=ex, 0x80=st
// 0x38 = skip ldA + ldB + ldD (operands already in spad; use spad addresses)
void issue_loop_ws_spad(int tiles_I, int tiles_J, int tiles_K,
                        uint32_t a_base, int spad_dest) {
  uint32_t b_end_addr = 8192; // scratchpad end address for B region
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,          // pad_I, pad_J, pad_K
      a_base,           // A scratchpad base address
      b_end_addr,       // B scratchpad end address
      0,                // D (bias) - none
      spad_dest,        // C accumulator destination address
      false, false,     // A_transpose, B_transpose
      false, false, false, // full_C, low_D, ex_accumulate
      NO_ACTIVATION,
      0, 0,             // a_spad_id, b_spad_id (double-buffer select)
      false,            // is_resadd
      0x38);            // skip ldA | ldB | ldD; perform compute + store to spad
}
```

```c
// Move-out: read C results from accelerator scratchpad via shared memory window
// smem_start_addr maps to spad at SPAD_DEST * 2 uint64 words
void mvout_C_from_spad(uint64_t C_hw[][MATMUL_N/8],
                       int spad_dest, int matmul_m, int matmul_n) {
  uint64_t *smem_start_addr = ((uint64_t *)SMEM) + spad_dest * 2;
  for (int i = 0; i < matmul_m; i++) {
    for (int j = 0; j < matmul_n / 8; j++) {
      C_hw[i][j] = *(smem_start_addr + (i * matmul_n / 8 + j));
    }
  }
}
```

```c
// Verify packed fp6 output nibble-by-nibble against reference, counting bit errors
static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) { count += x & 1; x >>= 1; }
  return count;
}

int verify_output(uint64_t C_hw[][MATMUL_N/8], int matmul_m, int matmul_n) {
  int errors = 0, diff1 = 0, diff2 = 0, diff3plus = 0;
  uint8_t *hw_bytes = (uint8_t *)C_hw;

  for (int i = 0; i < matmul_m / 2; i++) {
    for (int j = 0; j < matmul_n; j++) {
      uint8_t got = hw_bytes[i * matmul_n + j];
      uint8_t exp = C_proj_hw[i][j];

      // Check low nibble
      uint8_t got_lo = got & 0x0F, exp_lo = exp & 0x0F;
      if (got_lo != exp_lo) {
        errors++;
        int bits = popcount8(got_lo ^ exp_lo);
        if      (bits == 1) diff1++;
        else if (bits == 2) diff2++;
        else                diff3plus++;
      }

      // Check high nibble
      uint8_t got_hi = (got >> 4) & 0x0F, exp_hi = (exp >> 4) & 0x0F;
      if (got_hi != exp_hi) {
        errors++;
        int bits = popcount8(got_hi ^ exp_hi);
        if      (bits == 1) diff1++;
        else if (bits == 2) diff2++;
        else                diff3plus++;
      }
    }
  }
  return errors;
}
```

## example_gemm_simt_tiled.hpp

SUMMARY: Demonstrates a tiled GEMM kernel using SIMT warps with BF16 operands, shared memory staging, warp barriers, and thread-tile decomposition (TM/TN) for computing C = A * B on a Muon SIMT architecture with MX-Gemmini acceleration potential.

```cpp
#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>
#include <stdint.h>

#ifndef GEMM_SIMT_NUM_WARPS
#define GEMM_SIMT_NUM_WARPS 4
#endif

#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(GEMM_SIMT_NUM_WARPS)
#define THREADBLOCK_SIZE MU_BLOCK_SIZE(GEMM_SIMT_NUM_WARPS)
#define ILP_MEM 2

#define BK 32
#define BM 16
#define BN 32
#ifndef TM
#define TM 1
#endif
#ifndef TN
#define TN 2
#endif
#define TBM (BM / TM)
#define TBN (BN / TN)
#define BLOCK_SIZE (BM * BN)

#define A_WORDS (BM * BK / 2)
#define B_WORDS (BK * BN / 2)
#define C_TILES (TBM * TBN)
#define A_ITERS (A_WORDS / THREADBLOCK_SIZE)
#define B_ITERS (B_WORDS / THREADBLOCK_SIZE)
#define C_ITERS (C_TILES / THREADBLOCK_SIZE)
#define A_FULL_ITERS ((A_ITERS / ILP_MEM) * ILP_MEM)
#define B_FULL_ITERS ((B_ITERS / ILP_MEM) * ILP_MEM)

struct GEMMArgs {
  __global uint32_t* A;
  __global uint32_t* B;
  __global uint32_t* C;
  uint32_t M;
  uint32_t K;
  uint32_t N;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

// C = A * B where A is MxK, B is KxN, C is MxN (all bf16)
static inline void gemm(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GEMMArgs*>(arg);
  uint32_t tid = tid_in_threadblock;
  uint32_t total_blocks = args->M * args->N / BLOCK_SIZE;
  uint32_t blocks_per_cluster = total_blocks / MU_NUM_CLUSTERS;
  uint32_t block_N = args->N / BN;

  uint32_t M = args->M;
  uint32_t N = args->N;
  uint32_t K = args->K;
  __global uint32_t *A = args->A;
  __global uint32_t *B = args->B;
  __global uint32_t *C = args->C;
  __shared uint32_t *As = sdata;
  __shared uint32_t *Bs = sdata + BM * BK / 2;

  for (uint32_t c_block = 0; c_block < blocks_per_cluster; c_block++) {
    uint32_t block_idx = threadblock_id * blocks_per_cluster + c_block;
    uint32_t block_x_idx = block_idx / block_N;
    uint32_t block_y_idx = block_idx % block_N;

    // clear accumulators
    _Float16 acc[C_ITERS][TM * TN];
    #pragma unroll
    for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
      #pragma unroll
      for (uint32_t i = 0; i < TM*TN; i++) acc[c_iter][i] = 0;
    }

    // stream across K dimension
    for (uint32_t k_block = 0; k_block < K; k_block += BK) {

      // load A tile to smem with ILP
      #pragma unroll
      for (uint32_t base = 0; base < A_FULL_ITERS; base += ILP_MEM) {
        uint32_t a_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2));
          uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
          a_val[u] = A[A_x * (K / 2) + A_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          As[elem_idx] = a_val[u];
        }
      }
      #if (A_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = A_FULL_ITERS; block < A_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2));
        uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
        As[elem_idx] = A[A_x * (K / 2) + A_y];
      }
      #endif

      // load B tile to smem with ILP
      #pragma unroll
      for (uint32_t base = 0; base < B_FULL_ITERS; base += ILP_MEM) {
        uint32_t b_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2));
          uint32_t B_x = k_block + (elem_idx / (BN / 2));
          b_val[u] = B[B_x * (N / 2) + B_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          Bs[elem_idx] = b_val[u];
        }
      }
      #if (B_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = B_FULL_ITERS; block < B_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2));
        uint32_t B_x = k_block + (elem_idx / (BN / 2));
        Bs[elem_idx] = B[B_x * (N / 2) + B_y];
      }
      #endif

      // synchronize smem writes before compute
      mu_fence_smem();
      mu_barrier(0, BLOCK_NUM_WARPS);

      // compute thread tile: unpack bf16x2 pairs and accumulate
      #pragma unroll
      for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
        uint32_t c_tile = tid + c_iter * THREADBLOCK_SIZE;
        uint32_t thread_x = c_tile / TBN;
        uint32_t thread_y = c_tile % TBN;
        for (uint32_t k = 0; k < BK / 2; k++) {
          for (uint32_t i = 0; i < TM; i++) {
            uint32_t a_idx = thread_x * TM + i;
            auto [a0, a1] = unpack_bf16x2(As[a_idx * BK / 2 + k]);
            for (uint32_t j = 0; j < TN / 2; j++) {
              uint32_t b_idx = thread_y * TN / 2 + j;
              auto [b00, b10] = unpack_bf16x2(Bs[2*k * (BN / 2) + b_idx]);
              auto [b01, b11] = unpack_bf16x2(Bs[(2*k + 1) * (BN / 2) + b_idx]);
              acc[c_iter][i * TN + 2 * j]     += a0 * b00 + a1 * b01;
              acc[c_iter][i * TN + 2 * j + 1] += a0 * b10 + a1 * b11;
            }
          }
        }
      }

      mu_barrier(0, BLOCK_NUM_WARPS);
    }

    // store C tile: pack bf16x2 pairs and write to global memory
    #pragma unroll
    for (uint32_t c_iter = 0; c_iter < C_ITERS; c_iter++) {
      uint32_t c_tile = tid + c_iter * THREADBLOCK_SIZE;
      uint32_t thread_x = c_tile / TBN;
      uint32_t thread_y = c_tile % TBN;
      uint32_t c_row = block_x_idx * BM + thread_x * TM;
      uint32_t c_col = block_y_idx * BN / 2 + thread_y * TN / 2;
      #pragma unroll
      for (uint32_t i = 0; i < TM; i++) {
        uint32_t c_x = c_row + i;
        for (uint32_t j = 0; j < TN / 2; j++) {
          uint32_t c_y = c_col + j;
          C[c_x * (N / 2) + c_y] = pack_bf16x2(
            acc[c_iter][i * TN + 2*j],
            acc[c_iter][i * TN + 2*j + 1]
          );
        }
      }
    }
  }
}
```

## matmul_tiled_fp4_64x64.c

SUMMARY: Demonstrates MMIO-driven MX-Gemmini accelerator control from SIMT, covering scale factor loading, tiled mvin/mvout for fp8 operands, loop_ws_spad compute dispatch, and BF16-packed result readback with bit-exact verification.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp4_64x64.h"

#define GEMMINI_SF_MEM      0x40088000
#define GEMMINI_SF_MEM_A    (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B    GEMMINI_SF_MEM
#define SMEM                0x40000000

#define GEMMINI_CTRL        0x40084000
#define GEMMINI_RS1_ADDR    (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR    (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR   (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR   (GEMMINI_CTRL + 0x20)

#define DIM          16
#define ADDR_LEN     32
#define BF16_PER_WORD 4
#define OUT_COLS     (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;
typedef uint64_t out_t;

// MMIO RoCC instruction issue: write RS1, RS2, then instruction word
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                        \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                        \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) =                               \
        (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

// Poll BUSY register until accelerator is idle
#undef gemmini_fence
#define gemmini_fence() { \
    while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); \
}
```

```c
// Load e8m0 scale factors into SF_MEM: fills n/8 64-bit words with 0x7f bytes
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
    for (size_t i = 0; i < n / 8; i++) {
        sf_mem[i] = 0x7f7f7f7f7f7f7f7f;
    }
}
```

```c
// Accelerator setup: flush, configure weight-stationary mode, load A/B scale factors
void gemmini_setup_mx(void) {
    gemmini_flush(0);
    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 2, 2, 3, 0);

#ifdef SPIKE_SIM
    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
    // SF_MEM_A for row scales (A), SF_MEM_B for col scales (B)
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                       (uint8_t *) &A_scales_row, 1024);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                       (uint8_t *) &B_scales_col, 1024);
#endif
}
```

```c
// Tiled MVIN for A: each (i,k) tile loaded into scratchpad at a_base + (i*tiles_K+k)*DIM
void mvin_A_tiles(int tiles_I, int tiles_K, uint32_t a_base) {
    gemmini_config_ld(MATMUL_M * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)A_in_hw) + i * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }
}

// Tiled MVIN for B: each (k,j) tile loaded into scratchpad at b_base + (k*tiles_J+j)*DIM
// Stride accounts for fp4 packing: MATMUL_N * sizeof(elem_t) / 2
void mvin_B_tiles(int tiles_K, int tiles_J, uint32_t b_base) {
    gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);
    for (int k = 0; k < tiles_K; k++) {
        for (int j = 0; j < tiles_J; j++) {
            elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N / 2 + j * DIM;
            uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }
}
```

```c
// Issue weight-stationary loop over full scratchpad tile grid;
// results land at SPAD_DEST (scratchpad row 128, i.e. SMEM offset SPAD_DEST*16 bytes)
void dispatch_loop_ws(int tiles_I, int tiles_J, int tiles_K,
                      uint32_t a_base, uint32_t b_base, int SPAD_DEST) {
    gemmini_config_st(OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout(
        (uint64_t)/*scale_factors ptr*/ 0,
        tiles_I, tiles_J, tiles_K, 0, 0, 1);

    gemmini_loop_ws_spad(
        tiles_I, tiles_J, tiles_K,
        0, 0, 0,
        a_base,
        BANK_NUM * BANK_ROWS,   // b_base sentinel: end of scratchpad
        0,
        SPAD_DEST,
        false, false,
        false, false, false,
        NO_ACTIVATION,
        0, 0,
        false,
        0x38);                  // funct7 encoding for MX fp4 mode
}
```

```c
// Read BF16-packed results back from shared memory (scratchpad IS cluster SMEM)
// SPAD_DEST rows map to SMEM at base + SPAD_DEST*2 uint64_t words
void readback_C(out_t C_hw[MATMUL_M][OUT_COLS], int SPAD_DEST) {
#ifdef SPIKE_SIM
    gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
    uint64_t *smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            C_hw[i][j] = *(smem_start_addr + (i * OUT_COLS + j));
        }
    }
#endif
}
```

```c
// Bit-exact BF16 verification: golden values are uint16_t, HW output is 4xBF16 packed
// into uint64_t words (lane 0 in bits[15:0], lane 3 in bits[63:48])
int verify_bf16_output(out_t C_hw[MATMUL_M][OUT_COLS]) {
    int errors = 0;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            uint64_t got = C_hw[i][j];
            uint64_t exp =
                ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                ((uint64_t)C_out_bf16[i][j*4 + 0]);

            if (got != exp) {
                for (int lane = 0; lane < BF16_PER_WORD; lane++) {
                    uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
                    uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
                    if (got_bf16 != exp_bf16) {
                        errors++;
                    }
                }
            }
        }
    }
    return errors;
}
```

## matmul_tiled_fp8_64x64.c

SUMMARY: Demonstrates MMIO-driven MX-Gemmini accelerator control from SIMT, covering scale factor loading, tiled FP8 matrix multiply with loop_ws_spad, and BF16-packed result readback from scratchpad/shared memory.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define GEMMINI_SF_MEM      0x40088000
#define GEMMINI_SF_MEM_A    (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B    GEMMINI_SF_MEM
#define SMEM                0x40000000

#define DIM                 16
#define ADDR_LEN            32
#define BF16_PER_WORD       4
#define OUT_COLS            (MATMUL_M / BF16_PER_WORD)

#define GEMMINI_CTRL        0x40084000
#define GEMMINI_RS1_ADDR    (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR    (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR   (GEMMINI_CTRL + 0x0)

// MMIO command issue: override RoCC instruction macro for MMIO-mapped Gemmini
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                        \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                        \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) =                               \
        (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word
```

```c
// Load per-32-element e8m0 scale factors into Gemmini SF memory region.
// sf_mem: MMIO base for scale factor SRAM (A: SF_MEM+0x2000, B: SF_MEM+0x0)
// scale_factors: packed uint8 array, shape [K/32][INDIM/8] as uint64 words
// INDIM: number of rows/cols (M or N), K: reduction dimension
void load_scale_factors(volatile uint64_t *sf_mem,
                        uint8_t *scale_factors,
                        int INDIM, int K) {
    for (size_t k = 0; k < K / 32; k++) {
        for (size_t i = 0; i < INDIM / 8; i++) {
            sf_mem[k * INDIM/8 + i] =
                ((uint64_t*) scale_factors)[k * INDIM/8 + i];
        }
    }
}
```

```c
// Gemmini accelerator setup, tiled FP8 MVIN, loop_ws_spad compute dispatch,
// and BF16-packed result readback from scratchpad shared memory.
void run_fp8_matmul_tiled(out_t C_hw[MATMUL_M][OUT_COLS]) {
    int tiles_I = MATMUL_M / DIM;
    int tiles_J = MATMUL_N / DIM;
    int tiles_K = MATMUL_K / DIM;

    uint32_t a_base   = 0;
    uint32_t b_base   = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    uint32_t acc_addr = (1u << (ADDR_LEN - 1));

    // ---- Accelerator global config ----
    gemmini_flush(0);
    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 0, 0, 3, 0);

    // ---- Load e8m0 scale factors into SF SRAM via MMIO ----
    // A scales: SF_MEM+0x2000, B scales: SF_MEM+0x0
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                       (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                       (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);

    // ---- MVIN A: tile (i,k) -> scratchpad a_base + (i*tiles_K + k)*DIM ----
    // Stride = MATMUL_M bytes (row-major, fp8 elements)
    gemmini_config_ld(MATMUL_M * sizeof(elem_t));

    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }

    // ---- MVIN B: tile (k,j) -> scratchpad b_base + (j*tiles_K + k)*DIM ----
    for (int j = 0; j < tiles_J; j++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }

    int SPAD_DEST = 128;  // scratchpad destination row for C output

    // Output stride: OUT_COLS packed BF16 words per row
    gemmini_config_st(OUT_COLS * sizeof(out_t));

    // Configure MX quantized mvout: scale_factors buf, tile grid, no bias/relu
    uint32_t scale_factors[512] = {0};
    gemmini_mxquant_config_mvout(
        (uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

    // ---- Dispatch tiled WS matmul entirely in scratchpad ----
    // a_base: A tiles start, BANK_NUM*BANK_ROWS: B tiles end (b_base derived),
    // SPAD_DEST: C output row, 0x38: fp8 e4m3 datatype selector
    gemmini_loop_ws_spad(
        tiles_I, tiles_J, tiles_K,
        0, 0, 0,
        a_base,
        BANK_NUM * BANK_ROWS,
        0,
        SPAD_DEST,
        false, false,
        false, false, false,
        NO_ACTIVATION,
        0, 0,
        false,
        0x38);  // dtype=fp8 e4m3

    // ---- Read C results from scratchpad (shared memory) ----
    // SMEM base + SPAD_DEST*2 uint64 words (each spad row = 2x uint64 = 16B)
    uint64_t *smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            C_hw[i][j] = *(smem_start_addr + (i * OUT_COLS + j));
        }
    }

    gemmini_fence();
}
```

```c
// Verify BF16-packed hardware output against golden reference.
// C_out_bf16[M][N]: golden uint16 values, row-major.
// C_hw[M][OUT_COLS]: hardware output, 4x BF16 packed per uint64 word.
int verify_bf16_packed(out_t C_hw[MATMUL_M][OUT_COLS]) {
    int errors = 0;

    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            uint64_t got = C_hw[i][j];

            // Pack 4 consecutive bf16 golden values into expected uint64
            uint64_t exp =
                ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                ((uint64_t)C_out_bf16[i][j*4 + 0]);

            if (got != exp) {
                for (int lane = 0; lane < BF16_PER_WORD; lane++) {
                    uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
                    uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
                    if (got_bf16 != exp_bf16) {
                        errors++;
                    }
                }
            }
        }
    }
    return errors;
}
```

## matmul_tiled_fp8_64x64_requant.c

SUMMARY: Demonstrates MMIO-driven MX-Gemmini accelerator control from SIMT, covering scale factor loading, tiled FP8 matrix multiply with warp-specialized loop_ws_spad dispatch, scratchpad address management, and BF16-packed result readback with requantization verification.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

// MMIO register map for Gemmini control block at 0x40084000
#define GEMMINI_CTRL      0x40084000
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

// Scale factor memory regions (double-buffer: base / base+0x2000)
#define GEMMINI_SF_MEM   0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B  GEMMINI_SF_MEM

// Accelerator scratchpad is cluster shared memory
#define SMEM 0x40000000

#define DIM      16
#define ADDR_LEN 32

// Override RoCC instruction macro to issue via MMIO stores
// (used when not running under Spike simulation)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                        \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                        \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) =                               \
        (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
```

```c
// Load e8m0 scale factors into Gemmini SF memory region.
// n: total number of scale factor bytes; written as packed 0x7f sentinel words.
// sf_mem: pointer to MMIO-mapped scale factor SRAM (SF_MEM_A or SF_MEM_B).
// Double-buffer selection: rs1[60]/rs1[61] select odd-tile buffer at SF_MEM+0x800.
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
    for (size_t i = 0; i < n / 8; i++) {
        sf_mem[i] = 0x7f7f7f7f7f7f7f7f;
    }
}
```

```c
// Warp-specialized Gemmini manager: configure, stage operands, issue loop_ws_spad.
// Worker warps load A/B tiles into scratchpad before this point via mvin.
// SPAD_DEST=128 is the scratchpad row where C output tiles will be written.
void gemmini_manager_dispatch(
    int tiles_I, int tiles_J, int tiles_K,
    uint32_t a_base, uint32_t b_base,
    uint32_t *scale_factors)
{
    int SPAD_DEST = 128;

    // Configure systolic array: weight-stationary, identity accumulator scale
    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 0, 0, 0, 0);

    // Load per-32-K-block e8m0 scales for A (row) and B (col) into SF SRAM
#ifdef SPIKE_SIM
    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A,
                       (uint8_t *) &A_scales_row, 1024);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B,
                       (uint8_t *) &B_scales_col, 1024);
#endif

    // Configure store stride (packed 4x BF16 per 64-bit word)
    gemmini_config_st(1 * sizeof(uint64_t));

    // Configure MX requantization mvout: scale_factors array, tile counts,
    // K-tile depth, format selects (0=fp8 output, 1=requant enabled)
    gemmini_mxquant_config_mvout(
        (uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

    // Issue tiled weight-stationary loop over scratchpad:
    //   a_base: SP row for A tiles
    //   BANK_NUM*BANK_ROWS: SP row for B tiles (end of bank)
    //   SPAD_DEST: SP row for C output
    //   0x38: MX format tag (fp8 e4m3)
    gemmini_loop_ws_spad(
        tiles_I, tiles_J, tiles_K,
        0, 0, 0,
        a_base,
        BANK_NUM * BANK_ROWS,
        0,
        SPAD_DEST,
        false, false,
        false, false, false,
        NO_ACTIVATION,
        0, 0,
        false,
        0x38);

    gemmini_fence();
}
```

```c
// Stage A tiles into scratchpad via mvin (worker warp role).
// Layout: A_in is row-major [MATMUL_M][MATMUL_K] fp8 bytes.
// Tile (i,k) -> SP address a_base + (i*tiles_K + k)*DIM.
void mvin_A_tiles(
    uint8_t *A_in, uint32_t a_base,
    int tiles_I, int tiles_K)
{
    gemmini_config_ld(MATMUL_M * sizeof(uint8_t));
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            uint8_t *dram_ptr = A_in + i * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }
}

// Stage B tiles into scratchpad via mvin (worker warp role).
// Layout: B_in is col-major [MATMUL_N][MATMUL_K] fp8 bytes.
// Tile (k,j) -> SP address b_base + (j*tiles_K + k)*DIM.
void mvin_B_tiles(
    uint8_t *B_in, uint32_t b_base,
    int tiles_J, int tiles_K)
{
    for (int j = 0; j < tiles_J; j++) {
        for (int k = 0; k < tiles_K; k++) {
            uint8_t *dram_ptr = B_in + j * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr  = b_base + (j * tiles_K + k) * DIM;
            gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
        }
    }
}
```

```c
// Read C result tiles back from scratchpad (shared memory) after gemmini_fence.
// SPAD_DEST=128; each SP row is 16 bytes; output is packed 4x BF16 per uint64_t.
// smem_start_addr points to SMEM + SPAD_DEST * 2 (byte offset = SPAD_DEST * 32).
void readback_C_from_smem(
    uint64_t C_hw[][MATMUL_N / 8],
    int SPAD_DEST)
{
#ifdef SPIKE_SIM
    gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N / 2);
#else
    uint64_t *smem_start_addr = ((uint64_t *)SMEM) + SPAD_DEST * 2;
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < MATMUL_N / 8; j++) {
            C_hw[i][j] = *(smem_start_addr + (i * MATMUL_N / 8 + j));
        }
    }
#endif
}
```

```c
// Bit-exact verification of requantized fp8 output against reference.
// Counts mismatches and bins by Hamming distance (1-bit, 2-bit, 3+-bit).
int verify_fp8_output(uint64_t C_hw[][MATMUL_N / 8]) {
    int errors    = 0;
    int diff1     = 0, diff2 = 0, diff3plus = 0;
    uint8_t *hw_bytes = (uint8_t *)C_hw;

    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < MATMUL_N; j++) {
            uint8_t got = hw_bytes[i * MATMUL_N + j];
            uint8_t exp = C_out[i][j];
            if (got != exp) {
                errors++;
                // Hamming distance binning for MX rounding analysis
                uint8_t diff = got ^ exp;
                int bits = 0;
                for (uint8_t x = diff; x; x >>= 1) bits += x & 1;
                if      (bits == 1) diff1++;
                else if (bits == 2) diff2++;
                else                diff3plus++;
            }
        }
    }
    return errors;
}
```

## autocomp_harness_test0.c

SUMMARY: Demonstrates MX-Gemmini accelerator driver API usage from a SIMT/MMIO context, including scale factor loading, packed BF16 output construction, and the MMIO command interface for issuing RoCC-style instructions to the systolic array.

```c
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define GEMMINI_SF_MEM      0x40088000
#define GEMMINI_SF_MEM_A    (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B    GEMMINI_SF_MEM
#define GEMMINI_CTRL        0x40084000
#define GEMMINI_RS1_ADDR    (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR    (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR   (GEMMINI_CTRL + 0x0)

// MMIO RoCC instruction issue: write RS1, RS2, then instruction word
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                        \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                        \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                        \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) =                               \
        (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
```

```c
// Load e8m0 scale factors into Gemmini SF MMIO region.
// sf_mem: pointer to GEMMINI_SF_MEM_A or GEMMINI_SF_MEM_B
// scale_factors: flat array of uint8_t scales, shape [K/32][INDIM/8] packed as uint64_t
// INDIM: number of rows (A) or columns (B)
// K: reduction dimension
void load_scale_factors(volatile uint64_t *sf_mem,
                        uint8_t *scale_factors,
                        int INDIM, int K) {
    for (size_t k = 0; k < K / 32; k++) {
        for (size_t i = 0; i < INDIM / 8; i++) {
            sf_mem[k * INDIM / 8 + i] =
                ((uint64_t *) scale_factors)[k * INDIM / 8 + i];
        }
    }
}
```

```c
#define BF16_PER_WORD 4
#define OUT_COLS      (MATMUL_M / BF16_PER_WORD)

typedef uint64_t out_t;   // 4x bf16 packed per 64-bit word

// Pack four consecutive uint16_t BF16 values into one uint64_t output word.
// C_out_bf16: reference output, shape [MATMUL_M][MATMUL_N], each entry uint16_t
// gold:       packed output,    shape [MATMUL_M][OUT_COLS],  each entry uint64_t
void pack_bf16_output(out_t gold[MATMUL_M][OUT_COLS]) {
    for (int i = 0; i < MATMUL_M; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            gold[i][j] =
                ((uint64_t) C_out_bf16[i][j * 4 + 3] << 48) |
                ((uint64_t) C_out_bf16[i][j * 4 + 2] << 32) |
                ((uint64_t) C_out_bf16[i][j * 4 + 1] << 16) |
                ((uint64_t) C_out_bf16[i][j * 4 + 0]);
        }
    }
}
```

```c
// Bit-exact equality check over packed BF16 output matrix.
int full_is_equal(out_t x[MATMUL_M][OUT_COLS],
                  out_t y[MATMUL_M][OUT_COLS]) {
    for (int i = 0; i < MATMUL_M; i++)
        for (int j = 0; j < OUT_COLS; j++)
            if (x[i][j] != y[i][j])
                return 0;
    return 1;
}
```

```c
// Scratchpad address layout for weight-stationary tiled matmul.
// DIM=16 systolic array; tiles_K * tiles_J B-tiles packed at top of scratchpad.
#define DIM      16
#define ADDR_LEN 32

void compute_spad_layout(int *a_base_out, int *b_base_out,
                         uint32_t *acc_addr_out,
                         int tiles_I, int tiles_J, int tiles_K) {
    int      a_base   = 0;
    int      b_base   = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    uint32_t acc_addr = (1u << (ADDR_LEN - 1));   // accumulator address space bit

    *a_base_out   = a_base;
    *b_base_out   = b_base;
    *acc_addr_out = acc_addr;
}
```

## example_vecadd_ilp.hpp

SUMMARY: Demonstrates a warp-parallel vector addition kernel using SIMT intrinsics with ILP (instruction-level parallelism) unrolling, warp-based chunking, and multi-cluster scheduling via mu_schedule on a Muon SIMT architecture.

```cpp
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

struct VecAddArgs {
  __global float* A;
  __global float* B;
  __global float* C;
  uint32_t n;
};

template <uint32_t ILP>
static inline void vecadd_impl(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<VecAddArgs*>(raw_arg);

  constexpr uint32_t kIlp = ILP;
  constexpr uint32_t kWarpWidth = MU_NUM_THREADS;
  constexpr uint32_t kOuter = VECADD_OUTER_UNROLL;
  const uint32_t tid_in_warp = tid_in_threadblock % kWarpWidth;
  const uint32_t warp_id = tid_in_threadblock / kWarpWidth;
  const uint32_t warps_per_threadblock = threads_per_threadblock / kWarpWidth;
  const uint32_t elems_per_chunk = kWarpWidth * kIlp;
  uint32_t chunk;
  uint32_t chunk_stride;

#if MU_NUM_CLUSTERS == 1
  (void)threadblock_id;
  chunk = warp_id * elems_per_chunk;
  chunk_stride = warps_per_threadblock * elems_per_chunk;
#else
  const uint32_t global_warp_id = threadblock_id * warps_per_threadblock + warp_id;
  const uint32_t global_warp_stride = MU_NUM_CLUSTERS * warps_per_threadblock;
  chunk = global_warp_id * elems_per_chunk;
  chunk_stride = global_warp_stride * elems_per_chunk;
#endif

  const uint32_t outer_stride = chunk_stride * kOuter;
  float a_vals[kIlp];
  float b_vals[kIlp];
  float c_vals[kIlp];

  #pragma unroll 1
  for (; chunk < args->n; chunk += outer_stride) {
    #pragma unroll VECADD_OUTER_UNROLL
    for (uint32_t outer = 0; outer < kOuter; ++outer) {
      const uint32_t lane_base = chunk + outer * chunk_stride + tid_in_warp;

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        a_vals[i] = args->A[idx];
        b_vals[i] = args->B[idx];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        c_vals[i] = a_vals[i] + b_vals[i];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        args->C[idx] = c_vals[i];
      }
    }
  }
}

static inline void vecadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  vecadd_impl<VECADD_ILP>(
    raw_arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}
```

## mx_format_test.c

SUMMARY: Demonstrates MX-format configuration API usage for the Gemmini accelerator, specifically how to set microscaling floating-point formats (FP8/FP6/FP4) for activation, weight, and output operands via `gemmini_extended3_config_ex`, including the `uselut` control bit.

```c
#include "include/gemmini_testutils.h"

// MX format encoding: FP8=0, FP6=1, FP4=2

// Set all formats to FP8 (0)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);
gemmini_fence();

// Set all formats to FP6 (1)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 1, 1);
gemmini_fence();

// Set all formats to FP4 (2)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 2, 0);
gemmini_fence();

// Mixed formats: act=FP8(0), wt=FP6(1), out=FP4(2)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 1, 3, 0);
gemmini_fence();

// Mixed formats: act=FP4(2), wt=FP8(0), out=FP6(1)
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 0, 1, 0);
gemmini_fence();

// With uselut=1 enabled
gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 1);
gemmini_fence();
```

```c
// MMIO command issue macro for driving Gemmini over shared MMIO at 0x84000
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_RS1_ADDR  (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR  (GEMMINI_CTRL + 0x18)

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) {                          \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1);                          \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2);                          \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B)   |                      \
                                                (0 << 7)  |                      \
                                                (3 << 12) |                      \
                                                (1 << 15) |                      \
                                                (2 << 20) |                      \
                                                ((funct) << 25);                 \
}
```

## autocomp_baseline_sol0.c

SUMMARY: Demonstrates MX-Gemmini fp8 weight-stationary matrix multiplication kernel API usage, including accelerator configuration, scale factor loading, scratchpad tile addressing, MVIN/MVOUT operations, and the WS compute loop with MX requantization to bf16 output.

```c
// Configure execution: weight-stationary, MX requantize to bf16 output.
gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

// Load per-group scale factors into the scale-factor memory (A=0, B=1).
gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
```

```c
// MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM
gemmini_config_ld(MATMUL_M * sizeof(elem_t));
for (int i = 0; i < tiles_I; i++) {
  for (int k = 0; k < tiles_K; k++) {
    elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
    uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}
```

```c
// MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM
for (int j = 0; j < tiles_J; j++) {
  for (int k = 0; k < tiles_K; k++) {
    elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
    uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
    gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
  }
}
```

```c
// Configure store + MX requantizer for the mvout, then run the WS compute loop.
gemmini_config_st(OUT_COLS * sizeof(out_t));
gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
gemmini_loop_ws_spad(
    tiles_I, tiles_J, tiles_K,
    0, 0, 0,
    a_base,
    BANK_NUM * BANK_ROWS,
    0,
    SPAD_DEST,
    false, false,
    false, false, false,
    NO_ACTIVATION,
    0, 0,
    false,
    0x38);
```

```c
// Read the bf16-packed result out of shared memory into C_hw.
gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
gemmini_fence();
```

## example_warpspec_simt_contention.cpp

SUMMARY: Demonstrates warp specialization with SIMT contention on shared memory banks, where warp 0 acts as a Gemmini manager driving MX-GEMM via MMIO and warp 1 introduces synthetic SMEM read contention by round-robining through shared memory banks.

```cpp
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m64n64k512.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

void simt_contention_entry(void *arg, uint32_t tid_in_threadblock,
                             uint32_t threads_per_threadblock,
                             uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    auto dummy_gmem = reinterpret_cast<uint8_t *>(0x60000000);

    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;

    if (warp_id == 0) {
        // Gemmini manager warp: issues MX-GEMM over MMIO
        const auto threads_in_warpgroup = MU_NUM_THREADS * 1;
        const auto tid_in_warpgroup = tid_in_threadblock % MU_NUM_THREADS;
        mxgemm<C>(C.TILE_M, C.TILE_N, 512, C_gmem, tid_in_warpgroup,
                  threads_in_warpgroup, threadblock_id);
    } else if (warp_id == 1) {
        // SMEM read worker warp: introduces synthetic SMEM bank contention
        // by round-robining reads across 4 SMEM banks
        const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;

        constexpr auto SMEM_BANK_SIZE = MU_SMEM_SIZE_BYTES / 4;
#pragma unroll 32
        for (int i = 0; i < 1024 * 32; i++) {
            auto dummy_smem_base =
                reinterpret_cast<const volatile __shared uint8_t *>(
                    SMEM_BANK_SIZE * (i % 4));
            auto dummy_smem_addr =
                reinterpret_cast<const volatile __shared uint32_t *>(
                    dummy_smem_base) +
                tid_in_warp;
            auto dummy = *dummy_smem_addr;
        }
    }
}
```

## example_combined_baseline.cpp

SUMMARY: Demonstrates a baseline Gemmini-accelerated GEMM kernel using warp-specialized orchestration with software-pipelined K-tiles (TILE_K=64, 8 iterations for K=512), showing the `mxgemm` API entry point and `GemmConfig` configuration for FP8 microscaling accumulation into a 64x64 output tile in shared memory.

```cpp
void kernel_body(void *raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  constexpr GemmConfig GEMM_CFG{
      .TILE_M = MATMUL_M,
      .TILE_N = MATMUL_N,
      .TILE_K = 64, // fixed: 8 K-tiles for K=512
      .DATATYPE = GemmDatatype::FP8,
      .QUANT_OUTPUT = false,
  };
  auto *arg = reinterpret_cast<KernelArgs *>(raw_arg);
  mxgemm<GEMM_CFG>(arg->M, arg->N, arg->K, arg->C, tid_in_threadblock,
                   threads_per_threadblock, threadblock_id);
}
```