static inline void copy_C_to(__global uint16_t* dst, uint32_t nelem, uint32_t spad_dest,
                             uint32_t tid, uint32_t tpb){
  auto* smem32 = reinterpret_cast<const __shared uint32_t*>(spad_dest * DIM);
  auto* out32  = reinterpret_cast<__global uint32_t*>(dst);
  const uint32_t nwords = nelem / 2;
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid; i < nwords; i += tpb) out32[i] = smem32[i];
}

__attribute__((noinline))
static void quantize_fp4(const __global uint16_t* V, uint32_t Mr, uint32_t K,
                                __global uint8_t* A_fp4, __global uint8_t* scales,
                                uint32_t tid, uint32_t tpb){
  const uint32_t GK = K / FFN_GROUP;
  const uint32_t ntasks = (Mr/2) * GK;
  #pragma clang loop unroll(disable)
  for (uint32_t t = tid; t < ntasks; t += tpb){
    const uint32_t i = t / GK, g = t % GK;
    const uint32_t m0 = 2*i, m1 = 2*i + 1;
    float amax0 = 0.0f, amax1 = 0.0f;
    for (uint32_t j = 0; j < FFN_GROUP; j++){
      const uint32_t k = g*FFN_GROUP + j;
      float a0 = bf16_to_f(V[m0*K + k]); a0 = a0 < 0 ? -a0 : a0;
      float a1 = bf16_to_f(V[m1*K + k]); a1 = a1 < 0 ? -a1 : a1;
      if (a0 > amax0) amax0 = a0;
      if (a1 > amax1) amax1 = a1;
    }
    const uint8_t sc0 = block_scale_code(amax0), sc1 = block_scale_code(amax1);
    scales[g*Mr + m0] = sc0; scales[g*Mr + m1] = sc1;
    const float scale0 = b2f((uint32_t)(127 + ((int)sc0 - 127)) << 23);
    const float scale1 = b2f((uint32_t)(127 + ((int)sc1 - 127)) << 23);
    for (uint32_t j = 0; j < FFN_GROUP; j++){
      const uint32_t k = g*FFN_GROUP + j;
      uint8_t c0 = (amax0 == 0.0f) ? 0 : fp4_encode(bf16_to_f(V[m0*K + k]) / scale0);
      uint8_t c1 = (amax1 == 0.0f) ? 0 : fp4_encode(bf16_to_f(V[m1*K + k]) / scale1);
      A_fp4[i*K + k] = (uint8_t)((c1 << 4) | (c0 & 0xF));
    }
  }
  // mu_fence() removed: the run_ffn orchestration already issues
  // mu_barrier(0, MU_NUM_CORES) + mu_fence() after every mu_schedule,
  // which provides the required cross-core visibility before any
  // consumer kernel reads the quantized output. The per-thread fence
  // here was redundant and executed across all 64 threads unnecessarily.
}

static void quant_xn_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  quantize_fp4(Xn_store, FFN_M, FFN_HID, Xn_fp4, Xn_scales, tid, tpb);
}

static void quant_h_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  quantize_fp4(h_store, FFN_M, FFN_N, h_fp4, h_scales, tid, tpb);
}

static void rmsnorm_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t Mr = FFN_M, K = FFN_HID;
  #pragma clang loop unroll(disable)
  for (uint32_t m = tid; m < Mr; m += tpb){
    float ss = 0.0f;
    for (uint32_t k = 0; k < K; k++){ float x = bf16_to_f(X_bf16[m][k]); ss += x*x; }
    const float rms = my_rsqrt(ss/(float)K + RMS_EPS);
    for (uint32_t k = 0; k < K; k++){
      float xn = bf16_to_f(X_bf16[m][k]) * rms * bf16_to_f(gamma_bf16[k]);
      Xn_store[m*K + k] = f32_to_bf16_rne(xn);
    }
  }
  mu_fence();
}

// GATE_SMEM_STAGING: SMEM byte address for staging gate BF16 results between
// gate_body and swiglu_body.
// Gate tile size: FFN_M * FFN_N * sizeof(uint16_t) = 128*128*2 = 32768 bytes = 32 KiB.
// Placed at 0x10000 (64 KiB), occupying 0x10000..0x17FFF.
// Gemmini A tiles: rows 0..~255 -> bytes 0x0000..0x0FFF (safe below).
// Gemmini C accumulator: row 256 -> byte 0x1000, 32 KiB -> 0x1000..0x8FFF (safe below).
// B tiles grow down from top of 128 KiB SMEM (0x20000), well above 0x17FFF.
// No overlap with Gemmini scratchpad regions used by GU_CFG or DN_CFG.
static constexpr uint32_t GATE_SMEM_STAGING = 0x10000; // 64 KiB offset in SMEM

static void gate_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t nw = tpb / MU_NUM_THREADS;
  // Run gate matmul; result deposited as BF16 in SMEM at GU_CFG.SPAD_DEST()*DIM.
  mxgemm_single_output_tile<GU_CFG>(strip8(&Xn_fp4[0]), &Wg_fp4[0][0],
                                    strip8(&Xn_scales[0]), &Wg_scales[0][0],
                                    FFN_M, FFN_N, FFN_HID, tid, tpb);
  // Barrier: all threads see matmul completion (gemmini_fence inside mxgemm_single_output_tile).
  mu_barrier(1, nw);
  // Copy gate BF16 results from Gemmini SMEM accumulator to SMEM staging area.
  // Replaces expensive GMEM write (~400-cycle latency) with cheap SMEM copy (~2-cycle latency).
  auto* src_smem = reinterpret_cast<const __shared uint32_t*>(
      static_cast<uintptr_t>(GU_CFG.SPAD_DEST()) * DIM);
  auto* dst_smem = reinterpret_cast<__shared uint32_t*>(
      static_cast<uintptr_t>(GATE_SMEM_STAGING));
  // FFN_M * FFN_N BF16 elements = FFN_M * FFN_N / 2 uint32 words.
  const uint32_t nwords = (FFN_M * FFN_N) / 2;
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid; i < nwords; i += tpb){
    dst_smem[i] = src_smem[i];
  }
  // Fence: ensure all SMEM staging writes are visible before up_body overwrites
  // the Gemmini accumulator region at GU_CFG.SPAD_DEST()*DIM.
  mu_fence();
}

static void up_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t nw = tpb / MU_NUM_THREADS;
  // Run up matmul; result deposited as BF16 in SMEM at GU_CFG.SPAD_DEST()*DIM.
  // gate_body has completed and copied gate to GATE_SMEM_STAGING before this runs
  // (enforced by inter-kernel mu_barrier(0, MU_NUM_CORES) + mu_fence() in run_ffn).
  mxgemm_single_output_tile<GU_CFG>(strip8(&Xn_fp4[0]), &Wu_fp4[0][0],
                                    strip8(&Xn_scales[0]), &Wu_scales[0][0],
                                    FFN_M, FFN_N, FFN_HID, tid, tpb);
  // Leave up results in SMEM at GU_CFG.SPAD_DEST()*DIM; swiglu_body reads directly.
  mu_barrier(1, nw);
  mu_fence();
}

// swiglu_body: reads gate from SMEM staging area (GATE_SMEM_STAGING, ~2-cycle latency),
// reads up directly from SMEM at GU_CFG.SPAD_DEST()*DIM (~2-cycle latency).
// Both reads are now SMEM, eliminating all GMEM traffic for gate.
// Widened loop: each thread processes 2 uint32 words (4 BF16 elements) per iteration,
// halving loop control overhead and improving memory access coalescing.
static void swiglu_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  // Gate results staged in SMEM by gate_body (no GMEM round-trip).
  auto* g32    = reinterpret_cast<const __shared uint32_t*>(
      static_cast<uintptr_t>(GATE_SMEM_STAGING));
  // Up results left in SMEM by up_body (Gemmini accumulator output, BF16 packed).
  auto* u_smem = reinterpret_cast<const __shared uint32_t*>(
      static_cast<uintptr_t>(GU_CFG.SPAD_DEST()) * DIM);
  auto* h32    = reinterpret_cast<__global uint32_t*>(h_store);
  const uint32_t nwords  = (FFN_M * FFN_N) / 2;
  // Round down to nearest multiple of 2*tpb for the widened main loop.
  const uint32_t nwords2 = (nwords / (2 * tpb)) * (2 * tpb);
  // Main loop: each thread processes 2 uint32 words (4 BF16 elements) per iteration.
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid; i < nwords2; i += 2 * tpb){
    uint32_t gw0 = g32[i];
    uint32_t gw1 = g32[i + tpb];
    uint32_t uw0 = u_smem[i];
    uint32_t uw1 = u_smem[i + tpb];
    uint16_t lo0 = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)gw0))         * bf16_to_f((uint16_t)uw0));
    uint16_t hi0 = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)(gw0 >> 16))) * bf16_to_f((uint16_t)(uw0 >> 16)));
    uint16_t lo1 = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)gw1))         * bf16_to_f((uint16_t)uw1));
    uint16_t hi1 = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)(gw1 >> 16))) * bf16_to_f((uint16_t)(uw1 >> 16)));
    h32[i]       = ((uint32_t)hi0 << 16) | lo0;
    h32[i + tpb] = ((uint32_t)hi1 << 16) | lo1;
  }
  // Tail loop: scalar fallback for any remainder elements.
  #pragma clang loop unroll(disable)
  for (uint32_t i = nwords2 + tid; i < nwords; i += tpb){
    uint32_t gw = g32[i];
    uint32_t uw = u_smem[i];
    uint16_t lo = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)gw))         * bf16_to_f((uint16_t)uw));
    uint16_t hi = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)(gw >> 16))) * bf16_to_f((uint16_t)(uw >> 16)));
    h32[i] = ((uint32_t)hi << 16) | lo;
  }
  mu_fence();
}

static void down_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t nw = tpb / MU_NUM_THREADS;
  const uint32_t spad = DN_CFG.SPAD_DEST() * DIM;
  for (uint32_t jt = 0; jt < FFN_DN_TILES; jt++){
    mxgemm_single_output_tile<DN_CFG>(strip8(&h_fp4[0]), &Wd_fp4_tiled[jt][0][0],
                                      strip8(&h_scales[0]), &Wd_scales_tiled[jt][0][0],
                                      FFN_M, FFN_T, FFN_N, tid, tpb);
    mu_barrier(1, nw);
    auto* smem16 = reinterpret_cast<const __shared uint16_t*>(spad);
    #pragma clang loop unroll(disable)
    for (uint32_t idx = tid; idx < FFN_M * FFN_T; idx += tpb){
      const uint32_t m = idx / FFN_T, c = idx % FFN_T;
      const uint32_t j = jt * FFN_T + c;
      float d = bf16_to_f(smem16[m*FFN_T + c]);
      float x = bf16_to_f(X_bf16[m][j]);
      out_raw[m*FFN_DN + j] = f32_to_bf16_rne(d + x);
    }
    mu_fence();
    mu_barrier(1, nw);
  }
  mu_fence();
}

// ---------------------------------------------------------------------------
//  run_ffn: orchestrate the fused FFN chain.
//  gate_body now stages gate results into SMEM (GATE_SMEM_STAGING) instead of
//  writing to gate_store GMEM, eliminating the GMEM write+read round-trip
//  (~400-cycle latency saved per access).
//  swiglu_body reads both gate (from GATE_SMEM_STAGING) and up (from Gemmini
//  accumulator SMEM) directly at ~2-cycle SMEM latency.
//  No drain() after gate_body: there is no Gemmini DMA mvout to GMEM, so
//  no DMA completion to wait for beyond what gemmini_fence inside
//  mxgemm_single_output_tile already handles.
//  The mu_fence() calls previously inside quantize_fp4 have been removed;
//  the mu_barrier(0, MU_NUM_CORES) + mu_fence() sequence here after each
//  mu_schedule already provides the required cross-core ordering guarantee
//  before any consumer kernel reads the quantized data.
// ---------------------------------------------------------------------------
static void run_ffn(){
  mu_schedule(rmsnorm_body,  nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(quant_xn_body, nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  // gate_body: matmul + SMEM-to-SMEM staging of gate results (no GMEM write, no drain needed).
  mu_schedule(gate_body,     nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence();
  // up_body: matmul, leaves up result in SMEM at GU_CFG.SPAD_DEST()*DIM (no GMEM write, no drain needed).
  // gate results are safely in GATE_SMEM_STAGING before up overwrites the accumulator region.
  mu_schedule(up_body,       nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence();
  // swiglu_body: reads gate from GATE_SMEM_STAGING, reads up from SMEM accumulator, writes h_store GMEM.
  mu_schedule(swiglu_body,   nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(quant_h_body,  nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(down_body,     nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
}
