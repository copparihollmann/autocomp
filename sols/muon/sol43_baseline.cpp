// sol43_baseline.cpp -- SUBSTITUTABLE region of the fused fp4-MX FFN block (test43).
// Contains: copy_C_to, quantize_fp4, the 7 phase bodies, and run_ffn (orchestration).
// The fixed harness supplies: includes, `data`, LUTs, mxgemm_lib_param.hpp, GemmConfigs,
// libm-free fp/fp4 helpers, and main()+verify_body (tohost). DO NOT redefine those.

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
  mu_fence();
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

static void gate_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t nw = tpb / MU_NUM_THREADS;
  mxgemm_single_output_tile<GU_CFG>(strip8(&Xn_fp4[0]), &Wg_fp4[0][0],
                                    strip8(&Xn_scales[0]), &Wg_scales[0][0],
                                    FFN_M, FFN_N, FFN_HID, tid, tpb);
  mu_barrier(1, nw);
  copy_C_to(gate_store, FFN_M*FFN_N, GU_CFG.SPAD_DEST(), tid, tpb);
  mu_fence();
}

static void up_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  const uint32_t nw = tpb / MU_NUM_THREADS;
  mxgemm_single_output_tile<GU_CFG>(strip8(&Xn_fp4[0]), &Wu_fp4[0][0],
                                    strip8(&Xn_scales[0]), &Wu_scales[0][0],
                                    FFN_M, FFN_N, FFN_HID, tid, tpb);
  mu_barrier(1, nw);
  copy_C_to(up_store, FFN_M*FFN_N, GU_CFG.SPAD_DEST(), tid, tpb);
  mu_fence();
}

static void swiglu_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  auto* g32 = reinterpret_cast<const __global uint32_t*>(gate_store);
  auto* u32 = reinterpret_cast<const __global uint32_t*>(up_store);
  auto* h32 = reinterpret_cast<__global uint32_t*>(h_store);
  const uint32_t nwords = (FFN_M * FFN_N) / 2;
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid; i < nwords; i += tpb){
    uint32_t gw = g32[i], uw = u32[i];
    uint16_t lo = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)gw))       * bf16_to_f((uint16_t)uw));
    uint16_t hi = f32_to_bf16_rne(silu(bf16_to_f((uint16_t)(gw>>16))) * bf16_to_f((uint16_t)(uw>>16)));
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
    // residual add + bf16 store for this tile's T output columns.
    auto* smem16 = reinterpret_cast<const __shared uint16_t*>(spad);
    #pragma clang loop unroll(disable)
    for (uint32_t idx = tid; idx < FFN_M * FFN_T; idx += tpb){
      const uint32_t m = idx / FFN_T, c = idx % FFN_T;
      const uint32_t j = jt * FFN_T + c;
      float d = bf16_to_f(smem16[m*FFN_T + c]);
      float x = bf16_to_f(X_bf16[m][j]);
      out_raw[m*FFN_DN + j] = f32_to_bf16_rne(d + x);
    }
    // Order this tile's out_raw stores before the next tile's gemmini operand DMA (a real fence,
    // not the GMEM-backed busy-wait drain -- that would be ~16k slow GMEM round-trips over 8 tiles).
    mu_fence();
    mu_barrier(1, nw);
  }
  mu_fence();
}

// ---------------------------------------------------------------------------
//  run_ffn: orchestrate the fused FFN chain (full VERIFY_STAGE==3 path).
//  This is the OPTIMIZATION SURFACE: schedule/warp-count/fence/drain/barrier
//  choices live here; the per-phase bodies above are the compute levers.
// ---------------------------------------------------------------------------
static void run_ffn(){
  mu_schedule(rmsnorm_body,  nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(quant_xn_body, nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(gate_body,     nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(up_body,       nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(swiglu_body,   nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(quant_h_body,  nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
  mu_schedule(down_body,     nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES); mu_fence(); drain(); mu_fence();
}
