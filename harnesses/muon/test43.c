#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2      // mxgemm_lib is warp-specialized for 2 warps
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;

#include "data"

static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_lib_param.hpp"

#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
static inline void drain(){
  for (volatile uint32_t d = 0; d < DRAIN_ITERS; d++) { asm volatile("" ::: "memory"); }
}

// gate/up: [M x FN] = [M x HID] @ [HID x FN], fp4, K-tiled by 64.
constexpr GemmConfig GU_CFG{ .TILE_M = FFN_M, .TILE_N = FFN_N, .TILE_K = 64,
                             .DATATYPE = GemmDatatype::FP4, .QUANT_OUTPUT = false };
// down: [M x T] = [M x FN] @ [FN x T], fp4, single K-tile (TILE_K == FN).
constexpr GemmConfig DN_CFG{ .TILE_M = FFN_M, .TILE_N = FFN_T, .TILE_K = FFN_N,
                             .DATATYPE = GemmDatatype::FP4, .QUANT_OUTPUT = false };

// --- libm-free fp helpers ---------------------------------------------------
static inline uint32_t f2b(float f){ union{float f; uint32_t u;} v; v.f=f; return v.u; }
static inline float    b2f(uint32_t u){ union{float f; uint32_t u;} v; v.u=u; return v.f; }
static inline float    bf16_to_f(uint16_t h){ return b2f((uint32_t)h << 16); }

// f32 -> bf16 bits, RNE, mirroring mx_fp_math.h f32_to_bf16_rne (matches gen).
static inline uint16_t f32_to_bf16_rne(float x){
  if (x == 0.0f) return 0;
  uint32_t bits = f2b(x);
  if (((bits >> 23) & 0xFF) == 0xFF){
    uint32_t bf = bits >> 16;
    if ((bits & 0x7FFFFFu) && !(bf & 0x40)) bf |= 0x40;
    return (uint16_t)(bf & 0xFFFF);
  }
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounded = bits + 0x7FFFu + lsb;
  uint16_t out = (uint16_t)((rounded >> 16) & 0xFFFF);
  if ((out & 0x7FFF) == 0) return 0;
  return out;
}

// bf16 bits -> fp4 e2m1 code, mirroring mx_fp_math.h bf16_bits_to_fp4_e2m1_code (matches gen).
static inline uint8_t bf16_bits_to_fp4_e2m1_code(uint16_t bf16){
  int sign_bit = (bf16 >> 15) & 1;
  int E = (bf16 >> 7) & 0xFF;
  int Mm = bf16 & 0x7F;
  if (E == 0) return 0;
  if (E == 255) return (uint8_t)((sign_bit << 3) | 0x7);
  int e = E - 127;
  float e3m1_mag;
  if (e >= -2 && e <= 3){
    int S = 128 + Mm;
    int q = (S >> 6) & 1;
    int r = (S >> 5) & 1;
    int sticky = (S & 0x1F) != 0;
    int round_up = r & (sticky | q);
    int sig2 = q + round_up;
    int mant_out = (sig2 >= 2) ? 0 : sig2;
    int exp_out  = (sig2 >= 2) ? e + 1 : e;
    if (exp_out > 3) return (uint8_t)((sign_bit << 3) | 0x7);
    e3m1_mag = (1.0f + mant_out * 0.5f) * b2f((uint32_t)((exp_out + 127) << 23));
  } else if (e == -3){
    e3m1_mag = (Mm >= 64) ? 0.25f : 0.125f;
  } else if (e == -4){
    e3m1_mag = (Mm > 0) ? 0.125f : 0.0f;
  } else {
    e3m1_mag = 0.0f;
  }
  if (e3m1_mag == 0.0f || e3m1_mag == 0.125f || e3m1_mag == 0.25f) return 0;
  uint8_t mag;
  if (e3m1_mag == 0.375f || e3m1_mag == 0.5f) mag = 0x1;
  else if (e3m1_mag == 0.75f || e3m1_mag == 1.0f) mag = 0x2;
  else if (e3m1_mag == 1.5f) mag = 0x3;
  else if (e3m1_mag == 2.0f) mag = 0x4;
  else if (e3m1_mag == 3.0f) mag = 0x5;
  else if (e3m1_mag == 4.0f) mag = 0x6;
  else mag = 0x7;
  return (uint8_t)((sign_bit << 3) | mag);
}

static inline uint8_t fp4_encode(float x){ return bf16_bits_to_fp4_e2m1_code(f32_to_bf16_rne(x)); }

// Strip the __global address-space qualifier to a generic pointer (same address) so
// __global operands can feed the generic-pointer mxgemm signature.  (Same trick the
// autocomp mx kernels use: round-trip through the 32-bit address.)
static inline const uint8_t* strip8(const __global uint8_t* p){
  return reinterpret_cast<const uint8_t*>(reinterpret_cast<uint32_t>(p));
}

// e8m0 block-scale code: normalize block amax to [1,2) (exp = floor(log2 amax)). amax > 0.
static inline uint8_t block_scale_code(float amax){
  uint32_t bits = f2b(amax);
  uint32_t ef = (bits >> 23) & 0xFF;
  if (ef == 0) return 0;
  int e = (int)ef - 127;
  int sc = e + 127;
  return (uint8_t)(sc < 0 ? 0 : (sc > 254 ? 254 : sc));
}

// rsqrt (fast-inverse-sqrt + 3 Newton) and silu -- identical to autocomp_ffn_block.
static inline float my_rsqrt(float x){
  const float xhalf = 0.5f * x;
  uint32_t i = f2b(x);
  i = 0x5f3759dfu - (i >> 1);
  float y = b2f(i);
  y = y * (1.5f - xhalf * y * y);
  y = y * (1.5f - xhalf * y * y);
  y = y * (1.5f - xhalf * y * y);
  return y;
}
static inline float mu_exp(float x){
  const float L=1.4426950408889634f, N2=0.6931471805599453f;
  float t=x*L;
  // branchless copysign(0.5, t): avoids a SIMT divergent split (vx_split_n) in this hot path
  // -- a sub-word GMEM load landing inside such a split livelocks the cyclotron mem pipe.
  // Bit-exact with the (t>=0?0.5:-0.5) form for every reachable t (they differ only at t==-0.0,
  // where (int) truncation makes both k==0).
  float half = b2f((f2b(t) & 0x80000000u) | 0x3f000000u);
  int k=(int)(t+half);
  float r=x-(float)k*N2;
  float p=1.0f+r*(1.0f+r*(0.5f+r*(0.16666667f+r*(0.041666668f+r*0.008333334f))));
  union{uint32_t i;float f;}s; s.i=(uint32_t)((k+127)<<23); return p*s.f;
}
static inline float silu(float x){ return x/(1.0f+mu_exp(-x)); }

// SUBSTITUTE HERE
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
// SUBSTITUTE END

// ===================== FIXED verify + main (not optimizable) =================
#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};
static void verify_body(void*, uint32_t tid, uint32_t tpb, uint32_t){
  uint32_t errors = 0;
  for (uint32_t i = tid; i < VERIFY_COUNT; i += tpb)
    if (out_raw[i] != gold_raw[i]) errors++;
  lane_errors[tid] = errors;
  mu_fence();
}
static inline uint32_t hart_id(){
  uint32_t id; asm volatile("csrr %0, mhartid" : "=r"(id)::"memory"); return id;
}
int main(){
  run_ffn();
  mu_schedule(verify_body, nullptr, 1); mu_barrier(0, MU_NUM_CORES);
  if (hart_id() != 0) { for (;;) {} }
  mu_fence();
  uint32_t total = 0;
  for (uint32_t t = 0; t < VERIFY_LANES; t++) total += lane_errors[t];
  uint32_t code = total ? ((total << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
