// FUSED whole-Radiance kernel: MX-Gemmini QKV projection + Muon-SIMT RoPE, sharing SMEM.
//
//   Phase 1 (MX): C = X . W    (fp8 e4m3, per-32-K e8m0 scales) -> C[M][N] bf16 in SMEM
//                 at SPAD_DEST()*DIM.  mxgemm_single_output_tile() does config + K-loop and
//                 leaves C in the scratchpad WITHOUT moving it out.
//   Phase 2 (SIMT): RoPE on the SMEM-resident C, writing the rotated bf16 result STRAIGHT to
//                 GMEM. The intermediate Q/K projection never round-trips through DRAM -- that
//                 saved DRAM traffic is the whole-Radiance fusion win.
//
// HF Llama rotate_half, cos/sin duplicated across halves (cos[i]==cos[i+HALF]):
//   out[m][i]      = C[m][i]*cos - C[m][i+HALF]*sin
//   out[m][i+HALF] = C[m][i+HALF]*cos + C[m][i]*sin
// Paired form (one thread per (m,i) pair) reads both originals and writes both outputs -> no
// cross-thread read-after-write hazard, no 16-bit SMEM store needed.

static inline uint32_t f2b(float f){ union{float f; uint32_t u;} v; v.f=f; return v.u; }
static inline float    b2f(uint32_t u){ union{float f; uint32_t u;} v; v.u=u; return v.f; }
static inline float    bf16_to_f(uint16_t h){ return b2f((uint32_t)h << 16); }
static inline uint16_t f_to_bf16(float f){ return (uint16_t)(f2b(f) >> 16); }  // truncate (golden matches)

void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t M = MATMUL_M, N = MATMUL_N, HALF = N / 2;
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;

  // --- Phase 1: MX matmul X.W -> C(bf16) in SMEM (no move-out) ---
  mxgemm_single_output_tile<GEMM_CFG>(M, N, MATMUL_K, tid_in_threadblock, threads_per_threadblock);
  mu_barrier(1, nw);

  // --- Phase 2: SIMT RoPE on SMEM C -> GMEM out (fused move-out) ---
  const uint32_t Cbase = GEMM_CFG.SPAD_DEST() * DIM;           // byte addr of C in SMEM (bf16)
  auto* out = reinterpret_cast<uint16_t*>(a->C);               // bf16 GMEM output [M][N]
  const uint32_t pairs = M * HALF;
  #pragma clang loop unroll(disable)
  for (uint32_t p = tid_in_threadblock; p < pairs; p += threads_per_threadblock) {
    const uint32_t m = p / HALF, i = p % HALF;
    const uint32_t o0 = m * N + i, o1 = m * N + i + HALF;
    const float c0 = bf16_to_f(load16_shared(Cbase + o0 * 2));
    const float c1 = bf16_to_f(load16_shared(Cbase + o1 * 2));
    const float cs = a->cosc[o0];     // cos/sin duplicated: cos[i]==cos[i+HALF]
    const float sn = a->sinc[o0];
    out[o0] = f_to_bf16(c0 * cs - c1 * sn);
    out[o1] = f_to_bf16(c1 * cs + c0 * sn);
  }
}
