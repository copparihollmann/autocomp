// SMEM-staged attention (seq=64,d=64). Stage Q (row-major), K (TRANSPOSED+padded so phase-1
// lane accesses over j are stride-1/conflict-free, the matmul-B trick), V (row-major); scores
// live in SMEM. Cross-core mu_barrier(0,nw) between phases (SMEM visibility across both cores).
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;
  const uint32_t d = a->d;
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;
  const uint32_t PAD = 16;
  const uint32_t sp = seq + PAD;            // padded pitch for transposed K

  const uint32_t Q_B = 0;                   // Q[i][k]  : seq*d
  const uint32_t KT_B = Q_B + seq * d * 4;  // KT[k][j] : d*sp  (transposed, padded)
  const uint32_t P_B = KT_B + d * sp * 4;   // P[i][j]  : seq*seq
  const uint32_t V_B = P_B + seq * seq * 4; // V[j][k]  : seq*d

  // ---- stage ----
  const __global float* gQ = a->Q;
  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock)
    store_shared(Q_B + i * 4, 0, __builtin_bit_cast(uint32_t, gQ[i]));
  const __global float* gK = a->K;          // K[j][k] -> KT[k][j]
  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock) {
    uint32_t j = i / d, k = i % d;
    store_shared(KT_B + (k * sp + j) * 4, 0, __builtin_bit_cast(uint32_t, gK[i]));
  }
  const __global float* gV = a->V;
  for (uint32_t i = tid_in_threadblock; i < seq * d; i += threads_per_threadblock)
    store_shared(V_B + i * 4, 0, __builtin_bit_cast(uint32_t, gV[i]));
  mu_barrier(0, nw);

  // ---- phase 1: S = scale * Q K^T  (from SMEM) ----
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    uint32_t i = idx / seq, j = idx % seq;
    uint32_t qb = Q_B + i * d * 4;
    float acc = 0.0f;
    for (uint32_t k = 0; k < d; k++) {
      float q = __builtin_bit_cast(float, load32_shared(qb + k * 4));
      float kk = __builtin_bit_cast(float, load32_shared(KT_B + (k * sp + j) * 4));
      acc += q * kk;
    }
    store_shared(P_B + idx * 4, 0, __builtin_bit_cast(uint32_t, acc * scale));
  }
  mu_barrier(0, nw);

  // ---- phase 2: row softmax (in SMEM) ----
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    uint32_t rb = P_B + row * seq * 4;
    float m = __builtin_bit_cast(float, load32_shared(rb));
    for (uint32_t j = 1; j < seq; j++) {
      float v = __builtin_bit_cast(float, load32_shared(rb + j * 4));
      if (v > m) m = v;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      float e = mu_exp(__builtin_bit_cast(float, load32_shared(rb + j * 4)) - m);
      store_shared(rb + j * 4, 0, __builtin_bit_cast(uint32_t, e));
      sum += e;
    }
    float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      float e = __builtin_bit_cast(float, load32_shared(rb + j * 4));
      store_shared(rb + j * 4, 0, __builtin_bit_cast(uint32_t, e * inv));
    }
  }
  mu_barrier(0, nw);

  // ---- phase 3: O = P V  (from SMEM) ----
  __global float* gO = a->O;
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    uint32_t i = idx / d, k = idx % d;
    uint32_t pb = P_B + i * seq * 4;
    float acc = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      float p = __builtin_bit_cast(float, load32_shared(pb + j * 4));
      float v = __builtin_bit_cast(float, load32_shared(V_B + (j * d + k) * 4));
      acc += p * v;
    }
    gO[idx] = acc;
  }
}
