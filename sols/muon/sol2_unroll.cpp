// Attention, ILP-unrolled (register-legal, 1 output/thread). Same 3-phase structure
// as baseline but each reduction (QK^T dot, softmax sum, PV dot) uses 4 independent
// accumulators + branchless max + pointer walks to break the dependency chains —
// the trick that gave softmax/matmul their ILP wins. No SMEM (staging was slower here).
static inline float fmaxf_(float a, float b) { return a > b ? a : b; }

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
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;

  // Phase 1: S = scale * Q K^T, 4-way unrolled dot product.
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    const uint32_t i = idx / seq, j = idx % seq;
    const __global float* q = a->Q + i * d;
    const __global float* k = a->K + j * d;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t t = 0;
    for (; t + 4 <= d; t += 4) {
      a0 += q[t + 0] * k[t + 0]; a1 += q[t + 1] * k[t + 1];
      a2 += q[t + 2] * k[t + 2]; a3 += q[t + 3] * k[t + 3];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; t < d; t++) acc += q[t] * k[t];
    a->scratch[idx] = acc * scale;
  }
  mu_barrier(1, num_warps);

  // Phase 2: softmax per row (independent max/sum accumulators + branchless max).
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    __global float* s = a->scratch + row * seq;
    float m0 = -3.0e38f, m1 = -3.0e38f, m2 = -3.0e38f, m3 = -3.0e38f;
    uint32_t j = 0;
    for (; j + 4 <= seq; j += 4) {
      m0 = fmaxf_(m0, s[j + 0]); m1 = fmaxf_(m1, s[j + 1]);
      m2 = fmaxf_(m2, s[j + 2]); m3 = fmaxf_(m3, s[j + 3]);
    }
    float m = fmaxf_(fmaxf_(m0, m1), fmaxf_(m2, m3));
    for (; j < seq; j++) m = fmaxf_(m, s[j]);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (j = 0; j + 4 <= seq; j += 4) {
      const float e0 = mu_exp(s[j+0]-m), e1 = mu_exp(s[j+1]-m), e2 = mu_exp(s[j+2]-m), e3 = mu_exp(s[j+3]-m);
      s[j+0]=e0; s[j+1]=e1; s[j+2]=e2; s[j+3]=e3;
      c0+=e0; c1+=e1; c2+=e2; c3+=e3;
    }
    float sum = (c0 + c1) + (c2 + c3);
    for (; j < seq; j++) { const float e = mu_exp(s[j]-m); s[j]=e; sum+=e; }
    const float inv = 1.0f / sum;
    for (j = 0; j + 4 <= seq; j += 4) { s[j+0]*=inv; s[j+1]*=inv; s[j+2]*=inv; s[j+3]*=inv; }
    for (; j < seq; j++) s[j] *= inv;
  }
  mu_barrier(1, num_warps);

  // Phase 3: O = P V, 4-way unrolled over j (the seq reduction).
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    const uint32_t i = idx / d, kk = idx % d;
    const __global float* p = a->scratch + i * seq;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    uint32_t j = 0;
    for (; j + 4 <= seq; j += 4) {
      o0 += p[j+0] * a->V[(j+0) * d + kk];
      o1 += p[j+1] * a->V[(j+1) * d + kk];
      o2 += p[j+2] * a->V[(j+2) * d + kk];
      o3 += p[j+3] * a->V[(j+3) * d + kk];
    }
    float acc = (o0 + o1) + (o2 + o3);
    for (; j < seq; j++) acc += p[j] * a->V[j * d + kk];
    a->O[idx] = acc;
  }
}
