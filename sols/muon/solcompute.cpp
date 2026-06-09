// Compute probe kernel. CHAIN=1: serial FMA dependency chain (latency-bound).
// CHAIN=0: 8 independent FMA accumulators (ILP/throughput-bound). Result -> sink[tid] (no DCE).
static inline void kernel_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw); (void)tpb;
  // c,d derived from tid (runtime) so the FP recurrence can't be constant-folded/closed-formed.
  const float c = 1.0000001f + (float)(tid & 7) * 1e-9f;
  const float d = 0.0000001f + (float)(tid & 3) * 1e-9f;
  // asm barrier each iteration forces acc to be materialized -> prevents the compiler from
  // closed-forming the FP recurrence (else the loop is eliminated). zfinx: float lives in a GPR.
#if CHAIN
  float acc = (float)tid + 1.0f;
  for (uint32_t i = 0; i < ITERS; i++) { acc = acc * c + d; asm volatile("" : "+r"(acc)); } // serial
  a->out[tid & 255] = acc;
#else
  float a0=(float)tid+1, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
  for (uint32_t i = 0; i < ITERS; i++) { // 8 independent chains -> ILP
    a0=a0*c+d; a1=a1*c+d; a2=a2*c+d; a3=a3*c+d; a4=a4*c+d; a5=a5*c+d; a6=a6*c+d; a7=a7*c+d;
    asm volatile("" : "+r"(a0),"+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),"+r"(a5),"+r"(a6),"+r"(a7));
  }
  a->out[tid & 255] = a0+a1+a2+a3+a4+a5+a6+a7;
#endif
}
