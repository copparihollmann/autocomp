// fp4 MxGEMM: full mxgemm() writing bf16 C to GMEM.
void kernel_body(void* raw_arg, uint32_t tid, uint32_t tpb, uint32_t tbid) {
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  mxgemm<GEMM_CFG>(a->M, a->N, a->K, a->C, tid, tpb, tbid);
}
