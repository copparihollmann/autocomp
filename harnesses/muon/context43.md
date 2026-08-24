TARGET: the fused fp4 MX-Gemmini TinyLlama FFN block on Radiance (Muon SIMT + MX-Gemmini).

    out = ResAdd( down( SwiGLU( gate(RMSNorm(x)), up(RMSNorm(x)) ) ), x )

This is a MULTI-BODY fused kernel, NOT a single kernel_body. The code you optimize is the region
between `// SUBSTITUTE HERE` and `// SUBSTITUTE END`, which contains:
  - copy_C_to        : SMEM->GMEM move-out helper
  - quantize_fp4     : runtime bf16->fp4(e2m1)+e8m0 block-scale quantizer (bit-exact; do NOT change math)
  - quant_xn_body / quant_h_body : thin wrappers calling quantize_fp4
  - rmsnorm_body     : SIMT RMSNorm prologue (X -> Xn bf16)
  - gate_body / up_body : fp4 MX matmul (Xn_fp4 @ Wg/Wu) + move-out
  - swiglu_body      : h = silu(gate)*up (bf16), 32-bit vectorized
  - down_body        : 8 fp4 MX N-tiles (h_fp4 @ Wd) + residual add
  - run_ffn()        : the ORCHESTRATION (mu_schedule / mu_barrier / mu_fence / drain sequence)

FIXED (supplied by the harness; do NOT redefine): all includes, the `data` header, the fp/fp4
helper functions (f2b/b2f/bf16_to_f/f32_to_bf16_rne/fp4 encoders/block_scale_code/my_rsqrt/mu_exp/
silu/strip8), the GemmConfig constexprs GU_CFG/DN_CFG, and main()+verify_body (the tohost verdict).

YOUR GOAL: reduce total cycles while keeping the output bit-exact (verify_body compares out_raw vs
gold_raw; tohost=0 required). The fp4 MX matmul mesh throughput is fixed hardware — do NOT try to
change the matmul math. The reachable levers are in the SIMT bodies and the orchestration:
  - loop structure / vectorization (32-bit word loads already used in swiglu; apply elsewhere).
  - the thread->work mapping (grid-stride; derive from threads_per_threadblock, never a literal warp count).
  - removing redundant passes / fusing SIMT loops (e.g. rmsnorm two-pass; quantize amax+encode).
  - fence/barrier/drain placement in run_ffn (fewer synchronizations, but keep correctness — cross-core
    visibility needs mu_barrier(0, MU_NUM_CORES) after each mu_schedule, and Gemmini DMA must not read a
    GMEM buffer before the producing SIMT stores are visible).
  - warp count per phase (MX_NUM_WARPS=2; the physical regfile budget is a GLOBAL 256 (warp,rd)
    first-writes across all 8 warp slots — heavy bodies must stay well under it).

Keep the function names and signatures unchanged (main() calls run_ffn(); the bodies are dispatched by
mu_schedule). Do NOT add prints. Emit ONE fenced code block containing the full SUBSTITUTE region.

NOTE ON FITNESS: this fused megakernel deadlocks under the cyclotron TIMING model, so fitness is the
cyclotron FUNCTIONAL dynamic-issue cycle count (a coarse instruction-count pre-filter). Real hardware
cycles are decided by an RTL (Verilator) gate afterward; prefer changes that genuinely cut dynamic
work (fewer instructions / memory ops), not ones that only look cheaper in an issue model.
