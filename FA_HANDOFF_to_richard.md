# FA (flash_attention_mx) — port + findings to hand back to Richard

**TL;DR:** your kernel ported cleanly to our tapeout-330 tree and runs on our cyclotron + Verilator
RTL, but the version I pulled is the **serial-isolation WIP (O ≈ 30% rel err, the bank0-KV bug you
flagged)** — not the validated 2.48% one. I built a trace-db verifier so we can both gate it here.

## What I did on garden (tapeout-330)
- Copied `flash_attention_mx` → `radiance-kernels/kernels/flash_attention_mx_yrh/`.
- **Builds clean on tapeout-330** — only had to drop the external `gemmini_host_shim.h` `-include`
  (our `gemm_mxgemmini` builds without it; `include/gemmini.h` + `mxgemmini_mmio.h` resolve from our
  tree). No 329a→330 source drift issues.
- **Runs clean**: cyclotron 185K cyc; Verilator RTL 243K cyc, `$finish`, **no l0d assert** (2 warps).

## New tool: `fa_verify_sqlite.py` (in the kernel dir)
Our cyclotron/Verilator emit a **sqlite trace-db**, not the VCS `.out` `[ISSUE]` text your
`fa_verify_out.py` parses — so that verifier reads 0 stores here. `fa_verify_sqlite.py` reconstructs
O from the trace-db `inst` table (per-lane `rs1_data`=addr, `rs2_data`=word), **filtered to store PCs**
(`--elf`; an add/other instr can hold an in-range value in rs1 and corrupt the reconstruction — that
bit me first). Validated: reconstructs all 8192 O cells, max abs diff 0.19.
```
python3 fa_verify_sqlite.py trace_fa.sqlite --base 0x40040000 --rows 64 --cols 128 \
        --golden golden_O_flash_u16.npy --elf flash_attention_mx.radiance.elf
```

## The finding
- **O rel err = 30.3%** on the fetched `.cpp` — which matches your inline note *"if 34% → bank0-KV
  corrupts the matmul."* So this snapshot is the serial-isolation diagnostic with the open bank-layout
  bug, independent of the async overlap.
- Perf eval (once correct): MX util **6.7%** whole-kernel, SIMT-softmax-bound, engine **overlap 1.0×**
  (async QK‖softmax in serial-isolation). Real headroom via overlap + tile amortization.

## Proposed division of labor
1. **You**: land the bank0-KV correctness fix (get O back to ~2.48%). Point me at the validated commit.
2. **Me**: once it verifies, I'll RTL-gate the async-overlap flip + tile amortization here (I have the
   verifier + the whole-Radiance overlap metric), and/or feed it to the autocomp RTL-in-the-loop loop.
Happy to share `fa_verify_sqlite.py` upstream if useful.
