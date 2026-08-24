# Runtime-DRAM Weight-Blob Format (TinyLlama e2e on Radiance/cyclotron)

**Purpose.** Move large read-only model tensors OUT of the compiled kernel ELF and INTO
simulated DRAM, loaded at runtime by the harness. Keeps the ELF small and lets model dims
(FFN width, layers, vocab) scale past what a compiled static C-array could ever build.
This is the format the real-weights pipeline (agent e2e-2) should emit.

## Transport
- A single **flat binary** (`weights.bin`), loaded verbatim into cyclotron global memory
  (DRAM) at a fixed base address `WEIGHTS_BASE`.
- Loaded via the cyclotron env var (implemented in `sim/top.rs::preload_weights_if_requested`):

  ```
  CYCLOTRON_WEIGHTS=<hex_base>:<path_to_weights.bin>[,<hex_base>:<path>...]
  ```

  e.g. `CYCLOTRON_WEIGHTS=0x80000000:/abs/path/weights.bin`. Multiple comma-separated
  blobs are allowed. The bytes are copied straight into gmem starting at `hex_base`; this
  is a pure harness/loadmem op — **no RTL/DUT change**.

## Address map (why 0x80000000)
- cyclotron models a flat 4 GiB memory (`FlatMemory`, `0..0xFFFFFFFF`).
- Kernel ELF loads at `0x10000000`; `.args` at `0x7FFF0000`; io_cout at `0xFF080000`.
- `[0x80000000, 0xFF080000)` is a free ~1.9 GiB window — put the blob at **`0x80000000`**.
- On rv32 the base fits a 32-bit pointer. The MX-Gemmini co-model's `gpu_dram()` treats any
  address `< 0x1_0000_0000` as a raw gmem byte offset (the kernel's mxgemm driver adds
  `GPU_DRAM_OFFSET=0x1_0000_0000` before the RoCC command and `gpu_dram()` subtracts it, so
  both direct SIMT loads and accelerator DMAs resolve to `gmem[base+offset]`).
- **Kernel address of a tensor = `WEIGHTS_BASE + byte_offset`** (from the manifest).

## Blob layout
- Tensors are concatenated **C row-major (contiguous)**, each padded to a **64-byte
  (cache-line) aligned** start offset. Natural dtype per tensor (`uint8`/`uint16`).
- The exact tensor set, order, dtype, dims, offset and size are written to a companion
  **`weights_manifest.txt`** (tab-separated), header lines prefixed `#`:

  ```
  # base=0x80000000  total_bytes=<N>  align=64
  # name  ctype  dims  byte_offset  nbytes  (kernel addr = base + byte_offset)
  emb_table       uint16_t  [AT_V][AT_H]                       0        2097152
  Wq_all          uint8_t   [NLAYERS][AT_H][AT_D/2]            ...      ...
  ...
  ```

- fp4 weights are nibble-packed (2 values/byte), MX block-scales are `uint8` e8m0 codes —
  **identical byte layout to the previous static arrays** (only the storage moved). e2e-2's
  real-weight pipeline must reproduce the same per-tensor tiling/packing that
  `gen_data.py` produces (see `tile_w`/`tile_sc`/`quantize_rowmajor_to_fp4`) so offsets line
  up 1:1 with the kernel's indexing.

## Kernel side (zero kernel.cpp edits)
`gen_data.py` emits, into the tiny `data` header, one **typed pointer macro** per tensor so
`kernel.cpp`'s `name[i][j]...` indexing is byte-identical to the old static array:

```cpp
#define WEIGHTS_BASE 0x80000000u
#define Wq_all (reinterpret_cast<const uint8_t(*)[AT_H][AT_D/2]>((uintptr_t)(WEIGHTS_BASE + 2129920u)))
#define emb_table (reinterpret_cast<const uint16_t(*)[AT_H]>((uintptr_t)(WEIGHTS_BASE + 0u)))
#define gamma_final (reinterpret_cast<const uint16_t*>((uintptr_t)(WEIGHTS_BASE + 9928704u)))
```

Rule: drop the outermost array dimension → pointer-to-array-of-the-rest (1-D → `const T*`).
These are compile-time address constants (no static initializer / no `.init_array`).

## What stays in the ELF (small / writable)
`token_ids`, `cos_bits`/`sin_bits` (RoPE), `gold_logits` (regression oracle), and all
`__global` activation scratch. Everything large & read-only goes in the blob.

## Emitter API (in `gen_data.py`, reuse for real weights)
`class WeightBlob`: `.add(ctype, name, dims, arr)` appends a tensor (64-B aligned) and records
a manifest entry; `.emit_macros(fp)` writes `WEIGHTS_BASE` + the pointer macros; `.write(bin,
manifest)` writes `weights.bin` + `weights_manifest.txt`. e2e-2 can populate the same 20-tensor
set from real TinyLlama weights (MX fp4 + e8m0 scales, same tiling) and get a drop-in blob.
