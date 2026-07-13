#!/usr/bin/env python3
"""Cross-check OUR mx_golden against the UPSTREAM mxgemmini golden model.

Why: the mxgemmini bump (-> 6fc8ec7) ships "fully-matched" fp8/fp6/fp4 golden models. Our
cyclotron MX co-model is validated bit-exact against `scripts/muon/mx_golden` (itself
validated vs real spike). If the upstream model and mx_golden agree bit-for-bit on the same
inputs, then the co-model is transitively validated against the NEW oracle too -- which is
the whole point of revalidating after the bump. If they disagree, one of the two models is
wrong about the hardware and we must not proceed.

The upstream model (lib/mxgemmini/fp8_matmul_model.py::tiled_matmul_hwlike) has the same
structure as mx_golden: 16x16x16 tiles -> 16-deep accumulate with a per-(k % 16) precision
schedule -> e8m0 group scale -> bf16 accumulate. We feed it OUR acc_e/acc_m/prod schedule so
both models describe the same silicon.

Usage: xcheck_upstream_golden.py <harness_dir>   (e.g. harnesses/muon/test20)
"""
import pathlib
import sys

import numpy as np
import torch

MXG = pathlib.Path("/scratch/agustin/projects/radiance-kernels/lib/mxgemmini")
sys.path.insert(0, str(MXG))
import fp8_matmul_model as up  # noqa: E402

# The hardware schedule our mx_golden / co-model encode (spike libgemmini acc_e/acc_m).
ACC_E = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
ACC_M = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7]
PROD_E, PROD_M = 4, 3


def fp8_e4m3_decode(code) -> float:
    code = int(code)  # numpy uint8 would underflow on (e - 7)
    s = -1.0 if code & 0x80 else 1.0
    e = (code >> 3) & 0xF
    m = code & 0x7
    if e == 0:
        return s * (m / 8.0) * 2.0**-6
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def main():
    h = pathlib.Path(sys.argv[1])
    g = h / "_mxgen"
    # shapes come from the generated `data` header
    txt = (h / "data").read_text()
    dims = {k: int(txt.split(f"#define MATMUL_{k} ")[1].split()[0]) for k in ("M", "N", "K")}
    M, N, K = dims["M"], dims["N"], dims["K"]
    GK = K // 32

    A = np.frombuffer((g / "A.bin").read_bytes(), dtype=np.uint8).reshape(M, K)
    B = np.frombuffer((g / "B.bin").read_bytes(), dtype=np.uint8).reshape(K, N)
    SA = np.frombuffer((g / "SA.bin").read_bytes(), dtype=np.uint8).reshape(GK, M)
    SB = np.frombuffer((g / "SB.bin").read_bytes(), dtype=np.uint8).reshape(GK, N)
    gold = np.frombuffer((g / "C.bin").read_bytes(), dtype="<u2").reshape(M, N)  # our mx_golden

    dec = np.vectorize(fp8_e4m3_decode)
    A_f = torch.tensor(dec(A), dtype=torch.float32)
    B_f = torch.tensor(dec(B), dtype=torch.float32)
    # upstream wants A_scales_row [M, Gk] and B_scales_col [Gk, N], as float e8m0 values
    e8 = lambda c: 2.0 ** (c.astype(np.int32) - 127)
    A_scales_row = torch.tensor(e8(SA).T.copy(), dtype=torch.float32)  # [M, Gk]
    B_scales_col = torch.tensor(e8(SB), dtype=torch.float32)           # [Gk, N]

    C_up = up.tiled_matmul_hwlike(
        A_f, B_f, A_scales_row, B_scales_col, verbose=False,
        prod_precision_list=[(PROD_E, PROD_M)] * 16,
        acc_precision_list=list(zip(ACC_E, ACC_M)),
    )
    # upstream returns float32 holding bf16-valued numbers -> encode to bf16 bits (RNE)
    up_bits = (torch.tensor(C_up).view(torch.int32).numpy().astype(np.uint32) >> 16).astype(np.uint16)

    eq = int((up_bits == gold).sum())
    tot = M * N
    print(f"{h.name}: M={M} N={N} K={K}")
    print(f"  upstream golden vs OUR mx_golden: {eq}/{tot} bit-exact"
          + ("  ✓ MODELS AGREE" if eq == tot else f"  ✗ {tot-eq} DIFFER"))
    if eq != tot:
        idx = np.argwhere(up_bits != gold)[:5]
        for m, n in idx:
            print(f"    [{m},{n}] upstream={up_bits[m,n]:#06x} mx_golden={gold[m,n]:#06x}")
        sys.exit(1)


if __name__ == "__main__":
    main()
