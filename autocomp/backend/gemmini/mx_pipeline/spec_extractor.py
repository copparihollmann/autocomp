"""Extract optimizable op specs (matmul / batch-matmul / conv) from model2MLIR
output, so we can drive MX-Gemmini problem generation from real PyTorch models.

model2MLIR lowers PyTorch -> MLIR (linalg-on-tensors) and tags every op with
provenance attrs (prov.op, prov.family, prov.region_id, prov.aten, prov.orig_dtype).
This reads a captured .mlir and returns a deduped list of the contraction/attention
ops with their tensor shapes, which the problem generators turn into autocomp
MX-Gemmini problems (harness + golden data + baseline).

Usage:
    python -m autocomp.backend.gemmini.mx_pipeline.spec_extractor <file.mlir> [--family matmul]
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpSpec:
    family: str            # "matmul" | "batch_matmul" | "conv2d"
    dims: tuple            # matmul: (M, K, N); batch_matmul: (B, M, K, N); conv: raw shapes
    dtype: str             # element type seen in the MLIR tensor (e.g. "f32", "f8E4M3FN")
    aten: str = ""
    raw_shapes: tuple = field(default=(), compare=False)

    def key(self):
        return (self.family, self.dims, self.dtype)


_TENSOR = r"tensor<([0-9x]+)x([a-zA-Z0-9_]+)>"
# linalg.matmul { ...attrs... } ins(%a, %b : tensor<MxK x T>, tensor<KxN x T>)
_MATMUL_RE = re.compile(
    r"linalg\.matmul\s*\{([^}]*)\}\s*ins\([^:]*:\s*" + _TENSOR + r"\s*,\s*" + _TENSOR,
    re.DOTALL,
)
# linalg.batch_matmul (or generic tagged batch_matmul) ins with 3-D tensors
_BMM_RE = re.compile(
    r"linalg\.batch_matmul\s*\{([^}]*)\}\s*ins\([^:]*:\s*" + _TENSOR + r"\s*,\s*" + _TENSOR,
    re.DOTALL,
)


def _parse_dims(s):
    return tuple(int(x) for x in s.split("x"))


def extract_specs(mlir_text: str) -> list[OpSpec]:
    specs = []
    for m in _MATMUL_RE.finditer(mlir_text):
        attrs, ad, at, bd, bt = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        a, b = _parse_dims(ad), _parse_dims(bd)
        if len(a) == 2 and len(b) == 2 and a[1] == b[0]:
            M, K, N = a[0], a[1], b[1]
            aten = (re.search(r'prov\.aten = "([^"]*)"', attrs) or [None, ""])[1] \
                if re.search(r'prov\.aten = "([^"]*)"', attrs) else ""
            specs.append(OpSpec("matmul", (M, K, N), at, aten, (ad, bd)))
    for m in _BMM_RE.finditer(mlir_text):
        ad, at, bd = m.group(2), m.group(3), m.group(4)
        a, b = _parse_dims(ad), _parse_dims(m.group(4))
        if len(a) == 3 and len(b) == 3:
            specs.append(OpSpec("batch_matmul", (a[0], a[1], a[2], b[2]), at, "", (ad, bd)))
    # conv2d ops are tagged on linalg.generic/conv; count via provenance for now
    return specs


def summarize(mlir_text: str):
    specs = extract_specs(mlir_text)
    fam_counts = Counter(s.family for s in specs)
    # provenance-level census (catches ops we don't yet shape-parse, e.g. softmax/conv)
    prov = Counter(re.findall(r'prov\.op = "([a-z_0-9]+)"', mlir_text))
    uniq = {}
    for s in specs:
        uniq.setdefault(s.key(), 0)
        uniq[s.key()] += 1
    return specs, fam_counts, prov, uniq


if __name__ == "__main__":
    path = sys.argv[1]
    text = open(path).read()
    specs, fam_counts, prov, uniq = summarize(text)
    print(f"file: {path}")
    print(f"shape-parsed ops: {len(specs)}  by family: {dict(fam_counts)}")
    print(f"provenance census (prov.op): {dict(prov)}")
    print(f"\ndistinct kernels worth optimizing ({len(uniq)}):")
    for (fam, dims, dtype), n in sorted(uniq.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  x{n:<4} {fam:<13} dims={dims} dtype={dtype}")
