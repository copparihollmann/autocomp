#!/usr/bin/env python3
"""Extract per-op kernel specs from model2MLIR captures for the radiance Muon target.

Parses linalg ops + prov.* metadata from <model>.mlir and produces muon_specs.json with:
  - the real shapes seen for matmul / conv2d / attention(softmax+batch_matmul) / activations
  - a Muon-scaled tile per op (single threadblock: 2 cores x 8 warps x 16 lanes = 256 threads)

Usage: python extract_specs.py [--out muon_specs.json] [model.mlir ...]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

M2M = Path("/scratch/agustin/projects/model2MLIR/workloads")
DEFAULT_MODELS = [M2M / m / f"{m}.mlir" for m in ("openvla", "rdt", "smolvla")]

TENSOR_RE = re.compile(r"tensor<([0-9]+(?:x[0-9]+)*)x?f32>")
INS_RE = re.compile(r"ins\(([^)]*)\)")


def shapes_in(line: str):
    return ["x".join(m.split("x")) for m in TENSOR_RE.findall(line)]


def ins_shapes(line: str):
    m = INS_RE.search(line)
    return TENSOR_RE.findall(m.group(1)) if m else []


def parse(path: Path):
    """Return Counters: matmul (A,B), conv (in, w), softmax t, gelu/silu t."""
    out = {"matmul": Counter(), "conv2d": Counter(),
           "softmax": Counter(), "gelu": Counter(), "silu": Counter()}
    for line in path.read_text().splitlines():
        if "linalg.matmul" in line or "linalg.batch_matmul" in line:
            ins = ins_shapes(line)
            if len(ins) == 2:
                out["matmul"][f"{ins[0]} @ {ins[1]}"] += 1
        elif 'prov.op = "conv2d"' in line and "linalg.generic" in line:
            ins = ins_shapes(line)
            if len(ins) == 2:  # (activation, weight OIHW)
                out["conv2d"][f"{ins[0]} * {ins[1]}"] += 1
        elif 'prov.op = "softmax"' in line:
            ts = TENSOR_RE.findall(line)
            if ts:
                out["softmax"][ts[-1]] += 1
        elif 'prov.op = "gelu"' in line:
            ts = TENSOR_RE.findall(line)
            if ts:
                out["gelu"][ts[-1]] += 1
        elif 'prov.op = "sigmoid"' in line:  # silu lowers to sigmoid+mul
            ts = TENSOR_RE.findall(line)
            if ts:
                out["silu"][ts[-1]] += 1
    return out


def main():
    argv = sys.argv[1:]
    out_path = Path("muon_specs.json")
    if "--out" in argv:
        i = argv.index("--out")
        out_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    models = [Path(a) for a in argv] or DEFAULT_MODELS

    spec = {"models": {}, "muon_tiles": {}}
    for m in models:
        if not m.exists():
            print(f"skip missing {m}")
            continue
        counts = parse(m)
        spec["models"][m.stem] = {
            op: counter.most_common(8) for op, counter in counts.items()
        }

    # Muon-scaled single-threadblock tiles, derived from the real shapes above.
    # 2 cores x 8 warps x 16 lanes = 256 threads; 128 KiB smem/cluster; fp32.
    spec["muon_tiles"] = {
        "matmul": {  # from rdt MLP addmm 1x256 @ 256x2048
            "M": 64, "N": 64, "K": 64, "dtype": "fp32",
            "source": "rdt addmm 1x256@256x2048; tiled to one 64^3 block"},
        "conv2d": {  # SigLIP/DINO 16x16 stride-16 patch embed
            "H": 64, "W": 64, "C": 3, "K": 16, "stride": 16, "OC": 16, "dtype": "fp32",
            "source": "smolvla 1x3x512x512 * 768x3x16x16; one 64x64 tile, 16/768 out-ch"},
        "attention": {  # rdt single head
            "seqlen": 64, "head_dim": 64, "dtype": "fp32",
            "source": "rdt 32 heads x 67 seq x 64 dim; one head, 64 seq tile"},
        "flashattention": {"seqlen": 128, "head_dim": 64, "tile": 32, "dtype": "fp32",
                            "source": "same op, longer sequence to make tiling worthwhile"},
        "swiglu": {"m": 64, "n": 512, "dtype": "fp32",
                    "source": "rdt MLP silu(x)*y, 2048-dim FFN tiled to 64x512"},
        "softmax": {"rows": 64, "cols": 67, "dtype": "fp32",
                     "source": "rdt softmax 1x32x67 attention logits"},
    }

    out_path.write_text(json.dumps(spec, indent=2))
    print(f"wrote {out_path}")
    for model, ops in spec["models"].items():
        print(f"\n== {model} ==")
        for op, items in ops.items():
            if items:
                print(f"  {op}: {items[:3]}")


if __name__ == "__main__":
    main()
