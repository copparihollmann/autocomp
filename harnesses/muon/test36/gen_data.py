#!/usr/bin/env python3
"""Decode P.V (GEMV): out[d] = sum_s probs[s]*V[s,d], d in [0,D). S=128, D=64."""
import numpy as np
S, D = 128, 64
rng = np.random.default_rng(36)
p = rng.random(S, np.float32); p = (p / p.sum()).astype(np.float32)   # a probability row
V = rng.standard_normal((S, D), np.float32)
gold = (p.astype(np.float64) @ V.astype(np.float64)).astype(np.float32)  # [D]
def emit(f,n,a):
    fl=a.flatten(); f.write(f"__global float {n}[{fl.size}] = {{\n"+",\n".join(f"{x:.9e}f" for x in fl)+"\n};\n")
with open("data","w") as f:
    f.write(f"#define VERIFY_COUNT {D}\nstatic const uint32_t SS={S};\nstatic const uint32_t DD={D};\n")
    emit(f,"p_raw",p); emit(f,"v_raw",V); emit(f,"out_raw",np.zeros(D,np.float32)); emit(f,"gold_raw",gold)
    f.write("__global float tls_guard[1024]={0};\n")
print(f"wrote data: decode P.V GEMV S={S} D={D}")
