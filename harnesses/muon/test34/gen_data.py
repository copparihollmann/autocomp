#!/usr/bin/env python3
"""Decode Q.K^T (GEMV): scores[s] = (sum_d K[s,d]*q[d]) / sqrt(D), s in [0,S).
Single decode query attending to S keys of head_dim D. S=128, D=64."""
import numpy as np
S, D = 128, 64
rng = np.random.default_rng(34)
q = rng.standard_normal(D, np.float32)
K = rng.standard_normal((S, D), np.float32)
scale = np.float32(1.0/np.sqrt(D))
gold = ((K.astype(np.float64) @ q.astype(np.float64)) * scale).astype(np.float32)
def emit(f,n,a):
    fl=a.flatten(); f.write(f"__global float {n}[{fl.size}] = {{\n"+",\n".join(f"{x:.9e}f" for x in fl)+"\n};\n")
with open("test34/data" if False else "data","w") as f:
    f.write(f"#define VERIFY_COUNT {S}\nstatic const uint32_t SS={S};\nstatic const uint32_t DD={D};\n")
    f.write(f"static const float SCALE={scale:.9e}f;\n")
    emit(f,"q_raw",q); emit(f,"k_raw",K); emit(f,"out_raw",np.zeros(S,np.float32)); emit(f,"gold_raw",gold)
    f.write("__global float tls_guard[1024]={0};\n")
print(f"wrote data: decode Q.KT GEMV S={S} D={D}")
