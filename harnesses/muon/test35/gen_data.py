#!/usr/bin/env python3
"""GEMV softmax: probs[s] = softmax(scores)[s] over S=128 (one decode score row)."""
import numpy as np
S = 128
rng = np.random.default_rng(35)
X = rng.standard_normal(S, np.float32)
def mu_exp(x):
    x = np.float32(x)
    log2e = np.float32(1.4426950408889634)
    ln2 = np.float32(0.6931471805599453)
    t = x * log2e
    k = np.where(t >= 0, t + np.float32(0.5), t - np.float32(0.5)).astype(np.int32)
    r = x - k.astype(np.float32) * ln2
    p = np.float32(1.0) + r * (np.float32(1.0) + r * (np.float32(0.5) + r * (
        np.float32(0.16666667) + r * (np.float32(0.041666668) + r * np.float32(0.008333334)))))
    return (p * np.exp2(k.astype(np.float32))).astype(np.float32)
m = X.max(); e = mu_exp(X - m); gold = (e / e.sum()).astype(np.float32)
def emit(f,n,a):
    fl=a.flatten(); f.write(f"__global float {n}[{fl.size}] = {{\n"+",\n".join(f"{x:.9e}f" for x in fl)+"\n};\n")
with open("data","w") as f:
    f.write(f"#define VERIFY_COUNT {S}\nstatic const uint32_t SS={S};\n")
    emit(f,"x_raw",X); emit(f,"out_raw",np.zeros(S,np.float32)); emit(f,"gold_raw",gold)
    f.write("__global float tls_guard[1024]={0};\n")
print(f"wrote data: GEMV softmax S={S}")
