import numpy as np
rng=np.random.default_rng(0); R,Cc=64,67
def mu_exp(x):
    x=np.float32(x); log2e=np.float32(1.4426950408889634); ln2=np.float32(0.6931471805599453)
    t=x*log2e; k=np.where(t>=0,t+np.float32(0.5),t-np.float32(0.5)).astype(np.int32)
    r=x-k.astype(np.float32)*ln2
    p=np.float32(1.0)+r*(np.float32(1.0)+r*(np.float32(0.5)+r*(np.float32(0.16666667)+r*(np.float32(0.041666668)+r*np.float32(0.008333334)))))
    return (p*np.exp2(k.astype(np.float32))).astype(np.float32)
X=((rng.random((R,Cc),dtype=np.float32)-0.5)*0.6).astype(np.float32)  # [-0.3,0.3] -> k=0
gold=mu_exp(X)
def emit(f,n,a):
    fl=a.flatten(); f.write(f"__global float {n}[{fl.size}] = {{\n"); f.write(",\n".join(f"{x:.9e}f" for x in fl)); f.write("\n};\n")
with open("data","w") as f:
    f.write(f"#define VERIFY_COUNT {R*Cc}\n"); f.write(f"static const uint32_t ROWS={R}, COLS={Cc};\n")
    emit(f,"in_raw",X); emit(f,"gold_raw",gold); emit(f,"out_raw",np.zeros((R,Cc),np.float32)); f.write("__global float tls_guard[1024]={0};\n")
print("max|X|=",abs(X).max(),"-> all k=0")
