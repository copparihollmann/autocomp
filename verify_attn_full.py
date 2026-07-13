import re, numpy as np, pathlib
from autocomp.common import HARNESSES_DIR
from autocomp.backend.gemmini.mx_pipeline import attention_gen, flash_attention_gen
INC = pathlib.Path("/scratch/agustin/projects/chipyard-mx/generators/gemmini/software/gemmini-rocc-tests/include")

def grid(txt,name,dt):
    i=txt.index(name);b=txt.index("{",i);depth=0;j=b
    while j<len(txt):
        c=txt[j]
        if c=="{":depth+=1
        elif c=="}":
            depth-=1
            if depth==0:break
        j+=1
    return np.array([[int(x.strip().rstrip('uUlL'),0) for x in r.split(',') if x.strip()] for r in re.findall(r"\{([^{}]*)\}",txt[b+1:j])],dtype=object)
def arr1(txt,name):
    i=txt.index(name);b=txt.index("{",i);e=txt.index("}",b)
    return [int(x.strip().rstrip('uUlL'),0) for x in txt[b+1:e].split(',') if x.strip()]
def bf16f(u): return float(np.uint32(int(u)<<16).view(np.float32))
def fp8d(b):
    b=int(b);s=(b>>7)&1;e=(b>>3)&0xF;m=b&7;v=(m/8.)*2**-6 if e==0 else (1+m/8.)*2**(e-7);return (-1 if s else 1)*v
def fp6d(c):
    c=int(c);s=(c>>5)&1;e=(c>>2)&7;m=c&3;v=(m/4.)*2**-2 if e==0 else (1+m/4.)*2**(e-3);return (-1 if s else 1)*v
def fp4d(c):
    c=int(c);s=(c>>3)&1;e=(c>>1)&3;m=c&1;v=(m*.5) if e==0 else (1+m*.5)*2**(e-1);return (-1 if s else 1)*v

def scores(qk):  # bf16 score matrix [S][S]
    t=open(INC/qk).read(); nm="QK_C_out_bf16" if "QK_C_out_bf16" in t else "QKCB"
    g=grid(t,nm,object); return np.vectorize(bf16f)(g.astype(np.int64))
def Vdeq(vh,fmt):
    t=open(INC/vh).read(); Vs=grid(t,"V_scales_col",object).astype(np.int64)  # [GK][N]
    if fmt=="fp8":
        Vin=grid(t,"V_in",object).astype(np.int64)  # [K][N] e4m3
        K,N=Vin.shape
        return np.array([[fp8d(Vin[k][n])*2.0**(int(Vs[k//32][n])-127) for n in range(N)] for k in range(K)])
    dec=fp6d if fmt=="fp6" else fp4d
    lut=arr1(t,"V_lut") if fmt=="fp6" else list(range(16))   # fp4: nibble IS the e2m1 code
    Vin=grid(t,"V_in",object).astype(np.int64)  # [K][N/2] nibble-packed
    K=Vin.shape[0]; N=Vs.shape[1]
    out=np.zeros((K,N))
    for k in range(K):
        for n in range(N):
            byte=int(Vin[k][n//2]); nib=(byte>>4)&0xF if (n&1) else byte&0xF
            out[k][n]=dec(lut[nib])*2.0**(int(Vs[k//32][n])-127)
    return out
def true_O(qk,vh,fmt):
    sc=scores(qk); V=Vdeq(vh,fmt)
    P=np.exp(sc-sc.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
    return P@V
def kern_O_attn(prob,S,D):
    t=(HARNESSES_DIR/prob/"test0.c").read_text(); g=grid(t,"static const out_t gold",object)
    O=np.zeros((S,D))
    for i in range(S):
        for d in range(D):
            w=int(g[i][d//4]); O[i][d]=bf16f((w>>((d%4)*16))&0xFFFF)
    return O
def kern_O_flash(prob,S,D):
    vals=open(HARNESSES_DIR/prob/"gold0.txt").read().split()
    O=np.zeros((S,D))
    for i in range(S):
        for d in range(D):
            v=int(vals[i*D+d],16); v=v-(1<<32) if v>=(1<<31) else v; O[i][d]=v/65536.0
    return O

S,D=128,64
CASES=[
 ("attn","gemmini-mx-attention","fp8","mxgen_att_qk_128x64.h","mxgen_att_v_128x64.h"),
 ("attn","gemmini-mx-attn-fp6","fp6","mxgen_fp6_attqk_128x64.h","mxgen_fp6_attv_128x64.h"),
 ("attn","gemmini-mx-attn-fp4","fp4","mxgen_fp4_attqk_128x64.h","mxgen_fp4_attv_128x64.h"),
 ("flash","gemmini-mx-flash-attn","fp8","mxgen_fla_qk_128x64.h","mxgen_fla_v_128x64.h"),
 ("flash","gemmini-mx-flash-fp6","fp6","mxgen_fp6_flaqk_128x64.h","mxgen_fp6_flav_128x64.h"),
 ("flash","gemmini-mx-flash-fp4","fp4","mxgen_fp4_flaqk_128x64.h","mxgen_fp4_flav_128x64.h"),
]
from autocomp.backend.gemmini.mx_pipeline import attention_gen as _A, flash_attention_gen as _F
_FMTMAP={"fp8":"fp8:e4m3","fp6":"fp6:e3m2","fp4":"fp4:e2m1"}
for kind,prob,fmt,qk,vh in CASES:
  try:
    if kind=="attn":
        _A.gen_attention_problem(128,64,prob_type=prob,prob_id=0,fmt=_FMTMAP[fmt]) if fmt!="fp8" else _A.gen_attention_problem(128,64,prob_id=0)
    else:
        _F.gen_flash_attention_problem(128,64,prob_type=prob,prob_id=0,fmt=_FMTMAP[fmt]) if fmt!="fp8" else _F.gen_flash_attention_problem(128,64,prob_id=0)
    Ot=true_O(qk,vh,fmt)
    Ok=kern_O_attn(prob,S,D) if kind=="attn" else kern_O_flash(prob,S,D)
    w5=np.mean(np.abs(Ok-Ot)<=0.05*np.abs(Ot)+1e-2)*100
    w20=np.mean(np.abs(Ok-Ot)<=0.20*np.abs(Ot)+2e-2)*100
    print(f"{prob:26s} {fmt}: ACTUAL-C vs true fp32  within-5%={w5:5.1f}%  within-20%={w20:5.1f}%  mean|Ot|={np.abs(Ot).mean():.3f} corr={np.corrcoef(Ok.flatten(),Ot.flatten())[0,1]:.3f}")
  except Exception as ex:
    print(f"{prob:26s} {fmt}: ERROR {type(ex).__name__}: {ex}")
