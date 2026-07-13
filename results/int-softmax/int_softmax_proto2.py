import numpy as np, math
rng = np.random.default_rng(0)
def bf16_decode(u16): return (np.uint32(u16).astype(np.uint32)<<16).view(np.float32)
def f_to_bf16(x):
    u=x.astype(np.float32).view(np.uint32); return ((u+0x8000)>>16).astype(np.uint16)
def fp8_dec(c):
    s=(c>>7)&1; e=(c>>3)&0xF; m=c&0x7
    v=(m/8.0)*2**-6 if e==0 else (1+m/8.0)*2**(e-7)
    return (-1 if s else 1)*v
def fp8_rtz(x):
    if x<=0: return 0
    e=math.floor(math.log2(x))
    if e<-6: return 0
    M=int((x*2**-e-1)*8); E=e+7
    if E<1: return 0
    if E>15: return 0x7E
    if E==15 and M==7: M=6
    return (E<<3)|max(0,min(7,M))
def ilog2(n): return int(n).bit_length()-1
def ienc_fp8(p,sm,e,FRAC):  # integer encode of t=(p/sm)*2^-e
    if p<=0: return 0
    R=(int(p)<<FRAC)//int(sm)
    if R==0: return 0
    top=ilog2(R); lg=top-FRAC-e; E=lg+7
    if E<1:
        rs=1-E; return ((R>>(top-3+rs))&0x7) if (top-3+rs)>=0 and rs<=3 else 0
    M=((R>>(top-3))&0x7) if top>=3 else ((R<<(3-top))&0x7)
    if E>15: return 0x7E
    if E==15 and M==7: M=6
    return (E<<3)|M
def ibf16_to_q(u16,SL):
    u16=int(u16); s=(u16>>15)&1; e=(u16>>7)&0xFF; m=u16&0x7F
    mant=m if e==0 else (0x80|m); shift=(e-127-7+SL)
    if shift>=0: q=mant<<shift
    else: r=-shift; q=(mant+(1<<(r-1)))>>r
    return -q if s else q
def consts(SL):
    Sq=2**-SL
    return round(math.log(2)/Sq),round(2**16/round(math.log(2)/Sq)),round(1.353/Sq),round(0.344/(0.3585*Sq*Sq))
def iexp(qarg,QLN2,QLN2_INV,QB,QC):
    z=(-qarg*QLN2_INV)>>16; qp=qarg+z*QLN2; return ((qp+QB)*(qp+QB)+QC)>>z

def run(S,SL,FRAC,exp_mode):
    QLN2,QLN2_INV,QB,QC=consts(SL); TGT=4
    w5=[]
    for _ in range(300):
        sc=rng.normal(0,4,S).astype(np.float32); bf=f_to_bf16(sc)
        true=np.exp(sc-sc.max()); true/=true.sum()
        q=[ibf16_to_q(u,SL) for u in bf]; mq=max(q)
        if exp_mode=='iexp': p=[iexp(qi-mq,QLN2,QLN2_INV,QB,QC) for qi in q]
        else: # true exp scaled to integer pseudo-prob
            p=[int(round(math.exp((qi-mq)*2**-SL)*(QB*QB+QC))) for qi in q]
        sm=sum(p); G=S//32; deq=np.zeros(S)
        for g in range(G):
            grp=p[g*32:(g+1)*32]; gmax=max(grp)
            if gmax==0: continue
            e=ilog2(gmax)-ilog2(sm)-TGT
            for j in range(32):
                c=ienc_fp8(p[g*32+j],sm,e,FRAC); deq[g*32+j]=fp8_dec(c)*2**e
        w5.append(np.mean(np.abs(deq-true)<=0.05*np.abs(true)+1e-6))
    return np.mean(w5)*100
# float ref
def run_float(S):
    w5=[]
    for _ in range(300):
        sc=rng.normal(0,4,S).astype(np.float32); bf=f_to_bf16(sc)
        true=np.exp(sc-sc.max()); true/=true.sum()
        row=bf16_decode(bf).astype(np.float64); mx=row.max(); ex=np.exp(row-mx); pr=ex/ex.sum()
        G=S//32; deq=np.zeros(S)
        for g in range(G):
            grp=pr[g*32:(g+1)*32]; gmax=abs(grp).max(); e=(math.floor(math.log2(gmax/16.0))+1) if gmax>0 else 0
            for j in range(32): c=fp8_rtz(pr[g*32+j]*2**-e); deq[g*32+j]=fp8_dec(c)*2**e
        w5.append(np.mean(np.abs(deq-true)<=0.05*np.abs(true)+1e-6))
    return np.mean(w5)*100
print(f"FLOAT ref                : {run_float(128):.1f}%")
for SL in (4,6,8,10):
    print(f"INT true-exp  SL=1/{2**SL:<4} FRAC=24: {run(128,SL,24,'trueexp'):.1f}%   | INT iexp SL=1/{2**SL:<4}: {run(128,SL,24,'iexp'):.1f}%")
