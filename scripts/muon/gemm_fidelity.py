import sys, torch
sys.path.insert(0,"/scratch/agustin/projects/radiance-kernels/lib/mxgemmini")
import fp8_matmul_model as G
torch.manual_seed(0)
def mxq(M,spec):
    q,s=G.matrix_mx_requantize(M,spec); return G.make_fp_quantizer(spec,"nearest")(q), s
def run(Mm,Nn,Kk,spec):
    A=torch.randn(Mm,Kk); B=torch.randn(Kk,Nn)
    ref=(A@B).flatten().double()
    Aq,As=mxq(A,spec); Bqt,Bst=mxq(B.t().contiguous(),spec)
    o=G.tiled_matmul_hwlike(Aq,Bqt.t().contiguous(),As,Bst.t().contiguous(),verbose=False).flatten().double()
    return (ref@o/(ref.norm()*o.norm())).item(), ((o-ref).norm()/ref.norm()).item()
for spec,lbl in [("fp8:e4m3","fp8"),("fp6:e3m2","fp6"),("fp4:e2m1","fp4")]:
    c,r=run(128,128,2048,spec); print(f"GEMM_FIDELITY 128x128x2048 {lbl}: cosine={c:.5f} rel-err={100*r:.2f}%",flush=True)
