#!/usr/bin/env python3
"""BOTH-ENGINES-ACTIVE timeline from an RTL trace (warp-spec dual-engine kernels).

MX-mesh-busy  : per matmul_tile_async issue (manager warp), [issue, issue+drain]. drain measured.
SIMT-worker-busy: worker warps (warp>=1) making forward progress = distinct-PC density per bin
                  above a threshold (barrier-spin collapses to <THRESH distinct PCs).
both-active   : |MX-busy ∩ SIMT-busy| / kernel_span   <-- the headline metric.
"""
import sqlite3, sys
from collections import defaultdict

def union_len(iv):
    iv=sorted((a,b) for a,b in iv if b>a)
    if not iv: return 0,[]
    out=[list(iv[0])]
    for a,b in iv[1:]:
        if a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return sum(b-a for a,b in out), out

def intersect(A,B):
    res=[];i=j=0
    A=sorted(A);B=sorted(B)
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]);hi=min(A[i][1],B[j][1])
        if hi>lo: res.append((lo,hi))
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return res

def analyze(path, mt_pc, drain, binsz=2000, thresh=12):
    c=sqlite3.connect(path);cur=c.cursor()
    lo,hi=cur.execute('select min(cycle),max(cycle) from inst').fetchone()
    span=hi-lo
    # MX mesh busy: matmul issues by any manager (pc=mt_pc)
    issues=[r[0] for r in cur.execute('select cycle from inst where pc=? order by cycle',(mt_pc,))]
    mx_iv=[(t,t+drain) for t in issues]
    mx_busy,mx_merged=union_len(mx_iv)
    # SIMT worker busy: worker warps forward progress
    rows=cur.execute('select cycle,pc from inst where warp>=1 and warp<=3').fetchall()
    b=defaultdict(set)
    for cyc,pc in rows: b[(cyc-lo)//binsz].add(pc)
    simt_iv=[]
    for k,pcs in b.items():
        if len(pcs)>=thresh:
            s=lo+k*binsz; simt_iv.append((s,s+binsz))
    simt_busy,simt_merged=union_len(simt_iv)
    # intersection
    both=intersect(mx_merged,simt_merged)
    both_busy=sum(h-l for l,h in both)
    any_busy,_=union_len(mx_merged+simt_merged)
    return dict(path=path.split('/')[-2],span=span,lo=lo,hi=hi,issues=issues,drain=drain,
        mx_busy=mx_busy,mx_pct=100*mx_busy/span,mx_merged=mx_merged,
        simt_busy=simt_busy,simt_pct=100*simt_busy/span,simt_merged=simt_merged,
        both_busy=both_busy,both_pct=100*both_busy/span,
        any_pct=100*any_busy/span,idle_pct=100*(span-any_busy)/span)

def report(d):
    print(f"\n=== {d['path']} ===")
    print(f"kernel span      : {d['span']} cyc  [{d['lo']},{d['hi']}]")
    print(f"matmul issues    : {len(d['issues'])} @ {d['issues']}  (drain={d['drain']} cyc/tile)")
    print(f"MX-mesh   busy   : {d['mx_busy']:>7} cyc  = {d['mx_pct']:5.1f}%   windows {[(a,b) for a,b in d['mx_merged']]}")
    print(f"SIMT-work busy   : {d['simt_busy']:>7} cyc  = {d['simt_pct']:5.1f}%")
    print(f"** BOTH-ACTIVE   : {d['both_busy']:>7} cyc  = {d['both_pct']:5.1f}% **   <-- headline")
    print(f"any-engine busy  : {d['any_pct']:5.1f}%   (idle {d['idle_pct']:.1f}% = prologue/tail/barrier)")

if __name__=='__main__':
    for path,mt,drain in [
        ('/scratch/agustin/projects/radiance-kernels/kernels/autocomp_overlap_layer_banked/trace_autocomp_overlap_layer_banked.sqlite',0x10022c88,10040),
    ]:
        report(analyze(path,mt,drain))
