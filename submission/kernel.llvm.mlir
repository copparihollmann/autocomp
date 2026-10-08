module {
  llvm.func @radiance_kernel(%W: !llvm.ptr, %A0: !llvm.ptr, %Y0: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cM = llvm.mlir.constant(16 : i64) : i64
    %cK = llvm.mlir.constant(16 : i64) : i64
    %cN = llvm.mlir.constant(16 : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cM : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mK = llvm.mul %mi, %cK : i64
    %mN = llvm.mul %mi, %cN : i64
    llvm.br ^n(%c0 : i64)
  ^n(%ni: i64):
    %nc = llvm.icmp "slt" %ni, %cN : i64
    llvm.cond_br %nc, ^nbody, ^mnext
  ^nbody:
    llvm.br ^k(%c0, %zero : i64, f32)
  ^k(%ki: i64, %acc: f32):
    %kc = llvm.icmp "slt" %ki, %cK : i64
    llvm.cond_br %kc, ^kbody, ^store
  ^kbody:
    %lidx = llvm.add %mK, %ki : i64
    %lp = llvm.getelementptr %A0[%lidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %lv = llvm.load %lp : !llvm.ptr -> f32
    %kN = llvm.mul %ki, %cN : i64
    %widx = llvm.add %kN, %ni : i64
    %wp = llvm.getelementptr %W[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wv = llvm.load %wp : !llvm.ptr -> f32
    %prod = llvm.fmul %lv, %wv : f32
    %acc2 = llvm.fadd %acc, %prod : f32
    %ki2 = llvm.add %ki, %c1 : i64
    llvm.br ^k(%ki2, %acc2 : i64, f32)
  ^store:
    %acc_k = llvm.fadd %acc, %zero : f32
    %oidx = llvm.add %mN, %ni : i64
    %op = llvm.getelementptr %Y0[%oidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %acc_k, %op : f32, !llvm.ptr
    %ni2 = llvm.add %ni, %c1 : i64
    llvm.br ^n(%ni2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return
  }
}
