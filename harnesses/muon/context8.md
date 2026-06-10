Layer norm OUT[r,:] = (IN[r,:]-mean)/sqrt(var+1e-5)*gamma+beta, fp32, ROWS=64, COLS=768.
Reduction op (per-row mean+var) + affine. One threadblock (128 threads, 2 cores x 4 warps).
Real op: smolvla/openvla layer_norm over hidden=768. Tolerance rel 1e-3 / abs 2e-4.
