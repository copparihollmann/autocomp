GELU C[m,n] = x * sigmoid(1.702*x), fp32 sigmoid-approx, M=64, N=512.
Elementwise; sigmoid via mu_exp (golden matches). One threadblock (128 threads).
Real op: smolvla/pi05 MLP gelu (1x1024x3072 tiled). Tolerance rel 1e-3 / abs 2e-4.
