// SoC host carrier: runs on the Rocket RV64 core. Resets and launches the Muon
// GPU, then polls until all cores finish. The GPU kernel performs verification
// and writes the verdict via its tohost ecall ((errors<<1)|1), which ends the
// VCS simulation -> TestDriver prints "*** PASSED ***" (code 0) or
// "*** FAILED *** (tohost = N)". Do not reset the GPU after finish: that would
// suppress the GPU's terminating tohost write (see kernels/launch/host.cpp).
#include <stdio.h>
#include <inttypes.h>
#include <radiance.h>

int main() {
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
    *tocpu = tohost;
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

    uint32_t finished = 0;
    while (!finished) {
        SYNC_GPU();
        finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    }
    return 0;
}
