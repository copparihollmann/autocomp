from autocomp.hw_config.hardware_config import HardwareConfig


class MuonHardwareConfig(HardwareConfig):
    """Radiance Muon SIMT GPU (1 cluster, 2 cores) simulated on cyclotron."""

    def __init__(self, num_cores: int = 2, num_warps: int = 4, num_lanes: int = 16,
                 smem_size_kb: int = 128):
        self.num_cores = num_cores
        self.num_warps = num_warps  # default occupancy per core
        self.num_lanes = num_lanes
        self.smem_size_kb = smem_size_kb

    def __repr__(self):
        return (f"MuonHardwareConfig(cores={self.num_cores}, warps={self.num_warps}, "
                f"lanes={self.num_lanes}, smem={self.smem_size_kb}KB)")

    def get_hw_config_specific_rules(self) -> list[str]:
        threads = self.num_cores * self.num_warps * self.num_lanes
        return [
            f"The threadblock spans {self.num_cores} cores x {self.num_warps} warps "
            f"x {self.num_lanes} lanes = {threads} threads; tid_in_threadblock is "
            f"0..{threads - 1}.",
            f"Shared memory is {self.smem_size_kb} KiB per cluster; address 0 to 0x20000.",
            "FP32 SIMT ISA (RV32IM + Zfinx); 256 registers per thread.",
            "Memory loads are ~400-cycle latency; smem ~2 cycles.",
            "Barriers must include all warps: mu_barrier(id, total_warps).",
        ]
