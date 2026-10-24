// GPU detection and choice for the LiteRT GPU delegate (OpenCL backend). Detection goes through OpenCL itself
// (OpenCL.dll on Windows, libOpenCL.so on Linux), so it sees exactly the GPUs the delegate can use, AMD
// integrated Radeon graphics included. Without the delegate in the build (-DMEMECAM_GPU=OFF) nothing is found.
#pragma once

#include <string>
#include <vector>

namespace memecam::vision {

struct GpuDevice {
    std::string name;          // marketing name ("AMD Radeon(TM) 680M"), not AMD's "gfx1035" codename
    std::string vendor;        // "amd" / "nvidia" / "intel" / other vendor string
    bool integrated = false;   // shares memory with the CPU (APU / iGPU)
    unsigned score = 0;        // compute units x clock: the delegate picks the highest
};

// All OpenCL GPUs, the one the delegate will use first. Empty without OpenCL or the delegate.
std::vector<GpuDevice> gpu_devices();

struct GpuChoice {
    bool use = false;
    std::string why;   // for the "[memecam] tracking on ..." line
};

// mode (config `device`): cpu | gpu | auto. auto uses any NVIDIA / AMD GPU (dedicated or integrated Radeon)
// and any dedicated GPU, but not Intel integrated graphics, where the GPU path is no faster than XNNPACK.
// MEMECAM_DEVICE=cpu|gpu|auto in the environment overrides the config.
GpuChoice choose_gpu(std::string mode);

// Directory for the delegate's compiled-kernel cache (first start compiles OpenCL kernels, later ones load
// them). Empty if it can't be created.
std::string gpu_cache_dir();

}  // namespace memecam::vision
