// The pure part of the GPU choice (no OpenCL, no LiteRT): device description, vendor detection and the
// cpu / gpu / auto decision. Lives in the core library so it can be unit tested; vision/gpu.cpp feeds it the
// OpenCL device list.
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

struct GpuChoice {
    bool use = false;
    std::string why;   // for the "[memecam] tracking on ..." line
};

std::string lower(std::string s);
std::string trim(std::string s);   // also cuts at the first NUL (OpenCL strings)

// Vendor from the PCI id, else from the vendor string ("Advanced Micro Devices, Inc.", "NVIDIA Corporation").
std::string vendor_of(unsigned id, const std::string& text);

// mode: cpu | gpu | auto (anything else counts as auto). `gpus` has the GPU the delegate will use first.
// auto uses any NVIDIA / AMD GPU (dedicated or integrated Radeon) and any dedicated GPU, but not Intel
// integrated graphics. `have_delegate`: this build has the LiteRT GPU delegate.
GpuChoice decide_gpu(std::string mode, const std::vector<GpuDevice>& gpus, bool have_delegate = true);

}  // namespace memecam::vision
