// GPU choice logic (src/gpu_choice.cpp): the pure part of vision/gpu.cpp, tested without OpenCL.
#include <catch2/catch_test_macros.hpp>

#include "gpu_choice.h"

using namespace memecam::vision;

namespace {
GpuDevice gpu(const char* name, const char* vendor, bool integrated) {
    GpuDevice g;
    g.name = name, g.vendor = vendor, g.integrated = integrated, g.score = 1000;
    return g;
}
}  // namespace

TEST_CASE("vendor_of: PCI ids and vendor strings") {
    CHECK(vendor_of(0x1002, "") == "amd");
    CHECK(vendor_of(0x10de, "") == "nvidia");
    CHECK(vendor_of(0x8086, "") == "intel");
    CHECK(vendor_of(0, "Advanced Micro Devices, Inc.") == "amd");
    CHECK(vendor_of(0, "NVIDIA Corporation") == "nvidia");
    CHECK(vendor_of(0, "Intel(R) Corporation") == "intel");
    CHECK(vendor_of(0, "Acme GPU Inc ") == "Acme GPU Inc");
    CHECK(vendor_of(0, "") == "unknown");
}

TEST_CASE("decide_gpu: cpu mode never uses the GPU") {
    CHECK_FALSE(decide_gpu("cpu", {gpu("RTX", "nvidia", false)}).use);
    CHECK_FALSE(decide_gpu("CPU", {}).use);
}

TEST_CASE("decide_gpu: auto") {
    CHECK(decide_gpu("auto", {gpu("Radeon 680M", "amd", true)}).use);
    CHECK(decide_gpu("auto", {gpu("RTX 3060", "nvidia", false)}).use);
    CHECK(decide_gpu("", {gpu("RTX 3060", "nvidia", false)}).use);   // empty = auto
    CHECK(decide_gpu("auto", {gpu("Arc A380", "intel", false)}).use);   // dedicated Intel is fine

    auto intel = decide_gpu("auto", {gpu("UHD 620", "intel", true)});
    CHECK_FALSE(intel.use);
    CHECK(intel.why.find("Intel integrated") != std::string::npos);

    auto none = decide_gpu("auto", {});
    CHECK_FALSE(none.use);
    CHECK_FALSE(none.why.empty());
}

TEST_CASE("decide_gpu: gpu mode forces it") {
    auto c = decide_gpu("gpu", {gpu("UHD 620", "intel", true)});
    CHECK(c.use);
    CHECK(c.why.find("forced") != std::string::npos);
    CHECK_FALSE(decide_gpu("gpu", {}).use);
}

TEST_CASE("decide_gpu: build without the GPU delegate") {
    CHECK_FALSE(decide_gpu("auto", {gpu("RTX", "nvidia", false)}, false).use);
    CHECK_FALSE(decide_gpu("gpu", {gpu("RTX", "nvidia", false)}, false).use);
}
