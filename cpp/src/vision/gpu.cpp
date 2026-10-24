#include "vision/gpu.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#ifdef MEMECAM_HAVE_GPU
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#endif

namespace fs = std::filesystem;

namespace memecam::vision {

namespace {
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(std::string s) {
    s.erase(std::find(s.begin(), s.end(), '\0'), s.end());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    return s.substr(i);
}

// Vendor from the PCI id, else from the vendor string ("Advanced Micro Devices, Inc.", "NVIDIA Corporation").
std::string vendor_of(unsigned id, const std::string& text) {
    if (id == 0x1002 || id == 0x1022) return "amd";
    if (id == 0x10de) return "nvidia";
    if (id == 0x8086) return "intel";
    const std::string t = lower(text);
    if (t.find("advanced micro") != std::string::npos || t.find("amd") != std::string::npos) return "amd";
    if (t.find("nvidia") != std::string::npos) return "nvidia";
    if (t.find("intel") != std::string::npos) return "intel";
    return trim(text).empty() ? "unknown" : trim(text);
}

#ifdef MEMECAM_HAVE_GPU
namespace cl = tflite::gpu::cl;

std::string info_string(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (cl::clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    if (cl::clGetDeviceInfo(d, what, n, s.data(), nullptr) != CL_SUCCESS) return {};
    return trim(s);
}

template <typename T>
T info_value(cl_device_id d, cl_device_info what, T fallback) {
    T v{};
    return cl::clGetDeviceInfo(d, what, sizeof v, &v, nullptr) == CL_SUCCESS ? v : fallback;
}

constexpr cl_device_info kBoardNameAmd = 0x4038;   // CL_DEVICE_BOARD_NAME_AMD (cl_amd_device_attribute_query)
#endif
}  // namespace

std::vector<GpuDevice> gpu_devices() {
    std::vector<GpuDevice> out;
#ifdef MEMECAM_HAVE_GPU
    if (!cl::LoadOpenCL().ok()) return out;
    cl_uint np = 0;
    if (cl::clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return out;
    std::vector<cl_platform_id> platforms(np);
    if (cl::clGetPlatformIDs(np, platforms.data(), nullptr) != CL_SUCCESS) return out;
    for (cl_platform_id p : platforms) {
        cl_uint nd = 0;
        if (cl::clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        if (cl::clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_device_id d : devs) {
            GpuDevice g;
            g.vendor = vendor_of(info_value<cl_uint>(d, CL_DEVICE_VENDOR_ID, 0), info_string(d, CL_DEVICE_VENDOR));
            g.name = info_string(d, CL_DEVICE_NAME);
            // AMD's driver names its GPUs by codename (gfx90c, gfx1035); the board name is the real one.
            if (g.vendor == "amd") {
                const std::string board = info_string(d, kBoardNameAmd);
                if (!board.empty()) g.name = board;
            }
            if (g.name.empty()) g.name = "GPU";
            g.integrated = info_value<cl_bool>(d, CL_DEVICE_HOST_UNIFIED_MEMORY, CL_FALSE) == CL_TRUE;
            g.score = std::max(1u, info_value<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS, 1)) *
                      std::max(1u, info_value<cl_uint>(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, 1));
            out.push_back(std::move(g));
        }
    }
    // Same order as the delegate's CreateDefaultGPUDevice: the highest compute units x clock first.
    std::stable_sort(out.begin(), out.end(), [](const GpuDevice& a, const GpuDevice& b) { return a.score > b.score; });
#endif
    return out;
}

GpuChoice choose_gpu(std::string mode) {
    if (const char* e = std::getenv("MEMECAM_DEVICE"); e && *e) mode = e;
    mode = lower(mode.empty() ? "auto" : mode);
    if (mode == "cpu") return {false, "forced by config"};
#ifndef MEMECAM_HAVE_GPU
    return {false, mode == "gpu" ? "GPU requested, but this build has no LiteRT GPU delegate"
                                 : "LiteRT XNNPACK; no GPU delegate in this build"};
#else
    const auto gpus = gpu_devices();
    if (gpus.empty()) return {false, "no OpenCL GPU found (install / update the graphics driver for GPU tracking)"};
    const GpuDevice& g = gpus.front();
    const std::string desc = g.name + (g.integrated ? ", integrated" : "") + ", OpenCL";
    if (mode == "gpu") return {true, desc + "; forced by config"};
    if (g.vendor == "intel" && g.integrated)
        return {false, "only Intel integrated graphics (" + g.name + "): XNNPACK on the CPU is as fast"};
    return {true, desc};
#endif
}

std::string gpu_cache_dir() {
    fs::path base;
#ifdef _WIN32
    if (const char* e = std::getenv("LOCALAPPDATA"); e && *e) base = fs::path(e) / "memecam";
#else
    if (const char* e = std::getenv("XDG_CACHE_HOME"); e && *e) base = fs::path(e) / "memecam";
    else if (const char* h = std::getenv("HOME"); h && *h) base = fs::path(h) / ".cache" / "memecam";
#endif
    if (base.empty()) return {};
    const fs::path dir = base / "gpu-cache";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return fs::is_directory(dir, ec) ? dir.string() : std::string{};
}

}  // namespace memecam::vision
