#include "gpu_choice.h"

#include <algorithm>
#include <cctype>

namespace memecam::vision {

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

GpuChoice decide_gpu(std::string mode, const std::vector<GpuDevice>& gpus, bool have_delegate) {
    mode = lower(mode.empty() ? "auto" : mode);
    if (mode == "cpu") return {false, "forced by config"};
    if (!have_delegate)
        return {false, mode == "gpu" ? "GPU requested, but this build has no LiteRT GPU delegate"
                                     : "LiteRT XNNPACK; no GPU delegate in this build"};
    if (gpus.empty()) return {false, "no OpenCL GPU found (install / update the graphics driver for GPU tracking)"};
    const GpuDevice& g = gpus.front();
    const std::string desc = g.name + (g.integrated ? ", integrated" : "") + ", OpenCL";
    if (mode == "gpu") return {true, desc + "; forced by config"};
    if (g.vendor == "intel" && g.integrated)
        return {false, "only Intel integrated graphics (" + g.name + "): XNNPACK on the CPU is as fast"};
    return {true, desc};
}

}  // namespace memecam::vision
