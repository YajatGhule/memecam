// Config: config/config.toml (+ config/config_v2.toml layered on top for memecamv2). Mirrors the Python dicts.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace memecam {

struct MemeSpec {
    std::string name, asset, title, placement = "beside";
    std::string sound;                            // relative to assets/; empty = assets/sounds/<name>.wav if present
    double height = 0.35, hold = 0.3, min_show = 0.0, linger = 0.3, cooldown = 0.0;
    bool key_black = false;
    std::optional<std::pair<int, int>> frames;    // trim [first, last]
    std::optional<int> loop_from;
};

struct Config {
    // [camera]
    int camera_index = 0, width = 1280, height = 720;
    bool mirror = true;
    // [tracking]
    std::string device = "auto";
    // [person] (one person is tracked: the closest)
    double min_face_width = 0.08, forget = 1.0;
    // [display]
    bool meme_on_face = false;
    // [audio]
    bool audio_enabled = true;
    double audio_volume = 0.8;
    // [thresholds]: free-form numbers/bools (bools stored as 0/1), read via thr(key, default)
    std::map<std::string, double> thresholds;
    // [memes.*] in file order (base file first, then memes new in the overlay)
    std::vector<MemeSpec> memes;

    double thr(const std::string& key, double def) const {
        auto it = thresholds.find(key);
        return it == thresholds.end() ? def : it->second;
    }
    const MemeSpec* meme(const std::string& name) const {
        for (const auto& m : memes)
            if (m.name == name) return &m;
        return nullptr;
    }
};

// Load `base` and, if given, merge `overlay` on top (tables merged key by key, like Python merge()).
// Relative paths that don't exist from the working directory are resolved against MEMECAM_ROOT.
// Throws std::runtime_error on a missing file or a TOML syntax error.
Config load_config(const std::string& base, const std::string& overlay = "");

void set_project_root(const std::string& dir);   // before the first project_path() call: use dir instead of the exe / source tree
std::string project_path(const std::string& rel);   // MEMECAM_ROOT / rel (absolute paths unchanged)

}  // namespace memecam
