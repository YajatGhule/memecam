// Meme sounds (memecam/audio.py): each meme's sound plays while the meme is on screen. A sound starts from
// the beginning when its meme switches on and fades out quickly when it switches off (or the clip ends);
// several memes are mixed into one output stream (miniaudio). Sounds: `sound = "..."` in a meme's table
// (relative to assets/), else assets/sounds/<meme>.wav when it exists (tools/fetch_sounds.sh). WAV, MP3 or FLAC.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "config.h"

namespace memecam {

class Sounds {
public:
    Sounds(const Config& cfg, const std::string& assets_dir);
    ~Sounds();
    Sounds(const Sounds&) = delete;
    Sounds& operator=(const Sounds&) = delete;

    bool ok() const { return device_ != nullptr; }   // an output device is open and some sounds loaded
    // Once per displayed frame: the names of the memes on screen right now.
    void update(const std::vector<std::string>& active);
    bool muted = false;

private:
    struct Voice { const std::vector<float>* clip; size_t pos = 0; float gain = 1.f, step = 0.f; };
    static void callback(void* device, void* out, const void* in, unsigned frames);
    void mix(float* out, unsigned frames);

    float volume_;
    std::map<std::string, std::vector<float>> clips_;   // mono float at RATE
    std::map<std::string, Voice> voices_;
    std::mutex mu_;
    void* device_ = nullptr;   // ma_device*
};

}  // namespace memecam
