#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#include <miniaudio.h>

#include "audio.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace memecam {

namespace {
constexpr unsigned RATE = 44100;
constexpr float FADE = 0.08f;   // seconds of fade-out when a meme switches off

std::vector<float> load_clip(const std::string& path) {
    ma_decoder_config dc = ma_decoder_config_init(ma_format_f32, 1, RATE);   // decode + resample + downmix
    ma_decoder dec;
    std::vector<float> out;
    if (ma_decoder_init_file(path.c_str(), &dc, &dec) != MA_SUCCESS) {
        std::printf("[memecam] can't read sound %s\n", fs::path(path).filename().c_str());
        return out;
    }
    float buf[4096];
    for (;;) {
        ma_uint64 got = 0;
        if (ma_decoder_read_pcm_frames(&dec, buf, 4096, &got) != MA_SUCCESS && got == 0) break;
        out.insert(out.end(), buf, buf + got);
        if (got < 4096) break;
    }
    ma_decoder_uninit(&dec);
    return out;
}
}  // namespace

Sounds::Sounds(const Config& cfg, const std::string& assets_dir) : volume_(float(cfg.audio_volume)) {
    if (!cfg.audio_enabled) return;
    for (const MemeSpec& m : cfg.memes) {
        fs::path p = m.sound.empty() ? fs::path(assets_dir) / "sounds" / (m.name + ".wav")
                                     : fs::path(assets_dir) / m.sound;
        if (!fs::exists(p)) continue;
        auto clip = load_clip(p.string());
        if (!clip.empty()) clips_[m.name] = std::move(clip);
    }
    if (clips_.empty()) return;
    auto* dev = new ma_device;
    ma_device_config c = ma_device_config_init(ma_device_type_playback);
    c.playback.format = ma_format_f32;
    c.playback.channels = 1;
    c.sampleRate = RATE;
    c.dataCallback = [](ma_device* d, void* out, const void* in, ma_uint32 n) { callback(d, out, in, n); };
    c.pUserData = this;
    if (ma_device_init(nullptr, &c, dev) != MA_SUCCESS || ma_device_start(dev) != MA_SUCCESS) {
        std::printf("[memecam] no audio output; memes stay silent\n");
        delete dev;
        return;
    }
    device_ = dev;
}

Sounds::~Sounds() {
    if (auto* dev = static_cast<ma_device*>(device_)) {
        ma_device_uninit(dev);   // stops the callback thread first
        delete dev;
    }
}

void Sounds::update(const std::vector<std::string>& active) {
    if (!device_) return;
    std::lock_guard lock(mu_);
    auto on = [&](const std::string& n) { return !muted && std::find(active.begin(), active.end(), n) != active.end(); };
    for (const auto& n : active)
        if (on(n) && clips_.count(n) && !voices_.count(n)) voices_[n] = Voice{&clips_[n]};
    for (auto& [n, v] : voices_)
        if (!on(n) && v.step == 0.f) v.step = 1.f / (FADE * RATE);   // start fading out
}

void Sounds::callback(void* device, void* out, const void*, unsigned frames) {
    static_cast<Sounds*>(static_cast<ma_device*>(device)->pUserData)->mix(static_cast<float*>(out), frames);
}

void Sounds::mix(float* out, unsigned frames) {
    std::fill(out, out + frames, 0.f);
    std::lock_guard lock(mu_);
    for (auto it = voices_.begin(); it != voices_.end();) {
        Voice& v = it->second;
        const auto& x = *v.clip;
        const size_t k = std::min<size_t>(frames, x.size() - std::min(v.pos, x.size()));
        for (size_t i = 0; i < k; i++) {
            if (v.step != 0.f) v.gain = std::max(0.f, v.gain - v.step);
            out[i] += x[v.pos + i] * v.gain;
        }
        v.pos += k;
        // A finished clip stays (silent) until its meme switches off, so it doesn't restart; a faded one goes.
        if (v.step != 0.f && (v.gain <= 0.f || v.pos >= x.size())) it = voices_.erase(it);
        else ++it;
    }
    for (unsigned i = 0; i < frames; i++) out[i] = std::clamp(out[i] * volume_, -1.f, 1.f);
}

}  // namespace memecam
