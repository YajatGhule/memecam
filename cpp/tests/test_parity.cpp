// Logic parity with the Python reference: replay the cached MediaPipe tracking results of the recorded
// clips (tests/fixtures/replays.bin, from tools/export_fixtures.py) through Crowd with the v1 and v2
// configs. Every clip must fire the same memes, for the same person, at the same time (± one frame) as
// memecam.main.Crowd did, and every frame's detections (raw flags, scores) must match.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <tuple>

#include <nlohmann/json.hpp>

#include "config.h"
#include "crowd.h"

using namespace memecam;

namespace {

struct Clip {
    std::string source, file, expected;
    std::vector<std::pair<double, Frame>> frames;
};

class Reader {
public:
    explicit Reader(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        buf_.assign(std::istreambuf_iterator<char>(f), {});
    }
    bool ok() const { return !buf_.empty(); }
    template <typename T> T get() {
        if (pos_ + sizeof(T) > buf_.size()) throw std::runtime_error("fixture truncated");
        T v;
        std::memcpy(&v, buf_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        auto n = get<uint16_t>();
        std::string s(buf_.data() + pos_, n);
        pos_ += n;
        return s;
    }
    bool done() const { return pos_ == buf_.size(); }
private:
    std::vector<char> buf_;
    size_t pos_ = 0;
};

std::string fixture(const std::string& name) { return project_path("tests/fixtures/" + name); }

std::vector<Clip> load_clips() {
    Reader r(fixture("replays.bin"));
    if (!r.ok()) return {};
    char magic[4];
    for (char& c : magic) c = r.get<char>();
    if (std::string(magic, 4) != "MCFX" || r.get<uint32_t>() != 1) throw std::runtime_error("bad fixture header");
    std::map<uint32_t, std::shared_ptr<const Pose>> poses;
    std::vector<Clip> clips(r.get<uint32_t>());
    for (auto& clip : clips) {
        clip.source = r.str();
        clip.file = r.str();
        clip.expected = r.str();
        uint32_t n = r.get<uint32_t>();
        for (uint32_t k = 0; k < n; ++k) {
            Frame fr;
            double now = r.get<double>();
            fr.width = r.get<int32_t>();
            fr.height = r.get<int32_t>();
            fr.pid = r.get<int32_t>();
            if (r.get<uint8_t>()) {
                auto m = std::make_shared<cv::Mat>(cv::Mat::zeros(MOTION_H, MOTION_W, CV_8U));
                uint32_t runs = r.get<uint32_t>(), at = 0;
                for (uint32_t i = 0; i < runs; ++i) {
                    uint16_t len = r.get<uint16_t>();
                    if (i % 2) std::memset(m->data + at, 1, len);
                    at += len;
                }
                if (at != MOTION_W * MOTION_H) throw std::runtime_error("bad motion mask");
                fr.motion = m;
            }
            for (int i = r.get<uint8_t>(); i > 0; --i) {
                int old_id = r.get<int32_t>();
                fr.renamed[old_id] = r.get<int32_t>();
            }
            for (int i = r.get<uint8_t>(); i > 0; --i) {
                Frame q;
                q.pid = r.get<int32_t>();
                q.width = fr.width;
                q.height = fr.height;
                q.motion = fr.motion;
                if (r.get<uint8_t>()) {
                    std::vector<std::pair<int, Vec3>> pts(r.get<uint16_t>());
                    for (auto& [idx, p] : pts) {
                        idx = r.get<uint16_t>();
                        for (int c = 0; c < 3; ++c) p[c] = r.get<double>();
                    }
                    Vec3 nose;
                    for (auto& [idx, p] : pts)
                        if (idx == 1) nose = p;
                    Face f;
                    f.pts.assign(478, nose);
                    for (auto& [idx, p] : pts) f.pts[idx] = p;
                    q.face = std::move(f);
                }
                if (r.get<uint8_t>()) {
                    uint32_t uid = r.get<uint32_t>();
                    if (r.get<uint8_t>()) {
                        Pose p;
                        for (auto& l : p) {
                            l.x = r.get<double>();
                            l.y = r.get<double>();
                            l.z = r.get<double>();
                            l.visibility = r.get<double>();
                        }
                        poses[uid] = std::make_shared<const Pose>(p);
                    }
                    q.pose = poses.at(uid);
                }
                for (int h = r.get<uint8_t>(); h > 0; --h) {
                    Hand hand;
                    for (auto& p : hand.pts)
                        for (int c = 0; c < 3; ++c) p[c] = r.get<double>();
                    hand.label = r.str();
                    hand.gesture = r.str();
                    hand.gesture_score = r.get<double>();
                    q.hands.push_back(std::move(hand));
                }
                fr.people.push_back(std::move(q));
            }
            if (!fr.people.empty()) {   // the top-level frame mirrors its primary person
                fr.hands = fr.people[0].hands;
                fr.face = fr.people[0].face;
                fr.pose = fr.people[0].pose;
            }
            clip.frames.emplace_back(now, std::move(fr));
        }
    }
    if (!r.done()) throw std::runtime_error("trailing bytes in fixture");
    return clips;
}

using Fired = std::tuple<int, int, std::string>;   // frame, pid, meme

struct Result {
    std::vector<Fired> fired;
    std::vector<std::map<int, std::map<std::string, Detection>>> dets;
};

Result replay(const Config& cfg, const Clip& clip) {
    Crowd crowd(cfg);
    Result out;
    for (size_t k = 0; k < clip.frames.size(); ++k) {
        const auto& [now, fr] = clip.frames[k];
        std::set<std::pair<int, std::string>> was;
        for (const auto& [pid, p] : crowd.people())
            for (const auto& [name, m] : p->memes)
                if (m.started) was.emplace(p->pid, name);
        out.dets.push_back(crowd.update(fr, now));
        for (const auto& [pid, p] : crowd.people())
            for (const auto& [name, m] : p->memes)
                if (m.started && !was.count({p->pid, name})) out.fired.emplace_back(int(k), p->pid, name);
    }
    std::sort(out.fired.begin(), out.fired.end());
    return out;
}

// Same memes for the same people, each at the same frame ± 1.
bool same_fired(std::vector<Fired> a, std::vector<Fired> b) {
    if (a.size() != b.size()) return false;
    for (const auto& [k, pid, m] : a) {
        auto it = std::find_if(b.begin(), b.end(), [&](const Fired& x) {
            return std::get<1>(x) == pid && std::get<2>(x) == m && std::abs(std::get<0>(x) - k) <= 1;
        });
        if (it == b.end()) return false;
        b.erase(it);
    }
    return true;
}

std::string show(const std::vector<Fired>& v, const Clip& clip) {
    std::string s;
    for (const auto& [k, pid, m] : v) {
        char buf[96];
        std::snprintf(buf, sizeof buf, " %s#p%d@%.2f", m.c_str(), pid, clip.frames[k].first);
        s += buf;
    }
    return s.empty() ? " (none)" : s;
}

}  // namespace

TEST_CASE("logic parity with the Python replays", "[parity]") {
    if (!std::filesystem::exists(fixture("replays.bin")) || !std::filesystem::exists(fixture("expected.json")))
        SKIP("no fixtures: run tools/export_fixtures.py in the Python repo");
    auto clips = load_clips();
    std::ifstream jf(fixture("expected.json"));
    auto expected = nlohmann::json::parse(jf);
    REQUIRE(expected["clips"].size() == clips.size());
    std::vector<std::string> order = expected["dets_order"];

    Config v1 = load_config("config/config.toml");
    Config v2 = load_config("config/config.toml", "config/config_v2.toml");
    long frames = 0, det_rows = 0, raw_mismatch = 0, score_mismatch = 0;
    double max_score_diff = 0;
    for (size_t c = 0; c < clips.size(); ++c) {
        const Clip& clip = clips[c];
        const auto& ex = expected["clips"][c];
        REQUIRE(ex["file"] == clip.file);
        REQUIRE(ex["frames"] == clip.frames.size());
        frames += long(clip.frames.size());
        for (const char* key : {"v1", "v2"}) {
            Result got = replay(std::string(key) == "v1" ? v1 : v2, clip);
            std::vector<Fired> want;
            for (const auto& f : ex[key]["fired"]) want.emplace_back(int(f[2]), int(f[0]), std::string(f[1]));
            std::sort(want.begin(), want.end());
            bool ok = same_fired(got.fired, want);
            std::printf("%s %s %-12s %-30s python:%s\n%*s c++:   %s\n", ok ? "SAME" : "DIFF", key, clip.source.c_str(),
                        clip.file.c_str(), show(want, clip).c_str(), 49, "", show(got.fired, clip).c_str());
            CHECK(ok);
            // Per-frame detections: raw flags exact, scores within float noise.
            const auto& rows = ex[key]["dets"];
            for (size_t k = 0; k < clip.frames.size(); ++k) {
                const auto& row = rows[k];
                const auto& mine = got.dets[k];
                if (row.size() != mine.size()) { ++raw_mismatch; continue; }
                for (const auto& [pid_s, v] : row.items()) {
                    ++det_rows;
                    auto it = mine.find(std::stoi(pid_s));
                    if (it == mine.end()) { ++raw_mismatch; continue; }
                    long bits = 0;
                    for (size_t i = 0; i < order.size(); ++i) {
                        auto d = it->second.find(order[i]);
                        bool present = d != it->second.end();
                        if (present && d->second.raw) bits |= 1L << i;
                        const auto& s = v[1][i];
                        if (s.is_null() != !present) { ++score_mismatch; continue; }
                        if (present) {
                            double diff = std::abs(double(s) - d->second.score);
                            max_score_diff = std::max(max_score_diff, diff);
                            if (diff > 1e-6 * std::max(1.0, std::abs(double(s)))) ++score_mismatch;
                        }
                    }
                    if (bits != long(v[0])) {
                        ++raw_mismatch;
                        if (raw_mismatch <= 10)
                            UNSCOPED_INFO(key << " " << clip.file << " frame " << k << " pid " << pid_s << " raw python "
                                              << long(v[0]) << " c++ " << bits);
                    }
                }
            }
        }
    }
    std::printf("parity: %zu clips, %ld frames, %ld person-frames x2 configs: %ld raw mismatches, %ld score "
                "mismatches, max |score diff| %.3g\n",
                clips.size(), frames, det_rows / 2, raw_mismatch, score_mismatch, max_score_diff);
    CHECK(raw_mismatch == 0);
    CHECK(score_mismatch == 0);
}
