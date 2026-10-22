#include "compose.h"

#include <algorithm>
#include <array>
#include <future>
#include <optional>

#include <opencv2/imgproc.hpp>

#include "mocap.h"

namespace memecam {

namespace {
int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }   // Python // for b > 0
constexpr std::array<const char*, 4> CORNERS = {"tl", "tr", "bl", "br"};
bool is_corner(const std::string& w) { return std::find(CORNERS.begin(), CORNERS.end(), w) != CORNERS.end(); }
}  // namespace

MemeMedia load_meme_media(const Config& cfg, const std::string& assets_dir) {
    // GIF decoding is ~10-25 ms per frame: decode every asset on its own thread (startup ~3x faster).
    std::vector<std::pair<std::string, std::future<std::unique_ptr<Media>>>> jobs;
    for (const MemeSpec& s : cfg.memes)
        jobs.emplace_back(s.name, std::async(std::launch::async, [&assets_dir, s] {
            return std::make_unique<Media>(assets_dir + "/" + s.asset, 520, s.key_black,
                                           s.title.empty() ? s.name : s.title, s.frames, s.loop_from);
        }));
    MemeMedia out;
    for (auto& [name, job] : jobs) out[name] = job.get();
    return out;
}

cv::Point corner(cv::Size frame, int w, int h, const std::string& where, int k) {
    int step = k * (w + 30);
    int x = where[1] == 'l' ? 20 + step : frame.width - w - 20 - step;
    int y = where[0] == 't' ? 80 : frame.height - h - 40;
    return {x, y};
}

std::vector<std::string> compose(cv::Mat& out, const Frame& fr, const std::vector<Shown>& shown, double now,
                                 const Config& cfg, const MemeMedia& media, CinemaEffect& cinema,
                                 bool show_markers, bool on_face) {
    const int H = out.rows, W = out.cols;
    struct Active { const Shown* s; const MemeSpec* spec; const MemeState* m; const Media* media; double alpha; };
    std::vector<Active> active;
    for (const Shown& s : shown) {
        for (const MemeSpec& spec : cfg.memes) {   // config order, like the Python dicts
            auto it = s.memes.find(spec.name);
            if (it == s.memes.end() || !it->second.started) continue;
            auto mi = media.find(spec.name);
            if (mi == media.end() || !mi->second) continue;
            double a = std::clamp((now - *it->second.started) / FADE_IN, 0.0, 1.0);
            active.push_back({&s, &spec, &it->second, mi->second.get(), a});
        }
    }
    const Face* face = fr.face ? &*fr.face : nullptr;   // the tracked person's face (memes are theirs)
    auto frame_of = [&](const Active& a) -> const cv::Mat& { return a.media->at(now - *a.m->started); };

    double cine = -1;
    for (const Active& a : active)
        if (a.spec->placement == "cinema") cine = std::max(cine, a.alpha);
    if (cine >= 0) cinema.apply(out, cine);
    if (!on_face) {
        for (const Active& a : active)
            if (a.spec->placement == "face")
                if (face) warp_on_face(out, frame_of(a), face->pts, a.alpha);
    }

    if (show_markers) draw_markers(out, fr);

    std::map<std::string, int> stacked;
    for (const Active& a : active) {
        const std::string where = on_face ? "over_face" : a.spec->placement;
        const auto& head = a.s->head;
        if (where == "face" && face) continue;
        if (where == "over_face") {
            if (head) over_face(out, frame_of(a), *head, a.alpha);
            continue;
        }
        cv::Mat img = fit(frame_of(a), int(H * a.spec->height));
        const int h = img.rows, w = img.cols;
        if (where == "center") {
            // Big and frameless, horizontally centred: over the chest (just under the chin), or at the top of
            // the frame when the face sits low, whichever hides less of the face.
            const int low = std::max(30, H - h - 10), top = 40;
            int y = int(H * 0.4), x = floordiv(W - w, 2);
            if (head) {
                int fy0 = head->y, fy1 = y1_of(*head);
                int cands[2] = {std::min(std::max(fy1 - floordiv(h, 8), top), low), top};
                auto overlap = [&](int c) { return std::max(0, std::min(c + h, fy1) - std::max(c, fy0)); };
                y = overlap(cands[1]) < overlap(cands[0]) ? cands[1] : cands[0];
            }
            blend(out, img, x, std::min(y, low), a.alpha);
            continue;
        }
        cv::Point p;
        if (is_corner(where) || where == "cinema") {
            std::string c = where == "cinema" ? "tl" : where;
            p = corner(out.size(), w, h, c, stacked[c]++);
        } else {
            Box box = a.m->anchor ? *a.m->anchor : Box(W / 2 - 1, H / 2 - 1, 1, 1);
            p = place_beside(out.size(), box, w, h);
        }
        panel(out, img, p.x, p.y, a.spec->title, a.alpha, ACCENT_COL);
    }

    std::vector<std::string> names;
    for (const Active& a : active)
        names.push_back(a.spec->name);
    return names;
}

void debug_panel(cv::Mat& img, const std::map<std::string, Detection>& dets, const Config& cfg) {
    int y = 90;
    for (const MemeSpec& s : cfg.memes) {
        auto it = dets.find(s.name);
        double score = it != dets.end() ? it->second.score : 0.0;
        bool on = it != dets.end() && it->second.raw;
        cv::Scalar col = on ? cv::Scalar(80, 255, 80) : cv::Scalar(180, 180, 180);
        cv::rectangle(img, cv::Point(img.cols - 230, y - 11),
                      cv::Point(img.cols - 230 + int(std::min(score, 1.0) * 100), y + 1), col, -1);
        cv::putText(img, s.name.substr(0, 16), cv::Point(img.cols - 120, y), cv::FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                    cv::LINE_AA);
        y += 20;
    }
}

}  // namespace memecam
