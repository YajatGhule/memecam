// vision_parity: run the C++ Tracker over the clips dumped by tools/dump_mediapipe.py (MediaPipe Python) and
// report landmark error / detection agreement / gesture agreement per model, plus timings. The C++ tracker is
// single-person: compare against dumps made with `--max-people 1`.
//   build/vision_parity /tmp/parity/*.json [--async] [--max-frames N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "config.h"
#include "vision/tracker.h"

using json = nlohmann::json;
using namespace memecam;

namespace {

// Hand recall vs the body pose: a pose wrist that is clearly visible (and inside the frame) should have a hand
// whose wrist is near it. Counts such wrists and how many have a hand, for MediaPipe and for C++.
struct Recall {
    long wrists = 0, py_hit = 0, cpp_hit = 0, py_only = 0, cpp_only = 0;
};
bool has_hand_near(const std::vector<cv::Point2d>& hand_wrists, cv::Point2d w, double r) {
    for (auto& h : hand_wrists)
        if (cv::norm(h - w) < r) return true;
    return false;
}

struct Acc {   // error statistics of one model
    std::vector<double> err;   // per matched instance: mean landmark error / scale
    long frames = 0, count_agree = 0, ref_n = 0, cpp_n = 0, matched = 0;
    long label_agree = 0, gesture_agree = 0;
    std::map<std::string, long> confusions;
    std::vector<double> lm_err;   // per-landmark sum of normalized error (hands)
    void add_counts(size_t ref, size_t cpp) {
        ++frames;
        count_agree += ref == cpp;
        ref_n += ref;
        cpp_n += cpp;
    }
};

double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}
double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))];
}

using Pts = std::vector<cv::Point2d>;

// Greedy matching of instances by the mean distance of all points, normalized by the reference scale.
template <class F>
std::vector<std::pair<int, int>> match(const std::vector<Pts>& ref, const std::vector<Pts>& cpp,
                                       const std::vector<double>& scale, F&& accept) {
    std::vector<std::tuple<double, int, int>> c;
    for (size_t i = 0; i < ref.size(); ++i)
        for (size_t j = 0; j < cpp.size(); ++j) {
            double d = 0;
            for (size_t k = 0; k < ref[i].size(); ++k) d += cv::norm(ref[i][k] - cpp[j][k]);
            d /= ref[i].size() * scale[i];
            c.emplace_back(d, int(i), int(j));
        }
    std::sort(c.begin(), c.end());
    std::vector<bool> ui(ref.size()), uj(cpp.size());
    std::vector<std::pair<int, int>> out;
    for (auto& [d, i, j] : c)
        if (!ui[i] && !uj[j] && accept(d)) {
            ui[i] = uj[j] = true;
            out.emplace_back(i, j);
        }
    return out;
}

double mean_err(const Pts& a, const Pts& b, double scale, const std::vector<int>* idx = nullptr) {
    double s = 0;
    int n = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        if (idx && std::find(idx->begin(), idx->end(), int(k)) == idx->end()) continue;
        s += cv::norm(a[k] - b[k]);
        ++n;
    }
    return n ? s / n / scale : 0;
}

void report(const char* name, const Acc& a, const char* unit) {
    std::printf("  %-6s frames %ld, count agreement %.1f%% (ref %ld / c++ %ld instances), matched %ld\n", name, a.frames,
                100.0 * a.count_agree / std::max(1L, a.frames), a.ref_n, a.cpp_n, a.matched);
    if (!a.err.empty())
        std::printf("         landmark error (%% of %s): mean %.2f  median %.2f  p95 %.2f  max %.2f\n", unit,
                    100 * mean(a.err), 100 * pct(a.err, 0.5), 100 * pct(a.err, 0.95), 100 * pct(a.err, 1.0));
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> files;
    bool async = false;
    long max_frames = 0;
    const bool verbose = std::getenv("PARITY_VERBOSE") != nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--async") async = true;
        else if (a == "--max-frames" && i + 1 < argc) max_frames = std::atol(argv[++i]);
        else files.push_back(a);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: vision_parity <dump.json>... [--async] [--max-frames N]\n");
        return 2;
    }
    Acc tot_h, tot_f, tot_p;
    Recall tot_r;
    double tot_frames = 0, tot_secs = 0, tot_py_secs = 0;
    for (const auto& file : files) {
        std::ifstream in(file);
        json ref = json::parse(in);
        const std::string clip = ref["clip"];
        const bool mirror = ref.value("mirror", true);
        cv::VideoCapture cap(clip);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "cannot open %s\n", clip.c_str());
            return 1;
        }
        Config cfg;
        cfg.device = "cpu";
        const bool stateless = ref.value("stateless", false);   // dump of fresh-tracker frames (idx field)
        std::unique_ptr<Tracker> tracker_p = std::make_unique<Tracker>(cfg, true, async);
        long read_idx = -1;
        Acc h, f, p;
        Recall rec;
        long n = 0, decode_bad = 0;
        double secs = 0;
        cv::Mat bgr, rgb;
        for (const auto& fr : ref["frames"]) {
            if (max_frames && n >= max_frames) break;
            if (stateless) {
                const long want = fr["idx"].get<long>();
                bool ok = true;
                while (ok && read_idx < want) ok = cap.read(bgr), ++read_idx;
                if (!ok) break;
                tracker_p = std::make_unique<Tracker>(cfg, true, async);   // no temporal state
            } else if (!cap.read(bgr)) {
                break;
            }
            if (mirror) cv::flip(bgr, bgr, 1);
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            if (fr.contains("pixsum") && fr["pixsum"].get<long long>() >= 0) {
                const cv::Scalar s = cv::sum(rgb);
                if ((long long)(s[0] + s[1] + s[2]) != fr["pixsum"].get<long long>()) ++decode_bad;
            }
            const auto t0 = std::chrono::steady_clock::now();
            tracker_p->process(rgb, fr["ts"].get<int64_t>());
            secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const TrackerRaw& raw = tracker_p->last_raw();
            ++n;

            // hands: error normalized by the reference hand size (wrist -> middle MCP)
            std::vector<Pts> rh, ch;
            std::vector<double> rs;
            for (const auto& hh : fr["hands"]) {
                Pts pts;
                for (const auto& q : hh["pts"]) pts.emplace_back(q[0].get<double>(), q[1].get<double>());
                rs.push_back(cv::norm(pts[9] - pts[0]) + 1e-6);
                rh.push_back(std::move(pts));
            }
            for (const auto& hh : raw.hands) {
                Pts pts;
                for (const auto& q : hh.pts) pts.emplace_back(q[0], q[1]);
                ch.push_back(std::move(pts));
            }
            h.add_counts(rh.size(), ch.size());
            if (verbose) {
                std::printf("%ld ts %lld py:", n - 1, (long long)fr["ts"].get<int64_t>());
                for (auto& q : rh) std::printf(" (%.1f,%.1f|%.1f,%.1f)", q[0].x, q[0].y, q[9].x, q[9].y);
                std::printf("  c++:");
                for (auto& q : ch) std::printf(" (%.1f,%.1f|%.1f,%.1f)", q[0].x, q[0].y, q[9].x, q[9].y);
                std::printf("\n");
            }
            for (auto [i, j] : match(rh, ch, rs, [](double d) { return d < 1.0; })) {
                ++h.matched;
                h.err.push_back(mean_err(rh[i], ch[j], rs[i]));
                if (h.lm_err.empty()) h.lm_err.assign(21, 0.0);
                for (int k = 0; k < 21; ++k) h.lm_err[k] += cv::norm(rh[i][k] - ch[j][k]) / rs[i];
                const auto& rj = fr["hands"][i];
                h.label_agree += rj["label"].get<std::string>() == raw.hands[j].label;
                const bool g = rj["gesture"].get<std::string>() == raw.hands[j].gesture;
                h.gesture_agree += g;
                if (!g) ++h.confusions[rj["gesture"].get<std::string>() + "->" + raw.hands[j].gesture];
            }
            // hand recall against MediaPipe's own pose (same wrists for both)
            if (!fr["poses"].empty()) {
                const auto& pp = fr["poses"][0];
                const int W = ref.value("width", 1280), H = ref.value("height", 720);
                cv::Point2d l11(pp[11][0].get<double>(), pp[11][1].get<double>());
                cv::Point2d l12(pp[12][0].get<double>(), pp[12][1].get<double>());
                const double sw = cv::norm(l11 - l12);
                std::vector<cv::Point2d> pyw, cw;
                for (auto& q : rh) pyw.push_back(q[0]);
                for (auto& q : ch) cw.push_back(q[0]);
                for (int k : {15, 16}) {
                    const auto& q = pp[k];
                    cv::Point2d w(q[0].get<double>(), q[1].get<double>());
                    if (q[3].get<double>() < 0.8 || w.x < 20 || w.y < 20 || w.x > W - 20 || w.y > H - 20 || sw < 40)
                        continue;
                    ++rec.wrists;
                    const bool a = has_hand_near(pyw, w, 0.5 * sw), b = has_hand_near(cw, w, 0.5 * sw);
                    rec.py_hit += a;
                    rec.cpp_hit += b;
                    rec.py_only += a && !b;
                    rec.cpp_only += b && !a;
                }
            }
            // faces: normalized by the reference face width (234 - 454)
            std::vector<Pts> rf, cf;
            std::vector<double> fs;
            for (const auto& ff : fr["faces"]) {
                Pts pts;
                for (const auto& q : ff) pts.emplace_back(q[0].get<double>(), q[1].get<double>());
                fs.push_back(cv::norm(pts[234] - pts[454]) + 1e-6);
                rf.push_back(std::move(pts));
            }
            for (const auto& ff : raw.faces) {
                Pts pts;
                for (const auto& q : ff.pts) pts.emplace_back(q[0], q[1]);
                cf.push_back(std::move(pts));
            }
            f.add_counts(rf.size(), cf.size());
            if (verbose && rf.size() != cf.size()) std::printf("%ld faces py %zu c++ %zu\n", n - 1, rf.size(), cf.size());
            for (auto [i, j] : match(rf, cf, fs, [](double d) { return d < 1.0; })) {
                ++f.matched;
                f.err.push_back(mean_err(rf[i], cf[j], fs[i]));
            }
            // poses: normalized by the reference shoulder width (11 - 12), all 33 landmarks
            std::vector<Pts> rp, cp;
            std::vector<double> ps;
            for (const auto& pp : fr["poses"]) {
                Pts pts;
                for (const auto& q : pp) pts.emplace_back(q[0].get<double>(), q[1].get<double>());
                ps.push_back(cv::norm(pts[11] - pts[12]) + 1e-6);
                rp.push_back(std::move(pts));
            }
            if (raw.pose_fresh || !async)
                for (const auto& pp : raw.poses) {
                    Pts pts;
                    for (const auto& q : *pp) pts.emplace_back(q.x, q.y);
                    cp.push_back(std::move(pts));
                }
            p.add_counts(rp.size(), cp.size());
            for (auto [i, j] : match(rp, cp, ps, [](double d) { return d < 1.0; })) {
                ++p.matched;
                p.err.push_back(mean_err(rp[i], cp[j], ps[i]));
                if (verbose)
                    std::printf("%ld pose err %.2f%% nose py (%.1f,%.1f) c++ (%.1f,%.1f)\n", n - 1,
                                100 * p.err.back(), rp[i][0].x, rp[i][0].y, cp[j][0].x, cp[j][0].y);
            }
        }
        std::printf("== %s: %ld frames, C++ %.1f fps (Python dump %.1f fps)%s\n", clip.c_str(), n, n / secs,
                    ref.value("fps", 0.0), decode_bad ? "  DECODE MISMATCH" : "");
        report("hands", h, "hand size");
        auto report_recall = [](const Recall& r) {
            if (r.wrists)
                std::printf("         hand recall at visible pose wrists: MediaPipe %.1f%%  C++ %.1f%%  (%ld wrists; "
                            "only MediaPipe %ld, only C++ %ld)\n", 100.0 * r.py_hit / r.wrists, 100.0 * r.cpp_hit / r.wrists,
                            r.wrists, r.py_only, r.cpp_only);
        };
        report_recall(rec);
        tot_r.wrists += rec.wrists, tot_r.py_hit += rec.py_hit, tot_r.cpp_hit += rec.cpp_hit;
        tot_r.py_only += rec.py_only, tot_r.cpp_only += rec.cpp_only;
        if (h.matched)
            std::printf("         handedness agreement %.1f%%, gesture agreement %.1f%%\n",
                        100.0 * h.label_agree / h.matched, 100.0 * h.gesture_agree / h.matched);
        if (!h.lm_err.empty() && h.matched) {
            std::printf("         per-landmark mean error %%:");
            for (double e : h.lm_err) std::printf(" %.0f", 100 * e / h.matched);
            std::printf("\n");
        }
        for (auto& [k, v] : h.confusions) std::printf("           gesture %s x%ld\n", k.c_str(), v);
        report("face", f, "face width");
        report("pose", p, "shoulder width");
        std::printf("  C++ %s", tracker_p->stats().c_str());
        if (ref.contains("timing_ms"))
            for (auto& [k, v] : ref["timing_ms"].items()) std::printf("  python %s %.1f ms/call\n", k.c_str(), v.get<double>());
        auto merge = [](Acc& t, const Acc& a) {
            t.err.insert(t.err.end(), a.err.begin(), a.err.end());
            t.frames += a.frames; t.count_agree += a.count_agree; t.ref_n += a.ref_n; t.cpp_n += a.cpp_n;
            t.matched += a.matched; t.label_agree += a.label_agree; t.gesture_agree += a.gesture_agree;
        };
        merge(tot_h, h);
        merge(tot_f, f);
        merge(tot_p, p);
        tot_frames += n;
        tot_secs += secs;
        tot_py_secs += n / std::max(1e-6, ref.value("fps", 0.0));
    }
    if (files.size() > 1) {
        std::printf("== TOTAL %.0f frames, C++ %.1f fps, Python %.1f fps\n", tot_frames, tot_frames / tot_secs,
                    tot_frames / tot_py_secs);
        report("hands", tot_h, "hand size");
        if (tot_r.wrists)
            std::printf("         hand recall at visible pose wrists: MediaPipe %.1f%%  C++ %.1f%%  (%ld wrists; "
                        "only MediaPipe %ld, only C++ %ld)\n", 100.0 * tot_r.py_hit / tot_r.wrists,
                        100.0 * tot_r.cpp_hit / tot_r.wrists, tot_r.wrists, tot_r.py_only, tot_r.cpp_only);
        if (tot_h.matched)
            std::printf("         handedness agreement %.1f%%, gesture agreement %.1f%%\n",
                        100.0 * tot_h.label_agree / tot_h.matched, 100.0 * tot_h.gesture_agree / tot_h.matched);
        report("face", tot_f, "face width");
        report("pose", tot_p, "shoulder width");
    }
    return 0;
}
