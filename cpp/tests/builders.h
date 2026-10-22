// Synthetic MediaPipe-style landmarks — port of the builders in tests/test_gestures.py.
//
// Hands are built in a local "hand frame" (u = along the fingers, v = across the palm, z = depth,
// unit = wrist→middle-MCP distance), then rotated in the image plane, scaled to pixels and placed.
// Rotation 0° = fingers pointing up in the image, 90° = pointing right, 180° = pointing down.
#pragma once

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "face_noise.h"
#include "gestures.h"
#include "tracking.h"

namespace mt {

using namespace memecam;
using Fingers = std::set<std::string>;

inline const Fingers OPEN{"index", "middle", "ring", "pinky"};
inline const Fingers FIST{};
inline const Fingers PEACE{"index", "middle"};
inline const Fingers MIDDLE{"middle"};
inline const Fingers INDEX{"index"};
inline const double RATIO = 1.15;

inline Vec3 unit(double a, double b, double c) {
    double n = std::sqrt(a * a + b * b + c * c);
    return {a / n, b / n, c / n};
}

struct FingerDef { const char* name; int mcp; double v, u; double len[3]; double spread; };
inline const FingerDef FINGER_DEFS[4] = {
    {"index", 5, 0.30, 0.95, {0.45, 0.27, 0.22}, 0.25},
    {"middle", 9, 0.10, 1.00, {0.50, 0.30, 0.23}, 0.0},
    {"ring", 13, -0.10, 0.95, {0.46, 0.28, 0.22}, -0.05},
    {"pinky", 17, -0.28, 0.85, {0.36, 0.20, 0.20}, -0.12},
};
inline const double CURL_DIRS[3][3] = {{0.0, 0.20, -0.98}, {0.0, -0.90, -0.30}, {0.0, -0.80, 0.60}};

// `extended` = fingers that are straight; `angle` = degrees clockwise from 'fingers pointing up'.
// thumb_out: -1 = auto (out for an open hand), 0 / 1 = forced.
inline Hand make_hand(const Fingers& extended = {}, double angle = 0.0, cv::Point2d wrist = {640, 400},
                      double size = 110.0, const std::string& label = "Right", const std::string& gesture = "None",
                      int thumb_out = -1) {
    std::array<Vec3, 21> local{};   // (v, u, z)
    for (const auto& f : FINGER_DEFS) {
        local[f.mcp] = {f.v, f.u, 0.0};
        Vec3 dirs[3];
        if (extended.count(f.name)) {
            for (auto& d : dirs) d = unit(f.spread, 1.0, 0.0);
        } else {
            for (int k = 0; k < 3; ++k) dirs[k] = unit(CURL_DIRS[k][0], CURL_DIRS[k][1], CURL_DIRS[k][2]);
        }
        Vec3 p = local[f.mcp];
        for (int k = 0; k < 3; ++k) {
            p = p + f.len[k] * dirs[k];
            local[f.mcp + 1 + k] = p;
        }
    }
    bool out = thumb_out < 0 ? extended.size() >= 3 : thumb_out != 0;
    if (out) {
        local[1] = {0.25, 0.20, 0}; local[2] = {0.50, 0.40, 0}; local[3] = {0.70, 0.55, 0}; local[4] = {0.85, 0.70, 0};
    } else {
        local[1] = {0.25, 0.20, 0}; local[2] = {0.40, 0.40, -0.1}; local[3] = {0.30, 0.60, -0.3}; local[4] = {0.12, 0.70, -0.4};
    }
    double a = angle * (M_PI / 180.0);
    double up[2] = {std::sin(a), -std::cos(a)};
    double lat[2] = {std::cos(a), std::sin(a)};
    Hand h;
    for (int i = 0; i < 21; ++i) {
        double v = local[i][0], u = local[i][1];
        h.pts[i] = {wrist.x + size * (v * lat[0] + u * up[0]), wrist.y + size * (v * lat[1] + u * up[1]),
                    size * local[i][2]};
    }
    h.label = label;
    h.gesture = gesture;
    h.gesture_score = gesture != "None" ? 0.9 : 0.0;
    return h;
}

inline const cv::Point2d FACE_C{640.0, 300.0};
inline const double FACE_W = 200.0, FACE_H = 260.0;

// `mouth` = lip gap as a fraction of the face height (vary it between frames = talking).
inline Face make_face(double nose_ratio = 0.6, cv::Point2d center = FACE_C, double w = FACE_W, double h = FACE_H,
                      double mouth = 0.02) {
    double cx = center.x, cy = center.y, top = cy - 0.5 * h, chin = cy + 0.5 * h;
    Face f;
    f.pts.resize(478);
    for (int i = 0; i < 478; ++i)
        f.pts[i] = {cx + FACE_NOISE[i][0] * w / 2, cy + FACE_NOISE[i][1] * h / 2, FACE_NOISE[i][2]};
    f.pts[10] = {cx, top, 0};
    f.pts[152] = {cx, chin, 0};
    f.pts[1] = {cx, top + nose_ratio * (chin - top), -30};
    f.pts[13] = {cx, top + 0.78 * (chin - top), -10};
    f.pts[14] = {cx, top + (0.78 + mouth) * (chin - top), -10};
    f.pts[234] = {cx - w / 2, cy, 10};
    f.pts[454] = {cx + w / 2, cy, 10};
    return f;
}

inline Face talk_face(int i) { return make_face(0.6, FACE_C, FACE_W, FACE_H, 0.02 + 0.06 * (i % 2)); }

// (33, 4) body pose matching make_face(): nose, mouth, ears, shoulders and both wrists.
inline Pose make_pose_arr(cv::Point2d wl, cv::Point2d wr, double vis = 0.95) {
    Pose p{};
    double cx = FACE_C.x, cy = FACE_C.y;
    p[0] = {cx, cy - 10, 0, vis};
    p[9] = {cx - 20, cy + 40, 0, vis};
    p[10] = {cx + 20, cy + 40, 0, vis};
    double half = FACE_W / 1.35 / 2;
    p[7] = {cx - half, cy, 0, vis};
    p[8] = {cx + half, cy, 0, vis};
    p[11] = {cx - 120, 450, 0, vis};
    p[12] = {cx + 120, 450, 0, vis};
    p[15] = {wl.x, wl.y, 0, vis};
    p[16] = {wr.x, wr.y, 0, vis};
    return p;
}
inline std::shared_ptr<const Pose> make_pose(cv::Point2d wl, cv::Point2d wr, double vis = 0.95) {
    return std::make_shared<const Pose>(make_pose_arr(wl, wr, vis));
}

inline std::shared_ptr<const cv::Mat> motion_frac(double motion) {
    auto m = std::make_shared<cv::Mat>(cv::Mat::zeros(MOTION_H, MOTION_W, CV_8U));
    int n = int(motion * MOTION_W * MOTION_H);
    for (int k = 0; k < n; ++k) m->data[k] = 1;
    return m;
}

inline std::map<std::string, Detection> run(Detectors& det, const std::vector<Hand>& hands = {},
                                            std::optional<Face> face = std::nullopt, double now = 0.0,
                                            double motion = 0.0) {
    Frame fr;
    fr.hands = hands;
    fr.face = std::move(face);
    fr.motion = motion_frac(motion);
    fr.width = 1280;
    fr.height = 720;
    return det.run(fr, now);
}

inline std::set<std::string> raws(const std::map<std::string, Detection>& d) {
    std::set<std::string> out;
    for (const auto& [k, v] : d)
        if (v.raw) out.insert(k);
    return out;
}

inline std::set<std::string> intersect(const std::set<std::string>& a, const std::set<std::string>& b) {
    std::set<std::string> out;
    for (const auto& x : a)
        if (b.count(x)) out.insert(x);
    return out;
}

inline bool raw(const std::map<std::string, Detection>& d, const std::string& k) {
    auto it = d.find(k);
    return it != d.end() && it->second.raw;
}

inline bool same_hand(const Hand& a, const Hand& b) { return a.pts == b.pts && a.label == b.label; }
inline bool same_hands(const std::vector<Hand>& a, const std::vector<Hand>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_hand(a[i], b[i])) return false;
    return true;
}

}  // namespace mt
