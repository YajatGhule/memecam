// 1:1 port of memecam/gestures.py. Numerics follow numpy float64 (including its summation order:
// sequential for short / axis-0 reductions, pairwise for long contiguous ones) so the C++ detectors
// make the same decisions on the same landmarks.
#include "gestures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>

namespace memecam {

namespace {

const std::map<std::string, double> DEFAULTS = {
    {"finger_extended_ratio", 1.15},   // |wrist→tip| / |wrist→pip| above this = extended
    {"cover_dist", 0.75},              // hand within this many face-widths of nose/mouth = covering it
    {"cross_dist", 0.25},              // gojo: index & middle tips closer than this (hand-sizes) = crossed
    {"peace_gap", 0.35},               // V sign: index & middle tips at least this far apart (hand-sizes)
    {"head_tilt_up", 0.10},            // dafoe: drop in nose position ratio vs calibrated baseline
    {"motion", 0.07},                  // talking-to-wall: fraction of changed pixels around the face
    {"hand_speed", 3.0},               // talking-to-wall: wrist speed in face-widths / second
    {"mahoraga_dist", 1.85},           // mahoraga: knuckles within this many face-widths of each other
    {"stuck_speed", 2.0},              // scuba: the hand on the face moves slower than this (face-widths/s)
    {"swim_speed", 2.5},               // scuba: the other hand moves faster than this (face-widths/s)
    {"mouth_move", 0.12},              // talking: lip-gap change rate (face-heights/s) that counts as talking
    {"mouth_window", 1.0},             // talking: mouth must have moved within this many seconds
    {"bounce_speed", 1.0},             // 67: opposite vertical wrist speed (face-widths / second, RMS)
};

constexpr double BOUNCE_TAU = 0.4, BOUNCE_WINDOW = 1.2, BOUNCE_CORR = 0.5, SIGN_TAU = 0.25;
constexpr double SWING_WINDOW = 1.0, SWING_REACH = 2.0, SWING_TOP = 0.3, SWING_MIN_MOTION = 0.02,
                 SWING_CORR = 0.5, SWING_AMP = 0.4;
constexpr double MOTION_REACH = 3.0, FLYING_KEEP = 0.6, PRESENT_SIDE = 0.7, PRESENT_GAP = 0.25;
constexpr double PRESENT_LOW[2] = {0.5, 3.0};

constexpr int P_NOSE = 0, P_MOUTH[2] = {9, 10}, P_EARS[2] = {7, 8}, P_SHOULDERS[2] = {11, 12},
              P_WRISTS[2] = {15, 16}, P_INDEX[2] = {19, 20};
constexpr int POSE_HAND_ENDS[2][4] = {{15, 17, 19, 21}, {16, 18, 20, 22}};   // wrist, pinky, index, thumb per arm
constexpr double VIS = 0.5;
constexpr int JAW[13] = {172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397};

using P2 = cv::Point2d;
double dist(const P2& a, const P2& b) { return norm2(a.x - b.x, a.y - b.y); }
double vlen(const P2& v) { return norm2(v.x, v.y); }
double dist3(const Vec3& a, const Vec3& b) {
    double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}
P2 lm(const Pose& p, int i) { return {p[i].x, p[i].y}; }

// Mean of hand points (numpy .mean(axis=0) on a few rows: sequential sum, then divide).
template <size_t N>
P2 mean_pts(const Hand& h, const int (&idx)[N]) {
    double sx = 0, sy = 0;
    for (int i : idx) { sx += h.pts[i][0]; sy += h.pts[i][1]; }
    return {sx / double(N), sy / double(N)};
}

// numpy's pairwise summation of a contiguous float64 array (what .sum() / .mean() do on 1-D data).
double pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}
double np_mean(const std::vector<double>& v) { return pairwise_sum(v.data(), v.size()) / double(v.size()); }
double np_std(const std::vector<double>& v) {
    double m = np_mean(v);
    std::vector<double> x(v.size());
    for (size_t i = 0; i < v.size(); ++i) { double d = v[i] - m; x[i] = d * d; }
    return std::sqrt(pairwise_sum(x.data(), x.size()) / double(x.size()));
}

// Python slice [start:stop] bounds on an axis of length n (negative stop counts from the end).
std::pair<int, int> py_slice(int start, int stop, int n) {
    auto norm = [n](int k) { return k < 0 ? std::max(0, k + n) : std::min(k, n); };
    int a = norm(start), b = norm(stop);
    return {a, std::max(a, b)};
}

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

struct BouncePrev {
    bool pose;                                  // source: pose wrists / tracked hands
    std::shared_ptr<const Pose> pid;            // the pose result (identity), null for hands
    double t;
    P2 lw, rw;
};

struct BounceWrists {
    bool pose;
    std::shared_ptr<const Pose> pid;
    P2 lw, rw;
};

}  // namespace

const std::map<std::string, double>& detector_defaults() { return DEFAULTS; }

int FingerStates::count() const { return int(index) + int(middle) + int(ring) + int(pinky); }

FingerStates finger_states(const Hand& hand, double ratio) {
    const Vec3& w = hand.pts[0];
    auto ext = [&](int pip, int tip) { return dist3(hand.pts[tip], w) > ratio * dist3(hand.pts[pip], w); };
    FingerStates s;
    s.index = ext(6, 8);
    s.middle = ext(10, 12);
    s.ring = ext(14, 16);
    s.pinky = ext(18, 20);
    return s;
}

std::optional<HeadRef> head_ref(const Frame& fr) {
    if (fr.face) {
        const auto& p = fr.face->pts;
        return HeadRef{P2{(p[1][0] + p[13][0]) / 2, (p[1][1] + p[13][1]) / 2}, fr.face->width()};
    }
    if (fr.pose && (*fr.pose)[P_NOSE].visibility > VIS) {
        const Pose& p = *fr.pose;
        P2 mouth{(0.0 + p[P_MOUTH[0]].x + p[P_MOUTH[1]].x) / 2, (0.0 + p[P_MOUTH[0]].y + p[P_MOUTH[1]].y) / 2};
        P2 ref{(p[P_NOSE].x + mouth.x) / 2, (p[P_NOSE].y + mouth.y) / 2};
        return HeadRef{ref, EAR_TO_FACE * dist(lm(p, P_EARS[0]), lm(p, P_EARS[1])) + 1e-6};
    }
    return std::nullopt;
}

struct Detectors::State {
    std::optional<double> nose_baseline;
    int face_frames = 0;
    std::vector<P2> prev_wrists;
    std::optional<double> prev_t;
    double hand_speed = 0.0;
    std::vector<double> hand_speeds;           // per tracked hand (same order as frame.hands), smoothed
    double motion = 0.0;
    std::shared_ptr<const Pose> prev_pose;     // (id, t, pose) of the last pose result used for arm speeds
    double prev_pose_t = 0.0;
    double arm_speed[2] = {0.0, 0.0};          // pose wrists (user's left, right), face-widths / second
    std::optional<std::pair<double, double>> prev_gap;
    double mouth_vel = 0.0;
    double mouth_at = -1e9;
    std::vector<int> covering_tracked;         // indices into the current frame's hands
    std::optional<BouncePrev> bounce_prev;
    double bounce_anti = 0.0, bounce_energy = 0.0;
    int bounce_sign = 0;
    std::vector<double> bounce_flips;
    std::vector<std::array<double, 3>> swing;  // (t, left, right motion height / face width)
    double swing_corr = 0.0, swing_amp = 0.0;
    int swing_crossings = 0;
    double signing_frac = 0.0;
    std::optional<double> signing_t;
    bool flying = false;
    double present_at = -1e9;
};

Detectors::Detectors(const std::map<std::string, double>& thresholds)
    : cfg_(DEFAULTS), st_(std::make_unique<State>()) {
    for (const auto& [k, v] : thresholds) cfg_[k] = v;
}
Detectors::~Detectors() = default;
Detectors::Detectors(Detectors&&) noexcept = default;
Detectors& Detectors::operator=(Detectors&&) noexcept = default;

double Detectors::cfg(const std::string& key, double def) const {
    auto it = cfg_.find(key);
    return it == cfg_.end() ? def : it->second;
}

void Detectors::calibrate() {
    st_->nose_baseline.reset();
    st_->face_frames = 0;
}

double Detectors::motion_level(const Frame& fr) const {
    if (!fr.motion || !fr.width) return 0.0;
    const cv::Mat& m = *fr.motion;
    const double sx = double(MOTION_W) / fr.width, sy = double(MOTION_H) / fr.height;
    int c0 = 0, c1 = m.cols;
    if (auto ref = head_ref(fr)) {
        double x = ref->nose.x, fw = ref->width;
        auto [a, b] = py_slice(std::max(0, int((x - MOTION_REACH * fw) * sx)),
                               std::max(0, int((x + MOTION_REACH * fw) * sx) + 1), m.cols);
        c0 = a;
        c1 = b;
    }
    int fr0 = 0, fr1 = 0, fc0 = 0, fc1 = 0;
    if (fr.face) {
        Box b = fr.face->bbox();
        int x0 = b.x, y0 = b.y, x1 = b.x + b.width, y1 = b.y + b.height;
        std::tie(fr0, fr1) = py_slice(std::max(0, int(y0 * sy) - 2), int(y1 * sy) + 3, m.rows);
        std::tie(fc0, fc1) = py_slice(std::max(0, int(x0 * sx) - 2), int(x1 * sx) + 3, m.cols);
    }
    long count = 0;
    for (int r = 0; r < m.rows; ++r) {
        const uint8_t* row = m.ptr<uint8_t>(r);
        bool in_face_rows = r >= fr0 && r < fr1;
        for (int c = c0; c < c1; ++c)
            if (row[c] && !(in_face_rows && c >= fc0 && c < fc1)) ++count;
    }
    return double(count) / double(m.rows * m.cols);
}

std::map<std::string, Detection> Detectors::run(const Frame& fr, double now) {
    State& S = *st_;
    const auto& c = cfg_;
    auto C = [&](const char* k) { return c.at(k); };
    std::map<std::string, Detection> d;
    const Face* face = fr.face ? &*fr.face : nullptr;
    std::optional<Box> fbox;
    if (face) fbox = face->bbox();
    const auto& hands = fr.hands;
    const int nh = int(hands.size());
    std::vector<FingerStates> states;
    std::vector<int> n_ext;
    for (const auto& h : hands) {
        states.push_back(finger_states(h, C("finger_extended_ratio")));
        n_ext.push_back(states.back().count());
    }
    auto find = [&](auto pred) {
        for (int i = 0; i < nh; ++i)
            if (pred(i)) return i;
        return -1;
    };
    auto bbox_or = [&](int i, std::optional<Box> def) { return i >= 0 ? std::optional<Box>(hands[i].bbox()) : def; };
    auto P = [&](int i, int k) { return xy(hands[i].pts[k]); };
    auto two_up = [](const FingerStates& s) { return s.index && s.middle && !s.ring && !s.pinky; };
    auto tip_gap = [&](int i) { return dist(P(i, 8), P(i, 12)) / hands[i].size(); };

    // ---- one-hand finger signs ----
    int mid = find([&](int i) { const auto& s = states[i]; return s.middle && !(s.index || s.ring || s.pinky); });
    d["middle_finger"] = Detection{mid >= 0, mid >= 0 ? 1.0 : 0.0, bbox_or(mid, std::nullopt)};

    auto crossed = [&](int i) {
        P2 a = P(i, 8) - P(i, 12), b = P(i, 5) - P(i, 9);
        bool flip = a.x * b.x + a.y * b.y < 0;
        return tip_gap(i) < C("cross_dist") || flip;
    };
    auto alone = [&](int i) {
        for (int o = 0; o < nh; ++o)
            if (o != i && !(dist(P(o, 9), P(i, 9)) > 1.5 * hands[i].size())) return false;
        return true;
    };
    auto fingers_up = [&](int i) {
        P2 v = P(i, 12) - P(i, 0);
        return -v.y > 0.5 * vlen(v);
    };
    int gojo = find([&](int i) {
        const auto& s = states[i];
        return s.index && s.middle && !s.ring && crossed(i) && alone(i) && fingers_up(i);
    });
    d["gojo"] = Detection{gojo >= 0, gojo >= 0 ? 1.0 : 0.0, bbox_or(gojo, fbox)};

    int vsign = find([&](int i) {
        return i != gojo && tip_gap(i) > C("peace_gap") && (two_up(states[i]) || hands[i].gesture == "Victory");
    });

    // ---- v2 gestures (only with [thresholds] extra_gestures = true) ----
    const bool extra = cfg("extra_gestures", 0.0) != 0.0;

    auto pinched = [&](int i) {
        const Hand& h = hands[i];
        static const int TIPS[5] = {4, 8, 12, 16, 20};
        P2 ctr = mean_pts(h, TIPS);
        double spread = 0;
        double sum_y = 0;
        for (int k : TIPS) {
            spread = std::max(spread, dist(P(i, k), ctr));
            sum_y += h.pts[k][1];
        }
        spread /= h.size();
        bool up = h.pts[0][1] - sum_y / 5.0 > cfg("pinch_up", 0.8) * h.size();
        bool reach = std::min(dist(P(i, 8), P(i, 0)), dist(P(i, 12), P(i, 0))) > 1.15 * h.size();
        return spread < cfg("pinch_spread", 0.3) && up && reach;
    };
    int pinch = find([&](int i) { return extra && i != gojo && pinched(i); });
    d["godfather"] = Detection{pinch >= 0, pinch >= 0 ? 1.0 : 0.0, bbox_or(pinch, std::nullopt)};

    auto on_jaw = [&](int i) {
        if (!face || !states[i].index) return false;
        const Hand& h = hands[i];
        double fw = face->width();
        double tip_d = INFINITY, seg_d = INFINITY;
        for (int j : JAW) {
            P2 jp = xy(face->pts[j]);
            tip_d = std::min(tip_d, dist(jp, P(i, 8)));
            seg_d = std::min(seg_d, dist(jp, P(i, 7)));
        }
        tip_d /= fw;
        seg_d /= fw;
        bool below_mouth = h.pts[8][1] > face->pts[13][1];
        static const int PALM[5] = {0, 5, 9, 13, 17};
        P2 palm = mean_pts(h, PALM);
        P2 mouth{(face->pts[13][0] + face->pts[1][0]) / 2, (face->pts[13][1] + face->pts[1][1]) / 2};
        return std::min(tip_d, seg_d) < cfg("jaw_dist", 0.12) && below_mouth && dist(palm, mouth) / fw > 0.45;
    };
    int jaw_hand = find([&](int i) { return extra && i != pinch && i != gojo && alone(i) && on_jaw(i); });
    d["looksmaxxing"] = Detection{jaw_hand >= 0, jaw_hand >= 0 ? 1.0 : 0.0, fbox};

    auto thumb_up = [&](int i) {
        const Hand& h = hands[i];
        if (h.gesture == "Thumb_Up") return true;
        if (states[i].count() > 0 || h.gesture == "Closed_Fist") return false;
        P2 t = P(i, 4) - P(i, 2);
        return -t.y > 0.7 * vlen(t) && vlen(t) > 0.45 * h.size();
    };
    int thumbs = find([&](int i) { return i != pinch && i != jaw_hand && thumb_up(i); });
    d["thumbs_up"] = Detection{thumbs >= 0, thumbs >= 0 ? 1.0 : 0.0, bbox_or(thumbs, std::nullopt)};

    auto pointing_up = [&](int i) {
        P2 v = P(i, 8) - P(i, 5);
        return -v.y > 0.5 * vlen(v);
    };
    auto on_face = [&](int i) {
        if (!face) return false;
        double x = hands[i].pts[8][0], y = hands[i].pts[8][1];
        return fbox->x < x && x < fbox->x + fbox->width && fbox->y < y && y < fbox->y + fbox->height;
    };
    int point = find([&](int i) {
        const auto& s = states[i];
        return i != vsign && i != gojo && i != thumbs && i != pinch && i != jaw_hand && pointing_up(i) && !on_face(i) &&
               (hands[i].gesture == "Pointing_Up" || (s.index && !s.middle && !s.ring && !s.pinky));
    });
    d["nerd"] = Detection{point >= 0 && mid < 0, point >= 0 ? 1.0 : 0.0, bbox_or(point, std::nullopt)};

    std::vector<int> signing;
    for (int h : {mid, gojo, vsign, point, thumbs, pinch, jaw_hand})
        if (h >= 0) signing.push_back(h);
    std::vector<int> fists, can_cover, busy, open_hands;
    for (int i = 0; i < nh; ++i)
        if ((n_ext[i] == 0 || hands[i].gesture == "Closed_Fist") && i != thumbs && i != pinch && i != jaw_hand)
            fists.push_back(i);
    for (int i = 0; i < nh; ++i)
        if (n_ext[i] >= 3 && !contains(signing, i) && !contains(fists, i)) can_cover.push_back(i);
    for (int i = 0; i < nh; ++i)
        if (!contains(can_cover, i)) busy.push_back(i);

    // ---- motion ----
    const double scale = face ? face->width() : (nh ? hands[0].size() * 2.5 : 1.0);
    {   // update_speed: match each wrist to the nearest unclaimed wrist from the previous frame
        std::vector<double> speeds;
        std::vector<P2> wrists;
        for (const auto& h : hands) wrists.push_back(xy(h.pts[0]));
        std::vector<double> per_hand(nh, 0.0);
        if (S.prev_t && now > *S.prev_t && !S.prev_wrists.empty()) {
            std::vector<std::tuple<double, int, int>> pairs;
            for (int i = 0; i < int(wrists.size()); ++i)
                for (int j = 0; j < int(S.prev_wrists.size()); ++j)
                    pairs.emplace_back(dist(wrists[i], S.prev_wrists[j]), i, j);
            std::sort(pairs.begin(), pairs.end());
            std::set<int> used_i, used_j;
            for (auto& [dpx, i, j] : pairs) {
                if (!used_i.count(i) && !used_j.count(j)) {
                    used_i.insert(i);
                    used_j.insert(j);
                    double v = std::min(dpx / scale / (now - *S.prev_t), 10.0);
                    speeds.push_back(v);
                    per_hand[i] = S.hand_speeds[j] + 0.35 * (v - S.hand_speeds[j]);
                }
            }
        }
        S.prev_wrists = wrists;
        S.prev_t = now;
        S.hand_speeds = per_hand;
        double inst = speeds.empty() ? 0.0 : *std::max_element(speeds.begin(), speeds.end());
        S.hand_speed += 0.25 * (inst - S.hand_speed);
    }
    auto ref = head_ref(fr);
    {   // update_arms: smoothed wrist speed of each arm from the body pose
        const double fw = ref ? ref->width : scale;
        const auto& p = fr.pose;
        if (p && !(S.prev_pose && S.prev_pose.get() == p.get())) {
            if (S.prev_pose && now > S.prev_pose_t) {
                const Pose& q = *S.prev_pose;
                double t = S.prev_pose_t;
                for (int arm = 0; arm < 2; ++arm) {
                    int i = P_WRISTS[arm];
                    double v = 0.0;
                    if ((*p)[i].visibility > VIS && q[i].visibility > VIS)
                        v = std::min(dist(lm(*p, i), lm(q, i)) / fw / (now - t), 10.0);
                    S.arm_speed[arm] += 0.35 * (v - S.arm_speed[arm]);
                }
            }
            S.prev_pose = p;
            S.prev_pose_t = now;
        }
    }
    {   // update_mouth: how fast the lips open and close (talking), in face-heights per second
        if (!face) {
            S.prev_gap.reset();
        } else {
            const auto& fp = face->pts;
            double gap = dist(xy(fp[13]), xy(fp[14])) / (dist(xy(fp[10]), xy(fp[152])) + 1e-6);
            if (S.prev_gap && now > S.prev_gap->second) {
                S.mouth_vel += 0.2 * (std::abs(gap - S.prev_gap->first) / (now - S.prev_gap->second) - S.mouth_vel);
                if (S.mouth_vel > C("mouth_move")) S.mouth_at = now;
            }
            S.prev_gap = std::make_pair(gap, now);
        }
    }
    S.motion += 0.25 * (motion_level(fr) - S.motion);
    auto speed_of = [&](int i) { return i >= 0 && i < int(S.hand_speeds.size()) ? S.hand_speeds[i] : 0.0; };

    // ---- hands over the face: scuba cat / higuruma ----
    std::vector<double> cover_speeds;
    std::set<int> cover_arms;
    S.covering_tracked.clear();
    if (ref) {
        P2 mouth = ref->nose;
        double reach = C("cover_dist") * ref->width;
        static const int PALM[5] = {0, 5, 9, 13, 17};
        static const int TIPS[5] = {4, 8, 12, 16, 20};
        for (int i : can_cover) {
            const Hand& h = hands[i];
            double near = std::min(dist(mean_pts(h, PALM), mouth), dist(P(i, 9), mouth));
            double top = INFINITY;
            for (int k : TIPS) top = std::min(top, h.pts[k][1]);
            if (near < reach && top < mouth.y) S.covering_tracked.push_back(i);
        }
        std::vector<double> tracked, from_pose;
        for (int i : S.covering_tracked) tracked.push_back(speed_of(i));
        if (fr.pose) {
            const Pose& p = *fr.pose;
            for (int arm = 0; arm < 2; ++arm) {
                std::vector<int> pts;
                for (int k : POSE_HAND_ENDS[arm])
                    if (p[k].visibility > 0.3) pts.push_back(k);
                if (pts.empty()) continue;
                double dmin = INFINITY, ymin = INFINITY;
                for (int k : pts) {
                    dmin = std::min(dmin, dist(lm(p, k), mouth));
                    ymin = std::min(ymin, p[k].y);
                }
                if (dmin > reach) continue;
                if (ymin > mouth.y) continue;           // under the chin, not over the mouth
                P2 wrist = lm(p, POSE_HAND_ENDS[arm][0]);
                bool on_busy = false;
                for (int b : busy)
                    if (dist(P(b, 0), wrist) < 1.5 * hands[b].size()) { on_busy = true; break; }
                if (on_busy) continue;
                from_pose.push_back(S.arm_speed[arm]);
                cover_arms.insert(arm);
            }
        }
        cover_speeds = tracked.size() >= from_pose.size() ? tracked : from_pose;
    }
    const int covering = int(cover_speeds.size());
    const bool higuruma = covering >= 2;
    std::vector<double> others;
    for (int i = 0; i < nh && i < int(S.hand_speeds.size()); ++i)
        if (!contains(S.covering_tracked, i)) others.push_back(S.hand_speeds[i]);
    if (!cover_arms.empty())
        for (int a = 0; a < 2; ++a)
            if (!cover_arms.count(a)) others.push_back(S.arm_speed[a]);
    double max_other = others.empty() ? 0.0 : *std::max_element(others.begin(), others.end());
    bool swimming = max_other > C("swim_speed") || S.motion > 0.02;
    bool scuba = covering == 1 && !higuruma && cover_speeds[0] < C("stuck_speed") && swimming;
    d["higuruma"] = Detection{higuruma, covering / 2.0, fbox};
    d["scuba_cat"] = Detection{scuba, covering / 2.0, fbox};

    // ---- dafoe: head tilted back ----
    if (face) {
        double tilt = 0.0;
        if (covering == 0) {   // head_tilt
            const auto& fp = face->pts;
            double top = fp[10][1], nose = fp[1][1], chin = fp[152][1];
            double r = (nose - top) / (chin - top + 1e-6);
            S.face_frames += 1;
            if (!S.nose_baseline) S.nose_baseline = r;
            double delta = *S.nose_baseline - r;
            if (S.face_frames <= 15) {          // warm-up: settle on the neutral pose, no triggers yet
                *S.nose_baseline += 0.25 * (r - *S.nose_baseline);
                tilt = 0.0;
            } else {
                double rate = std::abs(delta) < C("head_tilt_up") * 0.5 ? 0.02 : delta > 0 ? 0.004 : 0.001;
                *S.nose_baseline += rate * (r - *S.nose_baseline);
                tilt = delta;
            }
        }
        d["dafoe"] = Detection{tilt > C("head_tilt_up"), tilt / C("head_tilt_up") * 0.5, fbox};
    }

    // ---- two-arm poses ----
    const Pose* p = fr.pose.get();
    const bool arms = p && std::min((*p)[P_WRISTS[0]].visibility, (*p)[P_WRISTS[1]].visibility) > VIS &&
                      std::min((*p)[P_SHOULDERS[0]].visibility, (*p)[P_SHOULDERS[1]].visibility) > VIS;
    for (int i = 0; i < nh; ++i)
        if (n_ext[i] >= 3 || hands[i].gesture == "Open_Palm") open_hands.push_back(i);

    bool cinema = false;
    if (arms) {
        double ew = dist(lm(*p, P_EARS[0]), lm(*p, P_EARS[1])) + 1e-6;
        P2 lw = lm(*p, P_WRISTS[0]), rw = lm(*p, P_WRISTS[1]), ls = lm(*p, P_SHOULDERS[0]), rs = lm(*p, P_SHOULDERS[1]);
        bool flip = ls.x > rs.x;
        std::pair<P2, P2> outer_l = flip ? std::make_pair(lw, ls) : std::make_pair(rw, rs);
        std::pair<P2, P2> outer_r = flip ? std::make_pair(rw, rs) : std::make_pair(lw, ls);
        cinema = lw.y < ls.y - 0.6 * ew && rw.y < rs.y - 0.6 * ew && outer_l.first.x > outer_l.second.x &&
                 outer_r.first.x < outer_r.second.x && fists.empty();
    } else if (open_hands.size() == 2 && face) {
        int fx0 = fbox->x, fx1 = fbox->x + fbox->width;
        std::vector<int> lr = open_hands;
        std::stable_sort(lr.begin(), lr.end(), [&](int a, int b) { return hands[a].pts[0][0] < hands[b].pts[0][0]; });
        bool above = true;
        for (int i : open_hands)
            if (!(hands[i].pts[9][1] < face->pts[152][1])) above = false;
        cinema = above && hands[lr[0]].pts[9][0] < fx0 && hands[lr[1]].pts[9][0] > fx1;
    }
    d["absolute_cinema"] = Detection{cinema, cinema ? 1.0 : 0.0, fbox};

    // Mahoraga: Megumi's sign, two fists knuckle to knuckle in front of you.
    bool mahoraga = false;
    double score = 0.0;
    std::optional<Box> box;
    if (!fists.empty() && open_hands.empty() && covering == 0 && !cinema) {
        if (arms && std::min((*p)[P_INDEX[0]].visibility, (*p)[P_INDEX[1]].visibility) > 0.3) {
            double fw = ref ? ref->width : 1.0;
            double knuckles = dist(lm(*p, P_INDEX[0]), lm(*p, P_INDEX[1])) / fw;
            bool below_nose = std::min((*p)[P_WRISTS[0]].y, (*p)[P_WRISTS[1]].y) > (*p)[P_NOSE].y;
            mahoraga = knuckles < C("mahoraga_dist") && below_nose;
            score = std::max(0.0, 1 - knuckles / 4);
        } else if (fists.size() == 2) {
            const Hand& h1 = hands[fists[0]];
            const Hand& h2 = hands[fists[1]];
            double dd = dist(xy(h1.pts[9]), xy(h2.pts[9])) / ((h1.size() + h2.size()) / 2);
            bool chest = !face || std::min(h1.pts[9][1], h2.pts[9][1]) > face->pts[152][1] + 0.25 * face->width();
            mahoraga = dd < 1.8 && chest;
            score = std::max(0.0, 1 - dd / 4);
        }
        int x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
        for (int i : fists) {
            Box b = hands[i].bbox();
            x0 = std::min(x0, b.x); y0 = std::min(y0, b.y);
            x1 = std::max(x1, b.x + b.width); y1 = std::max(y1, b.y + b.height);
        }
        box = box_from_corners(x0, y0, x1, y1);
    }
    d["mahoraga"] = Detection{mahoraga, score, box};

    // ---- 67: open hands at chest height bouncing up and down in alternation ----
    const double fw = ref ? ref->width : scale;
    std::optional<BounceWrists> wrists;
    if (p && std::min((*p)[P_WRISTS[0]].visibility, (*p)[P_WRISTS[1]].visibility) > VIS) {
        P2 a = lm(*p, P_WRISTS[0]), b = lm(*p, P_WRISTS[1]);
        if (b.x < a.x) std::swap(a, b);
        wrists = BounceWrists{true, fr.pose, a, b};
    } else if (nh == 2 && open_hands.size() == 2) {
        P2 a = P(open_hands[0], 0), b = P(open_hands[1], 0);
        if (b.x < a.x) std::swap(a, b);
        wrists = BounceWrists{false, nullptr, a, b};
    }
    {   // update_bounce
        if (!wrists) {
            S.bounce_prev.reset();
            S.bounce_anti = S.bounce_energy = 0.0;
            S.bounce_sign = 0;
            S.bounce_flips.clear();
        } else {
            const auto& prev = S.bounce_prev;
            bool same_pose = prev && wrists->pose && prev->pose && prev->pid.get() == wrists->pid.get();
            if (!same_pose) {
                if (prev && prev->pose == wrists->pose && now > prev->t) {
                    double dt = now - prev->t;
                    double vl = std::clamp((wrists->lw.y - prev->lw.y) / fw / dt, -10.0, 10.0);
                    double vr = std::clamp((wrists->rw.y - prev->rw.y) / fw / dt, -10.0, 10.0);
                    double a = 1 - std::exp(-dt / BOUNCE_TAU);
                    S.bounce_anti += a * (-vl * vr - S.bounce_anti);
                    S.bounce_energy += a * ((vl * vl + vr * vr) / 2 - S.bounce_energy);
                    double rel = vl - vr;
                    if (std::abs(rel) > C("bounce_speed")) {
                        int sign = rel > 0 ? 1 : -1;
                        if (S.bounce_sign && sign != S.bounce_sign) S.bounce_flips.push_back(now);
                        S.bounce_sign = sign;
                    }
                }
                std::vector<double> keep;
                for (double t : S.bounce_flips)
                    if (now - t <= BOUNCE_WINDOW) keep.push_back(t);
                S.bounce_flips = keep;
                S.bounce_prev = BouncePrev{wrists->pose, wrists->pid, now, wrists->lw, wrists->rw};
            }
        }
    }
    const double amp = std::sqrt(std::max(S.bounce_anti, 0.0));
    const double corr = S.bounce_anti / (S.bounce_energy + 1e-6);

    bool swinging = false;
    {   // update_swing (67 from frame differencing)
        std::optional<std::pair<double, double>> hs;   // motion_heights
        if (fr.motion && ref && fr.width) {
            const cv::Mat& m = *fr.motion;
            double cx = ref->nose.x, cy = ref->nose.y, rw = ref->width;
            const double sx = double(MOTION_W) / fr.width, sy = double(MOTION_H) / fr.height;
            int y0 = std::max(0, int((cy - SWING_TOP * rw) * sy));
            double out[2];
            bool ok = true;
            const double ranges[2][2] = {{cx - SWING_REACH * rw, cx - 0.45 * rw}, {cx + 0.45 * rw, cx + SWING_REACH * rw}};
            for (int side = 0; side < 2 && ok; ++side) {
                auto [r0, r1] = py_slice(y0, m.rows, m.rows);
                auto [c0, c1] = py_slice(std::max(0, int(ranges[side][0] * sx)),
                                         std::max(0, std::min(MOTION_W, int(ranges[side][1] * sx))), m.cols);
                long size = long(r1 - r0) * (c1 - c0), cnt = 0;
                double row_sum = 0;   // sum of row indices of changed pixels (exact in double)
                for (int r = r0; r < r1; ++r) {
                    const uint8_t* row = m.ptr<uint8_t>(r);
                    for (int cc = c0; cc < c1; ++cc)
                        if (row[cc]) { ++cnt; row_sum += r - r0; }
                }
                if (size == 0 || double(cnt) / double(size) < SWING_MIN_MOTION) { ok = false; break; }
                out[side] = (row_sum / double(cnt) + y0) / sy / rw;
            }
            if (ok) hs = std::make_pair(out[0], out[1]);
        }
        if (hs) S.swing.push_back({now, hs->first, hs->second});
        std::vector<std::array<double, 3>> keep;
        for (const auto& x : S.swing)
            if (now - x[0] <= SWING_WINDOW) keep.push_back(x);
        S.swing = keep;
        if (S.swing.size() < 8 || now - S.swing[0][0] < 0.6 * SWING_WINDOW) {
            S.swing_corr = S.swing_amp = 0.0;
            S.swing_crossings = 0;
        } else {
            const size_t n = S.swing.size();
            double m0 = 0, m1 = 0;   // mean(axis=0): sequential
            for (const auto& x : S.swing) { m0 += x[1]; m1 += x[2]; }
            m0 /= double(n);
            m1 /= double(n);
            std::vector<double> prod(n), sq(2 * n), diff(n);
            for (size_t i = 0; i < n; ++i) {
                double a0 = S.swing[i][1] - m0, a1 = S.swing[i][2] - m1;
                prod[i] = a0 * a1;
                sq[2 * i] = a0 * a0;
                sq[2 * i + 1] = a1 * a1;
                diff[i] = a0 - a1;
            }
            double anti = -np_mean(prod);
            double scorr = anti / (np_mean(sq) + 1e-9);
            double samp = np_std(diff);
            int crossings = 0, side = 0;
            for (double v : diff) {                // with hysteresis, so jitter around zero doesn't count
                int sg = v > 0.5 * samp ? 1 : v < -0.5 * samp ? -1 : 0;
                if (sg && side && sg != side) crossings += 1;
                side = sg ? sg : side;
            }
            S.swing_corr = scorr;
            S.swing_amp = samp;
            S.swing_crossings = crossings;
            swinging = scorr > SWING_CORR && samp > SWING_AMP && crossings >= 2;
        }
    }
    // Only something that shows up in most recent frames blocks the 67.
    double dt = S.signing_t && now > *S.signing_t ? now - *S.signing_t : 0.0;
    S.signing_t = now;
    double signing_now = (!fists.empty() || !signing.empty() || covering) ? 1.0 : 0.0;
    S.signing_frac += (1 - std::exp(-dt / SIGN_TAU)) * (signing_now - S.signing_frac);
    const bool hands_ok = S.signing_frac < 0.5;
    bool bouncing = false;
    box.reset();
    if (wrists) {
        P2 lw = wrists->lw, rw = wrists->rw;
        bool low = !ref || std::min(lw.y, rw.y) > ref->nose.y + 0.5 * fw;   // below the chin
        bool apart = rw.x - lw.x > 0.8 * fw;                                 // not clapping
        bouncing = low && apart && corr > BOUNCE_CORR && amp > C("bounce_speed") && S.bounce_flips.size() >= 2;
        int r = int(0.3 * fw);
        box = box_from_corners(int(lw.x) - r, int(std::min(lw.y, rw.y)) - r, int(rw.x) + r, int(std::max(lw.y, rw.y)) + r);
    } else if (ref) {
        double x = ref->nose.x, y = ref->nose.y, r = ref->width;
        box = box_from_corners(int(x - 2 * r), int(y), int(x + 2 * r), int(y + 2 * r));
    }
    const bool sixty_seven = (bouncing || swinging) && hands_ok && !cinema && !mahoraga;
    score = std::max(std::max(corr, 0.0) * std::min(amp / C("bounce_speed"), 2.0) / 2,
                     std::max(S.swing_corr, 0.0) * std::min(S.swing_amp / SWING_AMP, 2.0) / 2);
    d["sixty_seven"] = Detection{sixty_seven, score, box};

    // ---- talking to a wall: hands flying around, or both open palms held out, while talking ----
    bool posed = covering > 0;
    for (const char* k : {"middle_finger", "thumbs_up", "nerd", "gojo", "absolute_cinema", "mahoraga", "sixty_seven",
                          "godfather", "looksmaxxing"})
        posed = posed || d[k].raw;
    const double floor_ = S.flying ? FLYING_KEEP : 1.0;
    S.flying = S.motion > floor_ * C("motion") || (S.hand_speed > floor_ * C("hand_speed") && S.motion > 0.03);
    const bool hands_flying = S.flying;
    {   // presenting: both open palms held out at chest height, one on each side of the face
        bool pres = false;
        if (open_hands.size() >= 2 && ref) {
            double x = ref->nose.x, y = ref->nose.y, rfw = ref->width;
            int left = open_hands[0], right = open_hands[0];
            for (int i : open_hands) {
                if (hands[i].pts[0][0] < hands[left].pts[0][0]) left = i;
                if (hands[i].pts[0][0] > hands[right].pts[0][0]) right = i;
            }
            pres = hands[left].pts[0][0] < x - PRESENT_SIDE * rfw && hands[right].pts[0][0] > x + PRESENT_SIDE * rfw;
            for (int i : {left, right}) {
                if (!pres) break;
                const Hand& h = hands[i];
                double dy = h.pts[0][1] - y;
                pres = PRESENT_LOW[0] * rfw < dy && dy < PRESENT_LOW[1] * rfw   // chest height
                       && h.pts[0][1] - h.pts[9][1] > 0.5 * h.size()             // held up, not resting
                       && speed_of(i) < C("hand_speed");
            }
        }
        if (pres) S.present_at = now;
    }
    const bool presenting = now - S.present_at < PRESENT_GAP;
    const bool talking = (hands_flying || presenting) && now - S.mouth_at < C("mouth_window");
    d["talking_to_wall"] = Detection{
        talking && !posed,
        std::max({S.motion / C("motion"), S.hand_speed / C("hand_speed"), 2.0 * double(presenting && talking)}) * 0.5,
        fbox};
    return d;
}

// ---- Trigger ----------------------------------------------------------------------------

Trigger::Trigger(double hold_, double min_show_, double linger_, double cooldown_)
    : hold(hold_), min_show(min_show_), linger(linger_), cooldown(cooldown_) {}

bool Trigger::update(bool raw, double now) {
    if (raw) {
        last_raw_ = now;
        if (!raw_since_) raw_since_ = now;
    } else {
        raw_since_.reset();
    }
    if (!started_) {
        if (raw && now >= cool_until_ && now - *raw_since_ >= hold) started_ = now;
    } else if (!raw && now - *started_ >= min_show && now - last_raw_ >= linger) {
        started_.reset();
        cool_until_ = now + cooldown;
    }
    return started_.has_value();
}

}  // namespace memecam
