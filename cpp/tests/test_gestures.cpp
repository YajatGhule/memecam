// Gesture logic tests on synthetic MediaPipe-style landmarks — port of tests/test_gestures.py.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <functional>

#include "builders.h"

using namespace mt;
using Catch::Approx;

namespace {

using Dets = std::map<std::string, Detection>;
const std::set<std::string> HAND_POSES{"middle_finger", "thumbs_up", "nerd"};
const std::vector<double> ANGLES{0, 45, -45, 90, -90, 30};

Hand thumbs_up_hand(cv::Point2d wrist = {640, 400}, double size = 110.0, const std::string& gesture = "None") {
    Hand h = make_hand(FIST, 0.0, wrist, size, "Right", gesture);
    double x = wrist.x, y = wrist.y;
    const double off[4][2] = {{-0.35, -0.25}, {-0.45, -0.55}, {-0.50, -0.85}, {-0.52, -1.10}};
    for (int k = 0; k < 4; ++k) {
        h.pts[1 + k][0] = x + off[k][0] * size;
        h.pts[1 + k][1] = y + off[k][1] * size;
    }
    return h;
}

std::vector<Hand> cinema_hands(double y = 380) {
    return {make_hand(OPEN, -15, {420, y}, 110, "Left"), make_hand(OPEN, 15, {860, y}, 110, "Right")};
}

std::vector<Hand> mahoraga_hands(cv::Point2d gap = {140, 0}, const Fingers& left_ext = FIST,
                                 const Fingers& right_ext = FIST, double y = 640) {
    return {make_hand(left_ext, 90, {560, y}, 110, "Left"), make_hand(right_ext, -90, {560 + gap.x, y + gap.y}, 110, "Right")};
}

Hand covering_hand(double dx = 0.0, const Fingers& ext = OPEN, const std::string& label = "Right") {
    return make_hand(ext, 0, {640 + dx, 470}, 110.0, label);
}

Hand crossed_hand(double angle = 0.0) {
    Hand h = make_hand(PEACE, angle);
    auto p7 = h.pts[7], p8 = h.pts[8];
    h.pts[7] = h.pts[11];
    h.pts[8] = h.pts[12];
    h.pts[11] = p7;
    h.pts[12] = p8;
    return h;
}

Dets neutral(Detectors& det, int n = 30, double ratio = 0.6, double t0 = 0.0) {
    Dets d;
    for (int i = 0; i < n; ++i) d = run(det, {}, make_face(ratio), t0 + i / 30.0);
    return d;
}

std::vector<Dets> wave(Detectors& det, const Fingers& ext = OPEN, int frames = 30, double step = 30.0, bool two = true,
                       std::pair<std::string, std::string> labels = {"Left", "Right"}, double t0 = 0.0,
                       double angle = 90.0) {
    std::vector<Dets> out;
    for (int i = 0; i < frames; ++i) {
        int phase = (i / 8) % 2;
        double dx = (i % 8) * step * (phase == 0 ? 1 : -1) + (phase == 0 ? 0 : 8 * step);
        std::vector<Hand> hands{make_hand(ext, angle, {350 + dx, 550}, 110, labels.first)};
        if (two) hands.push_back(make_hand(ext, angle, {930 - dx, 550 + dx / 3}, 110, labels.second));
        out.push_back(run(det, hands, talk_face(i), t0 + i / 30.0, std::min(0.3, step / 250)));
    }
    return out;
}

// pattern: (raw, seconds). Returns (t, active).
std::vector<std::pair<double, bool>> drive(Trigger& trig, std::vector<std::pair<bool, double>> pattern,
                                           double dt = 0.05, double t0 = 0.0) {
    std::vector<std::pair<double, bool>> out;
    int i = 0;
    for (auto [r, secs] : pattern) {
        long n = std::lround(secs / dt);
        for (long k = 0; k < n; ++k) {
            double t = std::round((t0 + i * dt) * 1e6) / 1e6;
            out.emplace_back(t, trig.update(r, t));
            ++i;
        }
    }
    return out;
}

std::optional<double> first_true(const std::vector<std::pair<double, bool>>& trace) {
    for (auto [t, a] : trace)
        if (a) return t;
    return std::nullopt;
}

// ---- 67 ----
const double SIX7_X[2] = {480.0, 800.0};
const double SIX7_Y = 560.0;

struct BounceOpts {
    double freq = 2.5, amp = 50.0;
    std::string mode = "alternate";
    double secs = 1.5;
    int fps = 30;
    double t0 = 0.0;
    bool pose = true, hands = true;
    Fingers ext = OPEN;
    double xl = SIX7_X[0], xr = SIX7_X[1], y = SIX7_Y;
    int repeat_pose = 1;
    bool talk = false;
    double motion = 0.0;
};

std::vector<Dets> bounce(Detectors& det, const BounceOpts& o) {
    std::vector<Dets> out;
    std::shared_ptr<const Pose> p;
    for (int i = 0; i < int(o.secs * o.fps); ++i) {
        double t = o.t0 + double(i) / o.fps;
        double s = o.amp * std::sin(2 * M_PI * o.freq * t);
        double xl = o.xl, xr = o.xr, y = o.y, yl = y, yr = y;
        if (o.mode == "alternate") { yl = y + s; yr = y - s; }
        else if (o.mode == "together") { yl = y + s; yr = y + s; }
        else if (o.mode == "left_only") { yl = y + s; yr = y; }
        if (o.mode == "clap") {
            double k = (1 - std::cos(2 * M_PI * o.freq * t)) / 2;
            double mid = (xl + xr) / 2;
            double nxl = xl + k * (mid - 15 - xl), nxr = xr + k * (mid + 15 - xr);
            xl = nxl;
            xr = nxr;
        }
        if (o.pose && (!p || i % o.repeat_pose == 0)) p = make_pose({xl, yl}, {xr, yr});
        std::vector<Hand> hs;
        if (o.hands) hs = {make_hand(o.ext, 0, {xl, yl}, 90, "Left"), make_hand(o.ext, 0, {xr, yr}, 90, "Right")};
        Frame fr;
        fr.hands = hs;
        fr.face = o.talk ? talk_face(i) : make_face();
        if (o.pose) fr.pose = p;
        fr.motion = motion_frac(o.motion);
        fr.width = 1280;
        fr.height = 720;
        out.push_back(det.run(fr, t));
    }
    return out;
}

bool fired(const std::vector<Dets>& res, const std::string& name = "sixty_seven", double hold = 0.4, int fps = 30) {
    Trigger tr(hold);
    bool any = false;
    for (size_t i = 0; i < res.size(); ++i) any = tr.update(raw(res[i], name), double(i) / fps) || any;
    return any;
}

std::shared_ptr<const cv::Mat> motion_blobs(const std::vector<cv::Point2d>& centers, double size = 120.0) {
    auto m = std::make_shared<cv::Mat>(cv::Mat::zeros(MOTION_H, MOTION_W, CV_8U));
    double sx = MOTION_W / 1280.0, sy = MOTION_H / 720.0;
    for (auto [x, y] : centers) {
        int r0 = std::max(0, int((y - size / 2) * sy)), r1 = std::min(MOTION_H, int((y + size / 2) * sy));
        int c0 = std::max(0, int((x - size / 2) * sx)), c1 = std::min(MOTION_W, int((x + size / 2) * sx));
        for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c) m->at<uint8_t>(r, c) = 1;
    }
    return m;
}

struct SwingOpts {
    std::string mode = "alternate";
    double freq = 2.0, amp = 100.0, secs = 1.5;
    int fps = 30;
    double t0 = 0.0;
    std::function<std::vector<Hand>(int)> hands;
    bool talk = false;
    double x0 = 390.0, x1 = 890.0, y = 560.0;
};

std::vector<Dets> swing(Detectors& det, const SwingOpts& o) {
    std::vector<Dets> out;
    for (int i = 0; i < int(o.secs * o.fps); ++i) {
        double t = o.t0 + double(i) / o.fps;
        double s = o.amp * std::sin(2 * M_PI * o.freq * t);
        std::vector<cv::Point2d> blobs;
        if (o.mode == "alternate") blobs = {{o.x0, o.y + s}, {o.x1, o.y - s}};
        else if (o.mode == "together") blobs = {{o.x0, o.y + s}, {o.x1, o.y + s}};
        else if (o.mode == "one_side") blobs = {{o.x0, o.y + s}};
        else blobs = {{o.x0, o.y}, {o.x1, o.y}};
        Frame fr;
        if (o.hands) fr.hands = o.hands(i);
        fr.face = o.talk ? talk_face(i) : make_face();
        fr.motion = motion_blobs(blobs);
        fr.width = 1280;
        fr.height = 720;
        out.push_back(det.run(fr, t));
    }
    return out;
}

Hand pinched_hand(cv::Point2d wrist = {820, 520}, double size = 100.0) {
    Hand h = make_hand(OPEN, 0, wrist, size);
    cv::Point2d tip{wrist.x + 0.05 * size, wrist.y - 1.6 * size};
    const int ks[5] = {4, 8, 12, 16, 20};
    const double off[5][2] = {{-0.08, 0.05}, {0.0, -0.06}, {0.06, -0.04}, {0.1, 0.02}, {0.07, 0.08}};
    for (int j = 0; j < 5; ++j) {
        h.pts[ks[j]][0] = tip.x + off[j][0] * size;
        h.pts[ks[j]][1] = tip.y + off[j][1] * size;
    }
    for (int k : {3, 7, 11, 15, 19}) {
        h.pts[k][0] = h.pts[k + 1][0] + 0.0;
        h.pts[k][1] = h.pts[k + 1][1] + 0.2 * size;
    }
    return h;
}

Hand jaw_hand(const Face& face) {
    cv::Point2d jaw = xy(face.pts[150]);
    Hand h = make_hand(INDEX, 60, {jaw.x - 120, jaw.y + 90}, 90);
    double sx = jaw.x - h.pts[8][0], sy = jaw.y - h.pts[8][1];
    for (auto& p : h.pts) { p[0] += sx; p[1] += sy; }
    return h;
}

const std::map<std::string, double> V2{{"extra_gestures", 1.0}};

}  // namespace

// ---------------------------------------------------------------------------------------
// finger_states
// ---------------------------------------------------------------------------------------

TEST_CASE("finger_states rotation invariant", "[gestures]") {
    auto angle = GENERATE(0.0, 45.0, -45.0, 90.0, -90.0, 30.0, 180.0);
    auto ext = GENERATE(OPEN, FIST, PEACE, MIDDLE, INDEX);
    FingerStates s = finger_states(make_hand(ext, angle), RATIO);
    CAPTURE(angle, ext.size());
    CHECK(s.index == bool(ext.count("index")));
    CHECK(s.middle == bool(ext.count("middle")));
    CHECK(s.ring == bool(ext.count("ring")));
    CHECK(s.pinky == bool(ext.count("pinky")));
}

TEST_CASE("finger_states scale invariant", "[gestures]") {
    auto size = GENERATE(40.0, 110.0, 300.0);
    FingerStates s = finger_states(make_hand(PEACE, 0, {640, 400}, size), RATIO);
    CHECK((s.index && s.middle && !s.ring && !s.pinky));
}

TEST_CASE("extended ratio has margin", "[gestures]") {
    for (const auto& ext : {OPEN, FIST}) {
        Hand h = make_hand(ext);
        for (auto [pip, tip] : {std::pair{6, 8}, {10, 12}, {14, 16}, {18, 20}}) {
            double r = cv::norm(h.pts[tip] - h.pts[0]) / cv::norm(h.pts[pip] - h.pts[0]);
            if (!ext.empty()) CHECK(r > RATIO + 0.1);
            else CHECK(r < RATIO - 0.3);
        }
    }
}

// ---------------------------------------------------------------------------------------
// single-hand pose exclusivity
// ---------------------------------------------------------------------------------------

TEST_CASE("middle finger only", "[gestures]") {
    auto angle = GENERATE(0.0, 20.0, -20.0);
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(MIDDLE, angle)})), HAND_POSES) == std::set<std::string>{"middle_finger"});
}

TEST_CASE("v sign fires nothing", "[gestures]") {
    auto angle = GENERATE(0.0, 30.0, -30.0, 90.0);
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(PEACE, angle)})), HAND_POSES).empty());
}

TEST_CASE("thumbs up from geometry", "[gestures]") {
    Detectors det;
    auto d = run(det, {thumbs_up_hand()});
    CHECK(intersect(raws(d), HAND_POSES) == std::set<std::string>{"thumbs_up"});
    CHECK_FALSE(d["mahoraga"].raw);
}

TEST_CASE("thumbs up from mediapipe label", "[gestures]") {
    Detectors det;
    CHECK(run(det, {make_hand(FIST, 0, {640, 400}, 110, "Right", "Thumb_Up")})["thumbs_up"].raw);
}

TEST_CASE("plain fist is not thumbs up", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, {make_hand(FIST)})["thumbs_up"].raw);
}

TEST_CASE("fist label vetoes geometric thumbs up", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, {thumbs_up_hand({640, 400}, 110, "Closed_Fist")})["thumbs_up"].raw);
}

TEST_CASE("index up is nerd", "[gestures]") {
    auto angle = GENERATE(0.0, 30.0, -30.0, 45.0);
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(INDEX, angle)})), HAND_POSES) == std::set<std::string>{"nerd"});
}

TEST_CASE("index pointing down is not nerd", "[gestures]") {
    auto angle = GENERATE(180.0, 150.0, -150.0, 135.0);
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(INDEX, angle)})), HAND_POSES).empty());
}

TEST_CASE("index pointing sideways is not nerd", "[gestures]") {
    auto angle = GENERATE(90.0, -90.0, 80.0, -80.0);
    Detectors det;
    CHECK_FALSE(raws(run(det, {make_hand(INDEX, angle)})).count("nerd"));
}

TEST_CASE("open palm and fist trigger nothing", "[gestures]") {
    for (const auto& ext : {OPEN, FIST}) {
        Detectors det;
        CHECK(raws(run(det, {make_hand(ext)})).empty());
    }
}

TEST_CASE("victory gesture label is peace not nerd", "[gestures]") {
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(INDEX, 0, {640, 400}, 110, "Right", "Victory")})), HAND_POSES).size() <= 1);
}

TEST_CASE("pointing up label with peace fingers is one pose", "[gestures]") {
    Detectors det;
    CHECK(intersect(raws(run(det, {make_hand(PEACE, 0, {640, 400}, 110, "Right", "Pointing_Up")})), HAND_POSES).size() <= 1);
}

// ---------------------------------------------------------------------------------------
// two-hand poses
// ---------------------------------------------------------------------------------------

TEST_CASE("absolute cinema above chin", "[gestures]") {
    Detectors det;
    auto d = run(det, cinema_hands(), make_face());
    CHECK(d["absolute_cinema"].raw);
    CHECK_FALSE(d["mahoraga"].raw);
}

TEST_CASE("absolute cinema below chin is false", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, cinema_hands(650), make_face())["absolute_cinema"].raw);
}

TEST_CASE("absolute cinema needs hands outside face", "[gestures]") {
    Detectors det;
    std::vector<Hand> hands{make_hand(OPEN, 0, {600, 380}, 110, "Left"), make_hand(OPEN, 0, {860, 380})};
    CHECK_FALSE(run(det, hands, make_face())["absolute_cinema"].raw);
}

TEST_CASE("absolute cinema needs open palms", "[gestures]") {
    Detectors det;
    std::vector<Hand> hands{make_hand(FIST, 0, {420, 380}, 110, "Left"), make_hand(OPEN, 0, {860, 380})};
    CHECK_FALSE(run(det, hands, make_face())["absolute_cinema"].raw);
}

TEST_CASE("mahoraga two fists", "[gestures]") {
    Detectors det;
    auto d = run(det, mahoraga_hands(), make_face());
    CHECK(d["mahoraga"].raw);
    CHECK_FALSE(d["absolute_cinema"].raw);
}

TEST_CASE("mahoraga without face", "[gestures]") {
    Detectors det;
    CHECK(run(det, mahoraga_hands())["mahoraga"].raw);
}

TEST_CASE("mahoraga far apart is false", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, mahoraga_hands({600, 0}), make_face())["mahoraga"].raw);
}

TEST_CASE("mahoraga not during cinema", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, cinema_hands(), make_face())["mahoraga"].raw);
}

TEST_CASE("mahoraga needs two fists", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, mahoraga_hands({140, 0}, FIST, OPEN), make_face())["mahoraga"].raw);
}

TEST_CASE("mahoraga not at chin", "[gestures]") {
    Detectors det;
    CHECK_FALSE(run(det, mahoraga_hands({140, 0}, FIST, FIST, 460), make_face())["mahoraga"].raw);
}

// ---------------------------------------------------------------------------------------
// hands over the face: scuba cat (one hand), higuruma (both)
// ---------------------------------------------------------------------------------------

TEST_CASE("scuba cat one hand over mouth", "[gestures]") {
    Detectors det;
    Dets last;
    for (int i = 0; i < 10; ++i) last = run(det, {covering_hand()}, make_face(), i / 30.0, 0.05);
    CHECK(raws(last) == std::set<std::string>{"scuba_cat"});
}

TEST_CASE("scuba cat other hand waving is not wall", "[gestures]") {
    Detectors det;
    std::vector<Dets> res;
    for (int i = 0; i < 30; ++i) {
        Hand wav = make_hand(OPEN, 0, {150.0 + (i % 8) * 40, 520}, 110, "Left");
        res.push_back(run(det, {covering_hand(), wav}, make_face(), i / 30.0));
    }
    for (size_t i = 5; i < res.size(); ++i) CHECK(res[i]["scuba_cat"].raw);
    for (auto& r : res) CHECK_FALSE(r["talking_to_wall"].raw);
}

TEST_CASE("higuruma both hands over face", "[gestures]") {
    Detectors det;
    auto d = run(det, {covering_hand(-30, OPEN, "Left"), covering_hand(30)}, make_face());
    CHECK(d["higuruma"].raw);
    CHECK_FALSE(d["scuba_cat"].raw);
}

TEST_CASE("hand far from face is not covering", "[gestures]") {
    Detectors det;
    auto d = run(det, {make_hand(OPEN, 0, {200, 650})}, make_face());
    CHECK_FALSE(d["scuba_cat"].raw);
    CHECK_FALSE(d["higuruma"].raw);
}

TEST_CASE("finger sign by the face is not covering", "[gestures]") {
    Detectors det;
    auto d = run(det, {make_hand(INDEX, 0, {640, 470})}, make_face());
    CHECK_FALSE(d["scuba_cat"].raw);
    CHECK_FALSE(d["higuruma"].raw);
    CHECK_FALSE(d["nerd"].raw);
}

TEST_CASE("nerd beside face", "[gestures]") {
    Detectors det;
    CHECK(raws(run(det, {make_hand(INDEX, 0, {820, 420})}, make_face())) == std::set<std::string>{"nerd"});
}

TEST_CASE("one hand resting on face still is not higuruma", "[gestures]") {
    Detectors det;
    for (int i = 0; i < 90; ++i) CHECK_FALSE(run(det, {covering_hand()}, make_face(), i / 30.0)["higuruma"].raw);
}

TEST_CASE("scuba never switches to higuruma", "[gestures]") {
    Detectors det;
    std::vector<Dets> res;
    for (int i = 0; i < 10; ++i) res.push_back(run(det, {covering_hand()}, make_face(), i / 30.0, 0.05));
    for (int i = 0; i < 80; ++i) res.push_back(run(det, {covering_hand()}, make_face(), (10 + i) / 30.0));
    CHECK(res[9]["scuba_cat"].raw);
    for (auto& r : res) CHECK_FALSE(r["higuruma"].raw);
}

// ---------------------------------------------------------------------------------------
// gojo: crossed index + middle; dafoe
// ---------------------------------------------------------------------------------------

TEST_CASE("gojo crossed fingers", "[gestures]") {
    auto angle = GENERATE(0.0, 20.0, -20.0);
    Detectors det;
    auto d = run(det, {crossed_hand(angle)}, make_face());
    CHECK(d["gojo"].raw);
    CHECK_FALSE(d["nerd"].raw);
}

TEST_CASE("v sign is not gojo", "[gestures]") {
    Detectors det;
    auto d = run(det, {make_hand(PEACE)}, make_face());
    CHECK_FALSE(d["gojo"].raw);
    CHECK_FALSE(d["nerd"].raw);
}

TEST_CASE("dafoe head tilt", "[gestures]") {
    Detectors det;
    CHECK_FALSE(neutral(det)["dafoe"].raw);
    CHECK(run(det, {}, make_face(0.45), 1.1)["dafoe"].raw);
}

TEST_CASE("dafoe small jitter does not fire", "[gestures]") {
    Detectors det;
    neutral(det);
    int i = 0;
    for (double r : {0.59, 0.61, 0.585, 0.6, 0.575}) CHECK_FALSE(run(det, {}, make_face(r), 2 + (i++) / 30.0)["dafoe"].raw);
}

TEST_CASE("dafoe held tilt stays on for a while", "[gestures]") {
    Detectors det;
    neutral(det);
    for (int i = 0; i < 60; ++i) CHECK(run(det, {}, make_face(0.45), 1 + i / 30.0)["dafoe"].raw);
}

TEST_CASE("dafoe not after looking down", "[gestures]") {
    Detectors det;
    neutral(det);
    neutral(det, 150, 0.7, 1);
    CHECK_FALSE(neutral(det, 15, 0.6, 6)["dafoe"].raw);
}

TEST_CASE("dafoe recovers from bad initial baseline", "[gestures]") {
    Detectors det;
    run(det, {}, make_face(0.7), 0);
    CHECK_FALSE(neutral(det, 300, 0.6, 0.1)["dafoe"].raw);
}

TEST_CASE("dafoe calibrate resets", "[gestures]") {
    Detectors det;
    neutral(det);
    det.calibrate();
    CHECK_FALSE(run(det, {}, make_face(0.5), 2)["dafoe"].raw);
}

// ---------------------------------------------------------------------------------------
// talking to a wall
// ---------------------------------------------------------------------------------------

TEST_CASE("talking to wall fast hands", "[gestures]") {
    Detectors det;
    auto res = wave(det);
    CHECK_FALSE(res[0]["talking_to_wall"].raw);
    CHECK(res.back()["talking_to_wall"].raw);
    int n = 0;
    for (size_t i = 10; i < res.size(); ++i) n += res[i]["talking_to_wall"].raw;
    CHECK(n >= 14);
}

TEST_CASE("talking to wall one hand", "[gestures]") {
    Detectors det;
    CHECK(wave(det, OPEN, 30, 30.0, false).back()["talking_to_wall"].raw);
}

TEST_CASE("talking to wall stationary", "[gestures]") {
    Detectors det;
    for (auto& r : wave(det, OPEN, 30, 0.0)) CHECK_FALSE(r["talking_to_wall"].raw);
}

TEST_CASE("talking to wall slow drift", "[gestures]") {
    Detectors det;
    for (auto& r : wave(det, OPEN, 30, 3.0)) CHECK_FALSE(r["talking_to_wall"].raw);
}

TEST_CASE("talking to wall suppressed by middle finger", "[gestures]") {
    Detectors det;
    auto res = wave(det, MIDDLE, 30, 30.0, false);
    for (auto& r : res) {
        CHECK(r["middle_finger"].raw);
        CHECK_FALSE(r["talking_to_wall"].raw);
    }
}

TEST_CASE("talking to wall duplicate handedness labels", "[gestures]") {
    Detectors det;
    for (auto& r : wave(det, OPEN, 30, 0.0, true, {"Right", "Right"})) CHECK_FALSE(r["talking_to_wall"].raw);
}

TEST_CASE("talking to wall handedness flip", "[gestures]") {
    Detectors det;
    for (int i = 0; i < 30; ++i) {
        std::string a = i % 2 ? "Left" : "Right", b = i % 2 ? "Right" : "Left";
        std::vector<Hand> hands{make_hand(OPEN, 0, {350, 550}, 110, a), make_hand(OPEN, 0, {930, 550}, 110, b)};
        CHECK_FALSE(run(det, hands, make_face(), i / 30.0)["talking_to_wall"].raw);
    }
}

TEST_CASE("talking to wall decays after stopping", "[gestures]") {
    Detectors det;
    wave(det);
    CHECK_FALSE(wave(det, OPEN, 30, 0.0, true, {"Left", "Right"}, 1.0).back()["talking_to_wall"].raw);
}

TEST_CASE("talking to wall palms out while talking", "[gestures]") {
    Detectors det;
    auto res = wave(det, OPEN, 30, 0.0, true, {"Left", "Right"}, 0.0, 0.0);
    CHECK_FALSE(res[0]["talking_to_wall"].raw);
    for (size_t i = 3; i < res.size(); ++i) CHECK(res[i]["talking_to_wall"].raw);
    Trigger tr(0.5);
    bool any = false;
    for (size_t i = 0; i < res.size(); ++i) any = tr.update(res[i]["talking_to_wall"].raw, i / 30.0) || any;
    CHECK(any);
}

TEST_CASE("palms out needs talking", "[gestures]") {
    Detectors det;
    std::vector<Hand> hands{make_hand(OPEN, 0, {350, 550}, 110, "Left"), make_hand(OPEN, 0, {930, 550})};
    for (int i = 0; i < 30; ++i) CHECK_FALSE(run(det, hands, make_face(), i / 30.0)["talking_to_wall"].raw);
}

TEST_CASE("palms out negatives", "[gestures]") {
    struct Kw { const char* name; double x0 = 350, x1 = 930, y = 550, angle = 0.0; bool two = true; Fingers ext = OPEN; };
    auto kw = GENERATE(Kw{"xs", 560, 720}, Kw{"y", 350, 930, 300}, Kw{"angle", 350, 930, 550, 90},
                       Kw{"two", 350, 930, 550, 0.0, false}, Kw{"ext", 350, 930, 550, 0.0, true, FIST});
    CAPTURE(kw.name);
    Detectors det;
    for (int i = 0; i < 30; ++i) {
        std::vector<Hand> hands{make_hand(kw.ext, kw.angle, {kw.x0, kw.y}, 110, "Left")};
        if (kw.two) hands.push_back(make_hand(kw.ext, -kw.angle, {kw.x1, kw.y}));
        CHECK_FALSE(run(det, hands, talk_face(i), i / 30.0)["talking_to_wall"].raw);
    }
}

TEST_CASE("talking to wall hysteresis", "[gestures]") {
    Detectors det;
    Dets d;
    for (int i = 0; i < 20; ++i) d = run(det, {}, talk_face(i), i / 30.0, 0.12);
    CHECK(d["talking_to_wall"].raw);
    for (int i = 0; i < 20; ++i) CHECK(run(det, {}, talk_face(i), (20 + i) / 30.0, 0.05)["talking_to_wall"].raw);
    for (int i = 0; i < 20; ++i) d = run(det, {}, talk_face(i), (40 + i) / 30.0, 0.0);
    CHECK_FALSE(d["talking_to_wall"].raw);
    Detectors det2;
    for (int i = 0; i < 40; ++i) CHECK_FALSE(run(det2, {}, talk_face(i), i / 30.0, 0.05)["talking_to_wall"].raw);
}

TEST_CASE("talking to wall ignores motion beside someone else", "[gestures]") {
    Face face = make_face(0.6, {250.0, 300.0}, 150.0, 195.0);
    auto m = std::make_shared<cv::Mat>(cv::Mat::zeros(MOTION_H, MOTION_W, CV_8U));
    (*m)(cv::Range::all(), cv::Range(int(900 * MOTION_W / 1280), MOTION_W)).setTo(1);
    Detectors det;
    for (int i = 0; i < 30; ++i) {
        Frame fr;
        fr.face = make_face(0.6, {250.0, 300.0}, 150.0, 195.0, 0.02 + 0.06 * (i % 2));
        fr.motion = m;
        fr.width = 1280;
        fr.height = 720;
        CHECK_FALSE(det.run(fr, i / 30.0)["talking_to_wall"].raw);
    }
    Frame fr;
    fr.face = face;
    fr.motion = m;
    fr.width = 1280;
    fr.height = 720;
    CHECK(det.motion_level(fr) == 0.0);
}

TEST_CASE("talking to wall hand entering frame", "[gestures]") {
    Detectors det;
    for (int i = 0; i < 10; ++i) run(det, {make_hand(OPEN, 0, {350, 550})}, make_face(), i / 30.0);
    for (int i = 10; i < 20; ++i) {
        std::vector<Hand> hands{make_hand(OPEN, 0, {350, 550}), make_hand(OPEN, 0, {930, 550})};
        CHECK(run(det, hands, make_face(), i / 30.0)["talking_to_wall"].score < 0.2);
    }
}

// ---------------------------------------------------------------------------------------
// Trigger
// ---------------------------------------------------------------------------------------

TEST_CASE("trigger needs hold", "[trigger]") {
    Trigger tr(0.3, 2.0, 0.6, 0.8);
    for (auto [t, a] : drive(tr, {{true, 0.2}, {false, 0.1}, {true, 0.2}})) CHECK_FALSE(a);
    Trigger tr2(0.3);
    auto t = first_true(drive(tr2, {{true, 1.0}}));
    REQUIRE(t);
    CHECK(*t == Approx(0.3));
}

TEST_CASE("trigger min show", "[trigger]") {
    Trigger tr(0.3, 2.0, 0.6, 0.8);
    std::vector<double> on;
    for (auto [t, a] : drive(tr, {{true, 0.5}, {false, 3.0}}))
        if (a) on.push_back(t);
    REQUIRE(!on.empty());
    CHECK(on.front() == Approx(0.3));
    CHECK(on.back() >= 2.25 - 1e-9);
    CHECK(on.back() <= 2.3 + 1e-9);
    CHECK_FALSE(tr.active());
}

TEST_CASE("trigger linger bridges dropouts", "[trigger]") {
    Trigger tr(0.3, 1.0, 0.6, 0.8);
    for (auto [t, a] : drive(tr, {{true, 0.5}, {true, 1.0}, {false, 0.3}, {true, 0.5}, {false, 0.3}}))
        if (t >= 0.3) CHECK(a);
    std::vector<double> on;
    for (auto [t, a] : drive(tr, {{false, 1.0}}, 0.05, 2.6))
        if (a) on.push_back(t);
    REQUIRE(!on.empty());
    CHECK(on.back() < 2.3 + 0.6 + 0.05 + 1e-9);
    CHECK_FALSE(tr.active());
}

TEST_CASE("trigger cooldown", "[trigger]") {
    Trigger tr(0.3, 1.0, 0.2, 0.8);
    drive(tr, {{true, 0.5}, {false, 1.0}});
    CHECK_FALSE(tr.active());
    auto t = first_true(drive(tr, {{true, 2.0}}, 0.05, 1.5));
    REQUIRE(t);
    CHECK(*t >= 1.3 + 0.8 - 1e-9);
}

TEST_CASE("trigger stays while raw", "[trigger]") {
    Trigger tr(0.3, 1.0, 0.2, 0.8);
    for (auto [t, a] : drive(tr, {{true, 5.0}}))
        if (t >= 0.3) CHECK(a);
}

// ---------------------------------------------------------------------------------------
// 67: open palms at chest height bouncing up and down in alternation
// ---------------------------------------------------------------------------------------

TEST_CASE("sixty seven alternating bounce", "[67]") {
    auto freq = GENERATE(2.0, 2.5, 3.0);
    Detectors det;
    BounceOpts o;
    o.freq = freq;
    auto res = bounce(det, o);
    CHECK(fired(res));
    CHECK(res.back()["sixty_seven"].raw);
    CHECK_FALSE(res[0]["sixty_seven"].raw);
}

TEST_CASE("sixty seven pose only", "[67]") {
    Detectors det;
    BounceOpts o;
    o.hands = false;
    CHECK(fired(bounce(det, o)));
}

TEST_CASE("sixty seven hands only", "[67]") {
    Detectors det;
    BounceOpts o;
    o.pose = false;
    CHECK(fired(bounce(det, o)));
}

TEST_CASE("sixty seven async pose", "[67]") {
    for (int rep : {2, 3}) {
        Detectors det;
        BounceOpts o;
        o.hands = false;
        o.repeat_pose = rep;
        CHECK(fired(bounce(det, o)));
    }
}

TEST_CASE("sixty seven suppresses other memes", "[67]") {
    Detectors det;
    BounceOpts o;
    o.talk = true;
    o.motion = 0.15;
    o.secs = 2.0;
    auto res = bounce(det, o);
    CHECK(fired(res));
    CHECK_FALSE(fired(res, "talking_to_wall", 0.5));
    size_t on = 0;
    while (on < res.size() && !res[on]["sixty_seven"].raw) ++on;
    REQUIRE(on < res.size());
    for (size_t i = on; i < res.size(); ++i) CHECK_FALSE(res[i]["talking_to_wall"].raw);
    for (const char* k : {"higuruma", "scuba_cat", "mahoraga", "absolute_cinema"})
        for (auto& r : res) CHECK_FALSE(r[k].raw);
}

TEST_CASE("sixty seven negatives", "[67]") {
    struct Kw { const char* name; std::function<void(BounceOpts&)> set; };
    auto kw = GENERATE(
        Kw{"amp0", [](BounceOpts& o) { o.amp = 0.0; }}, Kw{"together", [](BounceOpts& o) { o.mode = "together"; }},
        Kw{"left_only", [](BounceOpts& o) { o.mode = "left_only"; }},
        Kw{"clap amp0", [](BounceOpts& o) { o.mode = "clap"; o.amp = 0.0; }},
        Kw{"clap f2", [](BounceOpts& o) { o.mode = "clap"; o.freq = 2.0; }},
        Kw{"amp4", [](BounceOpts& o) { o.amp = 4.0; }},
        Kw{"slow", [](BounceOpts& o) { o.freq = 0.5; o.amp = 40.0; }},
        Kw{"fist", [](BounceOpts& o) { o.ext = FIST; }}, Kw{"peace", [](BounceOpts& o) { o.ext = PEACE; }},
        Kw{"face height", [](BounceOpts& o) { o.y = 300.0; }},
        Kw{"together middle", [](BounceOpts& o) { o.xl = 620.0; o.xr = 660.0; }});
    CAPTURE(kw.name);
    Detectors det;
    BounceOpts o;
    kw.set(o);
    CHECK_FALSE(fired(bounce(det, o)));
}

TEST_CASE("sixty seven negatives pose only", "[67]") {
    auto which = GENERATE(0, 1, 2);
    Detectors det;
    BounceOpts o;
    o.hands = false;
    if (which == 0) o.mode = "together";
    else if (which == 1) o.mode = "left_only";
    else o.amp = 0.0;
    CHECK_FALSE(fired(bounce(det, o)));
}

TEST_CASE("sixty seven stops after bouncing", "[67]") {
    Detectors det;
    bounce(det, {});
    BounceOpts o;
    o.amp = 0.0;
    o.t0 = 1.5;
    CHECK_FALSE(bounce(det, o).back()["sixty_seven"].raw);
}

TEST_CASE("sixty seven from motion", "[67]") {
    auto freq = GENERATE(1.5, 2.0, 3.0);
    Detectors det;
    SwingOpts o;
    o.freq = freq;
    auto res = swing(det, o);
    CHECK(fired(res));
    CHECK_FALSE(res[0]["sixty_seven"].raw);
}

TEST_CASE("sixty seven from motion negatives", "[67]") {
    auto which = GENERATE(0, 1, 2, 3, 4, 5);
    Detectors det;
    SwingOpts o;
    switch (which) {
        case 0: o.mode = "together"; break;
        case 1: o.mode = "one_side"; break;
        case 2: o.mode = "still"; break;
        case 3: o.amp = 15.0; break;
        case 4: o.freq = 0.4; break;
        default: o.x0 = 150.0; o.x1 = 1130.0;
    }
    CAPTURE(which);
    CHECK_FALSE(fired(swing(det, o)));
}

TEST_CASE("sixty seven from motion blocked by fists", "[67]") {
    Detectors det;
    SwingOpts o;
    o.hands = [](int) {
        return std::vector<Hand>{make_hand(FIST, 0, {390, 560}, 110, "Left"), make_hand(FIST, 0, {890, 560})};
    };
    CHECK_FALSE(fired(swing(det, o)));
}

TEST_CASE("sixty seven from motion tolerates misread frames", "[67]") {
    Detectors det;
    SwingOpts o;
    o.hands = [](int i) {
        if (i % 6 == 0) return std::vector<Hand>{make_hand(FIST, 0, {390, 560}, 110, "Left")};
        if (i % 6 == 3) return std::vector<Hand>{make_hand(INDEX, 0, {890, 560})};
        return std::vector<Hand>{make_hand(OPEN, 0, {390, 560}, 110, "Left")};
    };
    CHECK(fired(swing(det, o)));
}

TEST_CASE("sixty seven from motion suppresses talking", "[67]") {
    Detectors det;
    SwingOpts o;
    o.talk = true;
    o.secs = 2.0;
    auto res = swing(det, o);
    CHECK(fired(res));
    size_t on = 0;
    while (on < res.size() && !res[on]["sixty_seven"].raw) ++on;
    REQUIRE(on < res.size());
    for (size_t i = on; i < res.size(); ++i) CHECK_FALSE(res[i]["talking_to_wall"].raw);
}

TEST_CASE("sixty seven from motion stops", "[67]") {
    Detectors det;
    swing(det, {});
    SwingOpts o;
    o.mode = "still";
    o.t0 = 1.5;
    CHECK_FALSE(swing(det, o).back()["sixty_seven"].raw);
}

// ---------------------------------------------------------------------------------------
// v2 gestures: godfather 🤌 and looksmaxxing (only with extra_gestures on)
// ---------------------------------------------------------------------------------------

TEST_CASE("godfather pinched fingers", "[v2]") {
    Detectors det(V2);
    auto d = run(det, {pinched_hand()}, make_face());
    CHECK(d["godfather"].raw);
    CHECK_FALSE((d["thumbs_up"].raw || d["nerd"].raw || d["middle_finger"].raw || d["gojo"].raw));
}

TEST_CASE("godfather off in v1", "[v2]") {
    Detectors det;
    CHECK_FALSE(run(det, {pinched_hand()}, make_face())["godfather"].raw);
}

TEST_CASE("fist and open hand are not godfather", "[v2]") {
    for (const Hand& h : {make_hand(FIST, 0, {820, 520}), make_hand(OPEN, 0, {820, 520}), thumbs_up_hand()}) {
        Detectors det(V2);
        CHECK_FALSE(run(det, {h}, make_face())["godfather"].raw);
    }
}

TEST_CASE("looksmaxxing finger on jaw", "[v2]") {
    Face face = make_face();
    face.pts[150][0] = FACE_C.x - 0.4 * FACE_W;
    face.pts[150][1] = FACE_C.y + 0.32 * FACE_H;
    Detectors det(V2);
    auto d = run(det, {jaw_hand(face)}, face);
    CHECK(d["looksmaxxing"].raw);
    CHECK_FALSE((d["nerd"].raw || d["scuba_cat"].raw || d["higuruma"].raw));
}

TEST_CASE("chin on fist is not looksmaxxing", "[v2]") {
    Face face = make_face();
    Hand fist = make_hand(FIST, 0, {FACE_C.x, FACE_C.y + 0.75 * FACE_H}, 100);
    Detectors det(V2);
    CHECK_FALSE(run(det, {fist}, face)["looksmaxxing"].raw);
}

TEST_CASE("nerd beside face is not looksmaxxing", "[v2]") {
    Detectors det(V2);
    auto d = run(det, {make_hand(INDEX, 0, {820, 420})}, make_face());
    CHECK(d["nerd"].raw);
    CHECK_FALSE(d["looksmaxxing"].raw);
}
