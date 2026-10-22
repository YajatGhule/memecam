#include "mocap.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <set>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "tracking.h"

namespace memecam {

namespace {

const cv::Scalar FACE_COL(255, 220, 60), HAND_COL(90, 255, 90), POSE_COL(40, 160, 255), TEXT_COL(235, 235, 235);
constexpr int FONT = cv::FONT_HERSHEY_SIMPLEX;

using Edge = std::pair<int, int>;

// MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS.
const std::vector<Edge> FACE_EDGES = {
    {61, 146}, {146, 91}, {91, 181}, {181, 84}, {84, 17}, {17, 314}, {314, 405}, {405, 321}, {321, 375},
    {375, 291}, {61, 185}, {185, 40}, {40, 39}, {39, 37}, {37, 0}, {0, 267}, {267, 269}, {269, 270},
    {270, 409}, {409, 291}, {78, 95}, {95, 88}, {88, 178}, {178, 87}, {87, 14}, {14, 317}, {317, 402},
    {402, 318}, {318, 324}, {324, 308}, {78, 191}, {191, 80}, {80, 81}, {81, 82}, {82, 13}, {13, 312},
    {312, 311}, {311, 310}, {310, 415}, {415, 308}, {263, 249}, {249, 390}, {390, 373}, {373, 374},
    {374, 380}, {380, 381}, {381, 382}, {382, 362}, {263, 466}, {466, 388}, {388, 387}, {387, 386},
    {386, 385}, {385, 384}, {384, 398}, {398, 362}, {276, 283}, {283, 282}, {282, 295}, {295, 285},
    {300, 293}, {293, 334}, {334, 296}, {296, 336}, {33, 7}, {7, 163}, {163, 144}, {144, 145}, {145, 153},
    {153, 154}, {154, 155}, {155, 133}, {33, 246}, {246, 161}, {161, 160}, {160, 159}, {159, 158},
    {158, 157}, {157, 173}, {173, 133}, {46, 53}, {53, 52}, {52, 65}, {65, 55}, {70, 63}, {63, 105},
    {105, 66}, {66, 107}, {10, 338}, {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389},
    {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397}, {397, 365}, {365, 379},
    {379, 378}, {378, 400}, {400, 377}, {377, 152}, {152, 148}, {148, 176}, {176, 149}, {149, 150},
    {150, 136}, {136, 172}, {172, 58}, {58, 132}, {132, 93}, {93, 234}, {234, 127}, {127, 162}, {162, 21},
    {21, 54}, {54, 103}, {103, 67}, {67, 109}, {109, 10}};

// MediaPipe HandLandmarksConnections.HAND_CONNECTIONS.
const std::vector<Edge> HAND_EDGES = {{0, 1},   {1, 5},   {9, 13},  {13, 17}, {5, 9},   {0, 17},  {1, 2},
                                      {2, 3},   {3, 4},   {5, 6},   {6, 7},   {7, 8},   {9, 10},  {10, 11},
                                      {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}};

// PoseLandmarksConnections.POSE_LANDMARKS, body only (no face points < 11, no finger points 17..22).
const std::vector<Edge> POSE_EDGES = {{11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {11, 23},
                                      {12, 24}, {23, 24}, {23, 25}, {24, 26}, {25, 27}, {26, 28},
                                      {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32}};

const std::vector<std::pair<int, const char*>> POSE_LABELS = {{11, "L_SHO"}, {12, "R_SHO"}, {13, "L_ELB"},
                                                               {14, "R_ELB"}, {15, "L_WRI"}, {16, "R_WRI"},
                                                               {23, "L_HIP"}, {24, "R_HIP"}};
const std::vector<std::pair<int, const char*>> HAND_LABELS = {{0, "WRI"}, {4, "THB"},  {8, "IDX"},
                                                               {12, "MID"}, {16, "RNG"}, {20, "PNK"}};
// Python dict order: NOSE, CHIN, R_EYE, L_EYE.
const std::vector<std::pair<int, const char*>> FACE_LABELS = {{1, "NOSE"}, {152, "CHIN"}, {33, "R_EYE"},
                                                               {263, "L_EYE"}};

const std::vector<int>& face_dots() {
    static const std::vector<int> dots = [] {
        std::set<int> s;
        for (auto [a, b] : FACE_EDGES) s.insert(a), s.insert(b);
        return std::vector<int>(s.begin(), s.end());
    }();
    return dots;
}

bool is_hand_label(int i) {
    return std::any_of(HAND_LABELS.begin(), HAND_LABELS.end(), [i](const auto& l) { return l.first == i; });
}

cv::Point ip(double x, double y) { return {int(x), int(y)}; }   // numpy astype(int32): truncation
cv::Point ip(const Vec3& p) { return ip(p[0], p[1]); }
cv::Point ip(const PoseLm& p) { return ip(p.x, p.y); }

void tag(cv::Mat& img, cv::Point p, const std::string& text, cv::Scalar color) {
    cv::line(img, p, p + cv::Point(10, -10), color, 1, cv::LINE_AA);
    cv::putText(img, text, p + cv::Point(12, -12), FONT, 0.33, color, 1, cv::LINE_AA);
}

template <class Get>
void segments(cv::Mat& layer, const std::vector<Edge>& edges, Get get, cv::Scalar col, int thickness) {
    std::vector<std::vector<cv::Point>> segs;
    segs.reserve(edges.size());
    for (auto [a, b] : edges) segs.push_back({get(a), get(b)});
    if (!segs.empty()) cv::polylines(layer, segs, false, col, thickness, cv::LINE_AA);
}

}  // namespace

void brackets(cv::Mat& img, const Box& box, cv::Scalar color, const std::string& label, int pad, int arm, int t) {
    int x0 = box.x - pad, y0 = box.y - pad, x1 = box.x + box.width + pad, y1 = box.y + box.height + pad;
    const int corners[4][4] = {{x0, y0, 1, 1}, {x1, y0, -1, 1}, {x0, y1, 1, -1}, {x1, y1, -1, -1}};
    for (const auto& c : corners) {
        cv::line(img, {c[0], c[1]}, {c[0] + c[2] * arm, c[1]}, color, t, cv::LINE_AA);
        cv::line(img, {c[0], c[1]}, {c[0], c[1] + c[3] * arm}, color, t, cv::LINE_AA);
    }
    cv::rectangle(img, cv::Point(x0, y0), cv::Point(x1, y1), color, 1, cv::LINE_AA);
    if (!label.empty()) {
        int base = 0;
        cv::Size ts = cv::getTextSize(label, FONT, 0.45, 1, &base);
        cv::rectangle(img, cv::Point(x0, y0 - ts.height - 8), cv::Point(x0 + ts.width + 8, y0), color, -1);
        cv::putText(img, label, cv::Point(x0 + 4, y0 - 5), FONT, 0.45, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);
    }
}

void draw_markers(cv::Mat& img, const Frame& q) {
    cv::Mat layer = cv::Mat::zeros(img.size(), img.type());
    if (q.pose) {
        const Pose& pose = *q.pose;
        auto vis = [&](int i) { return pose[i].visibility > 0.5f; };
        std::vector<Edge> edges;
        for (auto e : POSE_EDGES)
            if (vis(e.first) && vis(e.second)) edges.push_back(e);
        segments(layer, edges, [&](int i) { return ip(pose[i]); }, POSE_COL, 2);
        for (int i = 11; i < 33; i++)
            if ((i < 17 || i > 22) && vis(i)) cv::circle(layer, ip(pose[i]), 5, POSE_COL, -1, cv::LINE_AA);
    }
    if (q.face && q.face->pts.size() >= 468) {
        const auto& pts = q.face->pts;
        segments(layer, FACE_EDGES, [&](int i) { return ip(pts[i]); }, FACE_COL, 1);
        for (int i : face_dots()) cv::circle(layer, ip(pts[i]), 2, FACE_COL, -1, cv::LINE_AA);
        for (int i : {1, 152, 33, 263, 10}) cv::circle(layer, ip(pts[i]), 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    }
    for (const Hand& hand : q.hands) {
        segments(layer, HAND_EDGES, [&](int i) { return ip(hand.pts[i]); }, HAND_COL, 2);
        for (int i = 0; i < 21; i++)
            cv::circle(layer, ip(hand.pts[i]), is_hand_label(i) ? 5 : 3, HAND_COL, -1, cv::LINE_AA);
    }

    // Glow: blur a quarter-size copy, double it there (img + glow + glow == img + 2*glow with saturation),
    // and upscale once, instead of two full-frame adds.
    cv::Mat small, glow, gray;
    cv::resize(layer, small, cv::Size(), 0.25, 0.25);
    cv::GaussianBlur(small, small, cv::Size(0, 0), 3);
    cv::add(small, small, small);
    cv::resize(small, glow, img.size());
    cv::add(img, glow, img);
    // Gray > 0 as the "something drawn here" mask.
    cv::cvtColor(layer, gray, cv::COLOR_BGR2GRAY);
    layer.copyTo(img, gray);

    // Labels and boxes are drawn crisp, without glow.
    if (q.pose) {
        for (auto [i, name] : POSE_LABELS)
            if ((*q.pose)[i].visibility > 0.5f) tag(img, ip((*q.pose)[i]), name, POSE_COL);
    }
    if (q.face && q.face->pts.size() >= 468) {
        brackets(img, q.face->bbox(), FACE_COL, "SUBJECT // TRACKED");
        for (auto [i, name] : FACE_LABELS) tag(img, ip(q.face->pts[i]), name, FACE_COL);
    } else if (auto box = head_box(q)) {
        brackets(img, *box, POSE_COL, "SUBJECT // POSE");
    }
    for (const Hand& hand : q.hands) {
        std::string side = hand.label == "Left" ? "L" : "R";
        brackets(img, hand.bbox(), HAND_COL, side + "_HAND " + hand.gesture, 10, 14);
        for (auto [i, name] : HAND_LABELS) tag(img, ip(hand.pts[i]), side + "_" + name, HAND_COL);
    }
}

void hud(cv::Mat& img, double fps, const std::vector<std::string>& active, bool recording, bool show_markers,
         std::optional<double> track_fps, bool show_memes, std::optional<bool> sound) {
    int h = img.rows;
    double secs = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    bool blink = long(secs * 2) % 2 == 0;
    if (recording && blink) cv::circle(img, {22, 24}, 7, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    char rate[96];
    std::snprintf(rate, sizeof rate, "MOCAP // %4.1f FPS", fps);
    std::string line = std::string(recording ? "REC  " : "LIVE ") + rate;
    if (track_fps) {
        char t[48];
        std::snprintf(t, sizeof t, " (tracking %4.1f)", *track_fps);
        line += t;
    }
    cv::putText(img, line, {36, 30}, FONT, 0.55, TEXT_COL, 1, cv::LINE_AA);
    if (!active.empty()) {
        std::string s = ">> ";
        for (size_t i = 0; i < active.size(); i++) {
            if (i) s += "  ";
            for (char c : active[i]) s += char(std::toupper(static_cast<unsigned char>(c)));
        }
        cv::putText(img, s, {36, 56}, FONT, 0.55, ACCENT_COL, 1, cv::LINE_AA);
    }
    std::string hint = std::string("[d] markers ") + (show_markers ? "on" : "off") + "  [f] memes " +
                       (show_memes ? "on" : "off") + "  ";
    if (sound) hint += std::string("[m] sound ") + (*sound ? "on" : "off") + "  ";
    hint += "[i] debug  [r] rec  [c] calibrate  [q] quit";
    cv::putText(img, hint, {12, h - 12}, FONT, 0.42, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
}

}  // namespace memecam
