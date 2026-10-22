// Core tracking types shared by every module (mirrors memecam/tracking.py: Hand, Face, Frame).
// Coordinates are double: the Python reference does all gesture math in numpy float64, and the logic
// parity tests replay its tracking results bit for bit.
#pragma once

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace memecam {

using Vec3 = cv::Vec3d;          // pixel x, y and relative z
using Box = cv::Rect;            // x0, y0, w, h  (Python bbox = (x0, y0, x1, y1) -> Box(x0, y0, x1-x0, y1-y0))

inline double norm2(double dx, double dy) { return std::sqrt(dx * dx + dy * dy); }
inline double dist2d(const Vec3& a, const Vec3& b) { return norm2(a[0] - b[0], a[1] - b[1]); }
inline cv::Point2d xy(const Vec3& p) { return {p[0], p[1]}; }

// Python-style bbox from points: int() of the min / max x, y (truncation towards zero).
Box bbox_of(const Vec3* pts, size_t n);
inline Box bbox_of(const std::vector<Vec3>& pts) { return bbox_of(pts.data(), pts.size()); }
inline Box box_from_corners(int x0, int y0, int x1, int y1) { return {x0, y0, x1 - x0, y1 - y0}; }

struct Hand {
    std::array<Vec3, 21> pts{};
    std::string label;           // "Left" / "Right" (user's hand, frame is mirrored)
    std::string gesture = "None";// MediaPipe canned gesture, e.g. "Victory", "Open_Palm"
    double gesture_score = 0.0;

    Box bbox() const { return bbox_of(pts.data(), pts.size()); }
    double size() const { return dist2d(pts[9], pts[0]) + 1e-6; }   // wrist -> middle MCP
};

struct Face {
    std::vector<Vec3> pts;       // 478 face-mesh points, pixels
    Box bbox() const { return bbox_of(pts); }
    double width() const { return dist2d(pts[234], pts[454]) + 1e-6; }
};

struct PoseLm { double x = 0, y = 0, z = 0, visibility = 0; };
using Pose = std::array<PoseLm, 33>;   // pixels + visibility (0..1)

constexpr int MOTION_W = 160, MOTION_H = 90;

// One person's tracking result. The top-level Frame returned by the tracker mirrors its primary
// (largest) person in hands/face/pose and lists everyone in `people`.
struct Frame {
    std::vector<Hand> hands;
    std::optional<Face> face;
    std::shared_ptr<const Pose> pose;          // shared: the async pose result is reused across frames;
                                               // pointer identity = Python id(pose) (same result as last frame)
    std::shared_ptr<const cv::Mat> motion;     // CV_8U MOTION_H x MOTION_W, nonzero = changed (shared)
    int width = 0, height = 0;
    int pid = 0;                               // persistent person id; 0 = not assigned
    std::vector<Frame> people;                 // top-level frame only
    std::map<int, int> renamed;                // ids renumbered this frame, old -> new (top-level only)
};

}  // namespace memecam
