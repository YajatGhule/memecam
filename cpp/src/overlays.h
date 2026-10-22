// GIF/PNG playback and compositing onto the webcam frame (port of memecam/overlays.py).
#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "types.h"

namespace memecam {

// Python boxes are (x0, y0, x1, y1); ours are cv::Rect. Helpers for the far corner.
inline int x1_of(const Box& b) { return b.x + b.width; }
inline int y1_of(const Box& b) { return b.y + b.height; }

// Animated GIF or still image, decoded once into BGRA frames.
// `trim` = (first, last) frame indices to keep. With `loop_from`, the clip plays once and then rocks back
// and forth between that frame and the end instead of restarting from the top.
class Media {
public:
    Media(const std::string& path, int max_side = 480, bool key_black = false, const std::string& label = "",
          std::optional<std::pair<int, int>> trim = std::nullopt, std::optional<int> loop_from = std::nullopt);
    const cv::Mat& at(double t) const;          // frame shown `t` seconds after the start
    double duration() const { return ends_.back(); }
    size_t size() const { return frames_.size(); }
    bool missing() const { return missing_; }
    std::optional<double> loop_start() const { return loop_start_; }

private:
    std::vector<cv::Mat> frames_;               // CV_8UC4 BGRA
    std::vector<double> ends_;                  // cumulative end time of each frame, seconds
    std::optional<double> loop_start_;
    bool missing_ = false;
};

// Text card used when an asset file is missing (420x260, dark, centred caps text).
cv::Mat text_card(const std::string& text);

// Alpha-blend BGRA `src` onto BGR `dst` with top-left at (x, y), clipped to the frame.
void blend(cv::Mat& dst, const cv::Mat& src, int x, int y, double alpha = 1.0);

// Resize to `height` keeping the aspect ratio. Cached (LRU): memes show the same GIF frames over and over.
cv::Mat fit(const cv::Mat& img, int height);

// Meme in a HUD-style frame with a title tab. `accent` is BGR.
void panel(cv::Mat& dst, const cv::Mat& img, int x, int y, const std::string& title, double alpha,
           cv::Scalar accent = {80, 220, 255});

// Put a w x h overlay next to `box`, on whichever side has more room. Returns its top-left.
cv::Point place_beside(cv::Size frame, const Box& box, int w, int h, int margin = 20);

// The meme (unwarped, aspect kept) about `grow` x the head width, centred a bit above the head's centre.
void over_face(cv::Mat& dst, const cv::Mat& img, const Box& head, double alpha, double grow = 1.6);

// Face quad for warping: forehead-left, forehead-right, jaw-right, jaw-left (MediaPipe face mesh ids).
inline constexpr int FACE_QUAD[4] = {103, 332, 365, 136};

// Homography-warp the meme onto the face quad (grown by `grow`), feathered oval mask.
void warp_on_face(cv::Mat& dst, const cv::Mat& img, const std::vector<Vec3>& face_pts, double alpha,
                  double grow = 1.35);

// Black-and-white frame + the big "ABSOLUTE CINEMA" caption.
class CinemaEffect {
public:
    void apply(cv::Mat& dst, double alpha);
    const cv::Mat& caption(int width);          // BGRA, width x width/5, cached per width
private:
    std::map<int, cv::Mat> cache_;
};

// Bold TTF text (Noto Sans Black / DejaVu Sans Bold, like the Python Pillow fonts) rendered to BGRA:
// white fill with an optional black stroke, the ink's top-centre placed at (cx, top). Falls back to
// OpenCV's built-in font when the freetype module or the font files are missing.
void draw_bold_text(cv::Mat& bgra, const std::string& text, int cx, int top, int size, int stroke = 0);

}  // namespace memecam
