// Motion-capture style rendering: glowing markers, skeletons, bracket bounding boxes, HUD
// (port of memecam/mocap.py).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "types.h"

namespace memecam {

// Meme panel frame colour (BGR).
inline const cv::Scalar ACCENT_COL(80, 220, 255);

// Corner-bracket bounding box, like tracking HUDs, with an optional label tab.
void brackets(cv::Mat& img, const Box& box, cv::Scalar color, const std::string& label = "", int pad = 12,
              int arm = 18, int t = 2);

// Draw the tracked person's markers in place: face mesh contours, hands, body skeleton, labels and brackets.
// Glow comes from a blurred quarter-size copy of the marker layer.
void draw_markers(cv::Mat& img, const Frame& fr);

// FPS line (+ tracking FPS), active memes, REC dot, key hints.
void hud(cv::Mat& img, double fps, const std::vector<std::string>& active, bool recording, bool show_markers,
         std::optional<double> track_fps = std::nullopt, bool show_memes = true,
         std::optional<bool> sound = std::nullopt);   // sound: on / off ([m]); nullopt = no sounds at all

}  // namespace memecam
