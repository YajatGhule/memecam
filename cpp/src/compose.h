// Frame composition: the person's active memes + markers (compose / corner / debug_panel in memecam/main.py).
// Shared by the app and tools/render_check.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "config.h"
#include "crowd.h"
#include "gestures.h"
#include "overlays.h"
#include "types.h"

namespace memecam {

// Decoded media per meme name (decoded once at startup, shared read-only by the display thread).
using MemeMedia = std::map<std::string, std::unique_ptr<Media>>;
MemeMedia load_meme_media(const Config& cfg, const std::string& assets_dir);

inline constexpr double FADE_IN = 0.15;   // seconds for a meme to fade in

// Top-left of a w x h panel in a corner ("tl"/"tr"/"bl"/"br"); the k-th panel in the same corner sits
// next to the previous one.
cv::Point corner(cv::Size frame, int w, int h, const std::string& where, int k = 0);

// Draw the active memes (and the markers) onto `frame`. `on_face` = put every meme over the person's face
// ([display] meme_on_face). Memes are drawn in config order. Returns the active meme names.
std::vector<std::string> compose(cv::Mat& frame, const Frame& fr, const std::vector<Shown>& shown, double now,
                                 const Config& cfg, const MemeMedia& media, CinemaEffect& cinema,
                                 bool show_markers, bool on_face);

// Gesture debug scores (key i): one bar per meme, right side, green while its raw detection is on.
void debug_panel(cv::Mat& img, const std::map<std::string, Detection>& dets, const Config& cfg);

}  // namespace memecam
