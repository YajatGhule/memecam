// Tracker: MediaPipe-equivalent hand / face / pose pipelines on LiteRT for one person (the closest one):
// returns a Frame with their hands, face mesh and body pose (`people` holds that one person, or nobody).
#pragma once

#include <memory>
#include <string>

#include "config.h"
#include "types.h"

namespace memecam {

// Ungrouped model results of the latest process() call (MediaPipe result lists, in MediaPipe order).
struct TrackerRaw {
    std::vector<Hand> hands;
    std::vector<Face> faces;
    std::vector<std::shared_ptr<const Pose>> poses;   // the pose results grouped this frame (may be older)
    bool pose_fresh = false;                          // a pose job finished and was taken this frame
};

class Tracker {
public:
    Tracker(const Config& cfg, bool use_pose = true, bool pose_async = true);
    ~Tracker();
    Frame process(const cv::Mat& rgb, int64_t ts_ms);   // rgb: CV_8UC3 RGB, mirrored
    std::string device() const;                          // "cpu" / "gpu"
    const TrackerRaw& last_raw() const;                  // debugging / parity tools
    std::string stats() const;                           // per-model mean ms, e.g. for --bench
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace memecam
