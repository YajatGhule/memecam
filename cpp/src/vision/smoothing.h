// MediaPipe landmark smoothing used in VIDEO mode with a single face / pose:
// LandmarksSmoothingCalculator (one-euro filter per coordinate, value scale = 1 / object size) and
// VisibilitySmoothingCalculator (low-pass on visibility). Ported from mediapipe/util/filtering.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "vision/geometry.h"

namespace memecam::vision {

class LowPassFilter {
public:
    explicit LowPassFilter(float alpha) : alpha_(alpha) {}
    float apply(float value);
    float apply_with_alpha(float value, float alpha);
    bool has_last_raw_value() const { return initialized_; }
    float last_raw_value() const { return raw_; }
private:
    float alpha_;
    bool initialized_ = false;
    float raw_ = 0, stored_ = 0;
};

class OneEuroFilter {
public:
    OneEuroFilter(double frequency, double min_cutoff, double beta, double derivate_cutoff);
    double apply(int64_t timestamp_ns, double value, double value_scale);
private:
    double alpha(double cutoff) const;
    double frequency_, min_cutoff_, beta_, derivate_cutoff_;
    LowPassFilter x_, dx_;
    int64_t last_time_ = -1;
};

struct OneEuroParams { double min_cutoff, beta, derivate_cutoff, frequency = 30.0; };

// One-euro smoothing of normalized landmarks in pixel space (x*W, y*H, z*W), like LandmarksSmoothingCalculator.
class LandmarksSmoother {
public:
    explicit LandmarksSmoother(OneEuroParams p) : p_(p) {}
    // object_scale: pixels; nullopt = from the landmarks' extent.
    void apply(Landmarks& lms, int64_t ts_us, int W, int H, std::optional<float> object_scale);
    void reset() { x_.clear(); y_.clear(); z_.clear(); }
private:
    OneEuroParams p_;
    std::vector<OneEuroFilter> x_, y_, z_;
};

class VisibilitySmoother {
public:
    explicit VisibilitySmoother(float alpha) : alpha_(alpha) {}
    void apply(Landmarks& lms);
    void reset() { f_.clear(); }
private:
    float alpha_;
    std::vector<LowPassFilter> f_;
};

}  // namespace memecam::vision
