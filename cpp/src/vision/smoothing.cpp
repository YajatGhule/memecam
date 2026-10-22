#include "vision/smoothing.h"

#include <algorithm>
#include <cmath>

namespace memecam::vision {

float LowPassFilter::apply(float value) {
    float result;
    if (initialized_) {
        result = alpha_ * value + (1.0 - alpha_) * stored_;
    } else {
        result = value;
        initialized_ = true;
    }
    raw_ = value;
    stored_ = result;
    return result;
}

float LowPassFilter::apply_with_alpha(float value, float alpha) {
    if (alpha >= 0.0f && alpha <= 1.0f) alpha_ = alpha;
    return apply(value);
}

OneEuroFilter::OneEuroFilter(double frequency, double min_cutoff, double beta, double derivate_cutoff)
    : frequency_(frequency), min_cutoff_(min_cutoff), beta_(beta), derivate_cutoff_(derivate_cutoff),
      x_(static_cast<float>(alpha(min_cutoff))), dx_(static_cast<float>(alpha(derivate_cutoff))) {}

double OneEuroFilter::alpha(double cutoff) const {
    const double te = 1.0 / frequency_;
    const double tau = 1.0 / (2 * M_PI * cutoff);
    return 1.0 / (1.0 + tau / te);
}

double OneEuroFilter::apply(int64_t ts, double value, double value_scale) {
    if (last_time_ >= ts) return value;
    if (last_time_ != 0 && ts != 0) frequency_ = 1.0 / ((ts - last_time_) * 1e-9);
    last_time_ = ts;
    const double dvalue = x_.has_last_raw_value() ? (value - x_.last_raw_value()) * value_scale * frequency_ : 0.0;
    const double edvalue = dx_.apply_with_alpha(static_cast<float>(dvalue), static_cast<float>(alpha(derivate_cutoff_)));
    const double cutoff = min_cutoff_ + beta_ * std::fabs(edvalue);
    return x_.apply_with_alpha(static_cast<float>(value), static_cast<float>(alpha(cutoff)));
}

void LandmarksSmoother::apply(Landmarks& lms, int64_t ts_us, int W, int H, std::optional<float> object_scale) {
    if (lms.empty()) {
        reset();
        return;
    }
    const int n = static_cast<int>(lms.size());
    if (x_.empty())
        for (int i = 0; i < n; ++i) {
            x_.emplace_back(p_.frequency, p_.min_cutoff, p_.beta, p_.derivate_cutoff);
            y_.emplace_back(p_.frequency, p_.min_cutoff, p_.beta, p_.derivate_cutoff);
            z_.emplace_back(p_.frequency, p_.min_cutoff, p_.beta, p_.derivate_cutoff);
        }
    // NormalizedLandmarksToLandmarks: pixels.
    std::vector<float> px(n), py(n), pz(n);
    for (int i = 0; i < n; ++i) {
        px[i] = lms[i].x * W;
        py[i] = lms[i].y * H;
        pz[i] = lms[i].z * W;
    }
    float scale;
    if (object_scale) {
        scale = *object_scale;
    } else {
        auto [x0, x1] = std::minmax_element(px.begin(), px.end());
        auto [y0, y1] = std::minmax_element(py.begin(), py.end());
        scale = ((*x1 - *x0) + (*y1 - *y0)) / 2.0f;
    }
    if (scale < 1e-6f) return;   // min_allowed_object_scale: pass through unchanged
    const float value_scale = 1.0f / scale;
    const int64_t ts_ns = ts_us * 1000;
    for (int i = 0; i < n; ++i) {
        const float x = static_cast<float>(x_[i].apply(ts_ns, px[i], value_scale));
        const float y = static_cast<float>(y_[i].apply(ts_ns, py[i], value_scale));
        const float z = static_cast<float>(z_[i].apply(ts_ns, pz[i], value_scale));
        lms[i].x = x / W;
        lms[i].y = y / H;
        lms[i].z = z / W;
    }
}

void VisibilitySmoother::apply(Landmarks& lms) {
    if (lms.empty()) {
        reset();
        return;
    }
    if (f_.empty()) f_.assign(lms.size(), LowPassFilter(alpha_));
    for (size_t i = 0; i < lms.size(); ++i) lms[i].visibility = f_[i].apply(lms[i].visibility);
}

}  // namespace memecam::vision
