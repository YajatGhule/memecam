// PoseLandmarker graph (VIDEO mode): PoseDetectorGraph + association with the previous frame's ROIs +
// PoseLandmarksDetectorGraph (lite model: 33 landmarks + 2 auxiliary ROI points, heatmap refinement), with
// visibility low-pass and one-euro landmark smoothing when a single pose is requested
// (pose_detector_graph.cc, pose_landmarker_graph.cc, pose_landmarks_detector_graph.cc).
#include <algorithm>
#include <chrono>

#include "vision/pipelines.h"

namespace memecam::vision {

namespace {
constexpr float kMinDetection = 0.5f, kMinPresence = 0.5f, kMinTracking = 0.5f, kMinSuppression = 0.5f;
constexpr int kDetSize = 224, kLmSize = 256, kNumLm = 39, kNumPose = 33;
constexpr float kHalfPi = static_cast<float>(M_PI) / 2.f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// RefineLandmarksFromHeatmapCalculator (kernel 7, min_confidence_to_refine 0.5), on crop-normalized coords.
void refine_from_heatmap(Landmarks& lms, const float* hm, int hm_h, int hm_w, int ch, int kernel) {
    const int row_size = hm_w * ch;
    for (int li = 0; li < static_cast<int>(lms.size()); ++li) {
        const int cc = static_cast<int>(lms[li].x * hm_w), cr = static_cast<int>(lms[li].y * hm_h);
        if (cc < 0 || cc >= hm_w || cr < 0 || cr >= hm_h) continue;
        const int off = (kernel - 1) / 2;
        const int c0 = std::max(0, cc - off), c1 = std::min(hm_w, cc + off + 1);
        const int r0 = std::max(0, cr - off), r1 = std::min(hm_h, cr + off + 1);
        float sum = 0, wc = 0, wr = 0, maxc = 0;
        for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c) {
                const float conf = sigmoid(hm[row_size * r + ch * c + li]);
                sum += conf;
                maxc = std::max(maxc, conf);
                wc += c * conf;
                wr += r * conf;
            }
        if (maxc >= 0.5f && sum > 0) {
            lms[li].x = wc / hm_w / sum;
            lms[li].y = wr / hm_h / sum;
        }
    }
}
}  // namespace

PoseModels::PoseModels(const std::string& dir, int threads, bool gpu)
    : detector(dir + "/pose_detector.tflite", threads, gpu),
      landmarks(dir + "/pose_landmarks_detector.tflite", threads, gpu) {
    AnchorOptions a;
    a.input_w = a.input_h = kDetSize;
    a.min_scale = 0.1484375f;
    a.max_scale = 0.75f;
    a.strides = {8, 16, 32, 32, 32};
    anchors = ssd_anchors(a);
}

PoseTracker::PoseTracker(PoseModels& m, int num_poses, int rescan_every)
    : m_(m), num_(num_poses), rescan_(num_poses == 1 ? rescan_every : 0) {
    if (num_ == 1) {
        smoother_.emplace(OneEuroParams{0.05, 80.0, 1.0});
        vis_smoother_.emplace(0.1f);
    }
}

std::vector<NormRect> PoseTracker::detect(const cv::Mat& rgb) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    TensorInput ti;
    image_to_tensor(rgb, nullptr, kDetSize, kDetSize, true, true, -1.f, 1.f, m_.detector.input(0), &ti);
    m_.detector.invoke();
    const int bi = m_.detector.output_size(0) > m_.detector.output_size(1) ? 0 : 1;
    DecodeOptions o{2254, 12, 0, 4, 4, 2, 224.f, 224.f, 224.f, 224.f, 100.f, kMinDetection, true};
    auto dets = weighted_nms(decode_detections(m_.detector.output(bi), m_.detector.output(1 - bi), m_.anchors, o),
                             kMinSuppression);
    std::vector<NormRect> rects;
    for (auto& d : dets) {
        remove_letterbox(d, ti.padding);
        NormRect r = alignment_points_to_rect(d, W, H, 0, 1, kHalfPi);
        rects.push_back(transform_rect(r, W, H, 1.25f, 1.25f, 0.f, 0.f, true));
    }
    if (rescan_) sort_largest_first(rects, W, H);   // one body wanted: the closest one, not the surest
    m_.stats.add(m_.stats.det_ms, ms_since(t0));
    m_.stats.det_runs.fetch_add(1);
    return rects;
}

std::vector<Landmarks> PoseTracker::process(const cv::Mat& rgb, int64_t ts_ms) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    std::vector<NormRect> rects;
    if (static_cast<int>(prev_.size()) < num_) {
        rects = norm_rect_association(prev_, detect(rgb), kMinTracking);
    } else {
        rects = norm_rect_association(prev_, {}, kMinTracking);
        if (rescan_ && ++frame_ % rescan_ == 0)
            if (auto r = bigger_instance(rects[0], detect(rgb), W, H)) {
                rects = {*r};
                if (smoother_) smoother_->reset(), vis_smoother_->reset();
            }
    }
    if (static_cast<int>(rects.size()) > num_) rects.resize(num_);

    std::vector<Landmarks> poses;
    prev_.clear();
    const auto t1 = Clock::now();
    auto& lm = m_.landmarks;
    for (const auto& rect : rects) {
        if (!image_to_tensor(rgb, &rect, kLmSize, kLmSize, false, false, 0.f, 1.f, lm.input(0), nullptr)) continue;
        lm.invoke();
        m_.stats.lm_runs.fetch_add(1);
        if (!(lm.output(1)[0] > kMinPresence)) continue;
        const float* raw = lm.output(0);
        const int dims = lm.output_size(0) / kNumLm;
        Landmarks l(kNumLm);
        for (int i = 0; i < kNumLm; ++i) {
            const float* p = raw + i * dims;
            l[i] = {p[0] / kLmSize, p[1] / kLmSize, p[2] / kLmSize, sigmoid(p[3]), sigmoid(p[4])};
        }
        const auto hs = lm.output_shape(3);   // 1 x 64 x 64 x 39
        refine_from_heatmap(l, lm.output(3), hs[1], hs[2], hs[3], 7);
        Landmarks aux(l.begin() + kNumPose, l.begin() + kNumPose + 2);
        l.resize(kNumPose);
        project_landmarks_rect(l, rect);
        project_landmarks_rect(aux, rect);
        NormRect r = alignment_points_to_rect(landmarks_to_detection(aux), W, H, 0, 1, kHalfPi);
        prev_.push_back(transform_rect(r, W, H, 1.25f, 1.25f, 0.f, 0.f, true));
        poses.push_back(std::move(l));
    }
    m_.stats.add(m_.stats.lm_ms, ms_since(t1));
    if (smoother_) {
        if (poses.empty()) {
            vis_smoother_->reset();
            smoother_->reset();
        } else {
            vis_smoother_->apply(poses[0]);
            const float scale = (prev_[0].w * W + prev_[0].h * H) / 2.0f;   // object scale from the next ROI
            smoother_->apply(poses[0], ts_ms * 1000, W, H, scale);
        }
    }
    m_.stats.add(m_.stats.total_ms, ms_since(t0));
    m_.stats.calls.fetch_add(1);
    return poses;
}

}  // namespace memecam::vision
