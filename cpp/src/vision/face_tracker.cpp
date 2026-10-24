// FaceLandmarker graph (VIDEO mode): FaceDetectorGraph (BlazeFace short range) + association with the previous
// frame's ROIs + FaceLandmarksDetectorGraph (478-point mesh incl. iris), with one-euro smoothing when a single
// face is requested (face_landmarker_graph.cc / face_landmarks_detector_graph.cc). Blendshapes not run.
#include <chrono>

#include "vision/pipelines.h"

namespace memecam::vision {

namespace {
constexpr float kMinDetection = 0.5f, kMinTracking = 0.5f, kMinSuppression = 0.5f;
// Face presence needed to keep tracking a face (MediaPipe default 0.5). Lower: a head tilted far back (Willem
// Dafoe looking up) drops to ~0.3 while it is clearly still a face, and the detector can't re-find it then.
constexpr float kMinPresence = 0.25f;
constexpr int kDetSize = 128, kLmSize = 256, kNumLm = 478;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

FaceModels::FaceModels(const std::string& dir, int threads, bool gpu)
    : detector(dir + "/face_detector.tflite", threads, gpu),
      landmarks(dir + "/face_landmarks_detector.tflite", threads, gpu) {
    AnchorOptions a;
    a.input_w = a.input_h = kDetSize;
    a.min_scale = 0.1484375f;
    a.max_scale = 0.75f;
    a.strides = {8, 16, 16, 16};
    anchors = ssd_anchors(a);
}

FaceTracker::FaceTracker(FaceModels& m, int num_faces, int rescan_every)
    : m_(m), num_(num_faces), rescan_(num_faces == 1 ? rescan_every : 0) {
    if (num_ == 1) smoother_.emplace(OneEuroParams{0.05, 80.0, 1.0});
}

std::vector<NormRect> FaceTracker::detect(const cv::Mat& rgb, int limit) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    TensorInput ti;
    image_to_tensor(rgb, nullptr, kDetSize, kDetSize, true, true, -1.f, 1.f, m_.detector.input(0), &ti);
    m_.detector.invoke();
    const int bi = m_.detector.output_size(0) > m_.detector.output_size(1) ? 0 : 1;
    DecodeOptions o{896, 16, 0, 4, 6, 2, 128.f, 128.f, 128.f, 128.f, 100.f, kMinDetection, true};
    auto dets = weighted_nms(decode_detections(m_.detector.output(bi), m_.detector.output(1 - bi), m_.anchors, o),
                             kMinSuppression);
    std::vector<NormRect> rects;
    for (auto& d : dets) {
        project_detection(d, ti.matrix);
        NormRect r = detection_to_rect(d, W, H, 0, 1, 0.f);   // eyes, target 0 degrees
        rects.push_back(transform_rect(r, W, H, 1.5f, 1.5f, 0.f, 0.f, false));
    }
    if (rescan_) sort_largest_first(rects, W, H);   // one face wanted: the closest one, not the surest
    if (limit >= 0 && static_cast<int>(rects.size()) > limit) rects.resize(limit);   // ClipDetectionVectorSize
    m_.stats.add(m_.stats.det_ms, ms_since(t0));
    m_.stats.det_runs.fetch_add(1);
    return rects;
}

std::vector<Landmarks> FaceTracker::process(const cv::Mat& rgb, int64_t ts_ms) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    std::vector<NormRect> rects;
    if (static_cast<int>(prev_.size()) < num_) {
        rects = norm_rect_association(prev_, detect(rgb, num_), kMinTracking);
    } else {
        rects = norm_rect_association(prev_, {}, kMinTracking);
        if (rescan_ && ++frame_ % rescan_ == 0) {
            if (auto r = bigger_instance(rects[0], detect(rgb, -1), W, H)) {
                rects = {*r};
                if (smoother_) smoother_->reset();   // a different face: don't glide over from the old one
            }
        }
    }
    if (static_cast<int>(rects.size()) > num_) rects.resize(num_);

    std::vector<Landmarks> faces;
    prev_.clear();
    const auto t1 = Clock::now();
    for (const auto& rect : rects) {
        if (!image_to_tensor(rgb, &rect, kLmSize, kLmSize, false, false, 0.f, 1.f, m_.landmarks.input(0), nullptr))
            continue;
        m_.landmarks.invoke();
        m_.stats.lm_runs.fetch_add(1);
        if (!(sigmoid(m_.landmarks.output(1)[0]) > kMinPresence)) continue;
        const float* raw = m_.landmarks.output(0);
        Landmarks l(kNumLm);
        for (int i = 0; i < kNumLm; ++i)
            l[i] = {raw[i * 3] / kLmSize, raw[i * 3 + 1] / kLmSize, raw[i * 3 + 2] / kLmSize, 0, 0};
        project_landmarks_matrix(l, rect, W, H);
        // Next frame's ROI: landmarks -> detection -> rect rotated by the eye corners 33 -> 263.
        NormRect r = detection_to_rect(landmarks_to_detection(l), W, H, 33, 263, 0.f);
        prev_.push_back(transform_rect(r, W, H, 1.5f, 1.5f, 0.f, 0.f, true));
        faces.push_back(std::move(l));
    }
    m_.stats.add(m_.stats.lm_ms, ms_since(t1));
    if (smoother_) {
        if (faces.empty()) smoother_->reset();
        else smoother_->apply(faces[0], ts_ms * 1000, W, H, std::nullopt);
    }
    m_.stats.add(m_.stats.total_ms, ms_since(t0));
    m_.stats.calls.fetch_add(1);
    return faces;
}

}  // namespace memecam::vision
