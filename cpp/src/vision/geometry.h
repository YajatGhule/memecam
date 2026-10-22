// MediaPipe calculator math, reimplemented 1:1 (names follow the MediaPipe calculators they mirror):
// SsdAnchors, TensorsToDetections, NonMaxSuppression (weighted), DetectionProjection / letterbox removal,
// DetectionsToRects (+ AlignmentPointsRects), RectTransformation, ImageToTensor (rotated crop),
// TensorsToLandmarks + LandmarkLetterboxRemoval + LandmarkProjection, IoU / association helpers.
#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace memecam::vision {

// mediapipe::NormalizedRect: center / size relative to the image, rotation in radians (clockwise in image).
struct NormRect {
    float xc = 0, yc = 0, w = 0, h = 0, rot = 0;
};

// mediapipe::Detection with a RELATIVE_BOUNDING_BOX and relative keypoints (single class).
struct Detection {
    float xmin = 0, ymin = 0, w = 0, h = 0;
    float score = 0;
    std::vector<cv::Point2f> kps;
};

struct NormLandmark {
    float x = 0, y = 0, z = 0, visibility = 0, presence = 0;
};
using Landmarks = std::vector<NormLandmark>;

float normalize_radians(float a);
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// ---- SSD detection decoding ------------------------------------------------------------------
struct Anchor { float xc, yc, w, h; };
struct AnchorOptions {
    int input_w, input_h;
    float min_scale, max_scale, offset_x = 0.5f, offset_y = 0.5f;
    std::vector<int> strides;
    std::vector<float> aspect_ratios{1.f};
    float interpolated_scale_aspect_ratio = 1.f;
    bool fixed_anchor_size = true;
};
std::vector<Anchor> ssd_anchors(const AnchorOptions& o);

struct DecodeOptions {
    int num_boxes, num_coords, box_coord_offset = 0, keypoint_coord_offset = 4, num_keypoints, values_per_keypoint = 2;
    float x_scale, y_scale, w_scale, h_scale;
    float score_clipping_thresh = 100.f;
    float min_score_thresh = 0.5f;
    bool reverse_output_order = true;   // XYWH
};
// raw_boxes: num_boxes x num_coords, raw_scores: num_boxes (one class; sigmoid applied here).
std::vector<Detection> decode_detections(const float* raw_boxes, const float* raw_scores,
                                         const std::vector<Anchor>& anchors, const DecodeOptions& o);

// NonMaxSuppressionCalculator, WEIGHTED algorithm, INTERSECTION_OVER_UNION overlap.
std::vector<Detection> weighted_nms(const std::vector<Detection>& dets, float min_suppression_threshold);

// ---- ImageToTensor -----------------------------------------------------------------------------
struct RotatedRoi { float xc, yc, w, h, rot; };   // pixels
struct TensorInput {
    std::array<float, 4> padding{};               // letterbox padding (left, top, right, bottom), relative
    std::array<float, 16> matrix{};               // tensor-normalized -> image-normalized
    RotatedRoi roi{};                             // ROI after padding
};
// Crops `rect` (or the whole image if null) rotated, resizes to tw x th, writes RGB floats in
// [range_min, range_max] to dst (tw*th*3). border_zero: BORDER_CONSTANT 0 (else BORDER_REPLICATE).
// Returns false for a sentinel / invalid ROI (MediaPipe then produces no output).
bool image_to_tensor(const cv::Mat& rgb, const NormRect* rect, int tw, int th, bool keep_aspect, bool border_zero,
                     float range_min, float range_max, float* dst, TensorInput* info);

void get_rotated_subrect_to_rect_matrix(const RotatedRoi& r, int w, int h, std::array<float, 16>& m);

// DetectionProjectionCalculator (box corners and keypoints through `matrix`).
void project_detection(Detection& d, const std::array<float, 16>& matrix);
// DetectionLetterboxRemovalCalculator.
void remove_letterbox(Detection& d, const std::array<float, 4>& padding);

// DetectionsToRectsCalculator (box mode) with optional rotation from keypoints [start] -> [end].
NormRect detection_to_rect(const Detection& d, int img_w, int img_h, int start_kp, int end_kp, float target_angle);
// AlignmentPointsRectsCalculator: center = kp[start], size = 2 * |kp[end] - kp[start]| (pixels).
NormRect alignment_points_to_rect(const Detection& d, int img_w, int img_h, int start_kp, int end_kp,
                                  float target_angle);
// RectTransformationCalculator on a NormalizedRect.
NormRect transform_rect(NormRect r, int img_w, int img_h, float scale_x, float scale_y, float shift_x, float shift_y,
                        bool square_long);

// ---- landmarks -------------------------------------------------------------------------------
// LandmarkLetterboxRemovalCalculator.
void remove_letterbox(Landmarks& lms, const std::array<float, 4>& padding);
// LandmarkProjectionCalculator with NORM_RECT only (the "square ROI" path used by hands and pose).
void project_landmarks_rect(Landmarks& lms, const NormRect& r);
// LandmarkProjectionCalculator with NORM_RECT + IMAGE_DIMENSIONS (matrix path, used by the face mesh).
void project_landmarks_matrix(Landmarks& lms, const NormRect& r, int img_w, int img_h);
// LandmarksToDetectionCalculator (all landmarks as keypoints, bbox = their extent).
Detection landmarks_to_detection(const Landmarks& lms);

// ---- rect association ------------------------------------------------------------------------
float rect_iou(const NormRect& a, const NormRect& b);     // axis-aligned, rotation ignored (MediaPipe)
// HandAssociationCalculator: keep `base`, add rects from `added` that overlap none (IoU > thr) so far.
std::vector<NormRect> hand_association(const std::vector<NormRect>& base, const std::vector<NormRect>& added,
                                       float min_similarity);
// Single-instance re-scan: among `found` (detector rects, largest first), the one to switch to instead of the
// tracked `cur` — a detection that does not overlap `cur` and is clearly bigger than the detection that does
// (same units: both come from the detector). nullopt = keep tracking `cur`.
std::optional<NormRect> bigger_instance(const NormRect& cur, const std::vector<NormRect>& found, int img_w, int img_h,
                                        float min_ratio = 1.25f);
// Sort detector rects by pixel area, largest first (stable).
void sort_largest_first(std::vector<NormRect>& rects, int img_w, int img_h);

// AssociationNormRectCalculator: base then added; each element replaces the earlier ones it overlaps.
std::vector<NormRect> norm_rect_association(const std::vector<NormRect>& base, const std::vector<NormRect>& added,
                                            float min_similarity);

}  // namespace memecam::vision
