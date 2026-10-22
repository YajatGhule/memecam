#include "vision/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>   // OpenCV 5: boxPoints / getPerspectiveTransform live here
#endif

namespace memecam::vision {

float normalize_radians(float a) {
    return a - 2 * static_cast<float>(M_PI) * std::floor((a - (-static_cast<float>(M_PI))) / (2 * static_cast<float>(M_PI)));
}

// ---- SsdAnchorsCalculator --------------------------------------------------------------------
static float calculate_scale(float min_scale, float max_scale, int stride_index, int num_strides) {
    if (num_strides == 1) return (min_scale + max_scale) * 0.5f;
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
}

std::vector<Anchor> ssd_anchors(const AnchorOptions& o) {
    std::vector<Anchor> anchors;
    const int n = static_cast<int>(o.strides.size());
    int layer = 0;
    while (layer < n) {
        std::vector<float> heights, widths, ratios, scales;
        int last = layer;
        while (last < n && o.strides[last] == o.strides[layer]) {
            const float scale = calculate_scale(o.min_scale, o.max_scale, last, n);
            for (float ar : o.aspect_ratios) {
                ratios.push_back(ar);
                scales.push_back(scale);
            }
            if (o.interpolated_scale_aspect_ratio > 0.f) {
                const float next = last == n - 1 ? 1.0f : calculate_scale(o.min_scale, o.max_scale, last + 1, n);
                scales.push_back(std::sqrt(scale * next));
                ratios.push_back(o.interpolated_scale_aspect_ratio);
            }
            ++last;
        }
        for (size_t i = 0; i < ratios.size(); ++i) {
            const float r = std::sqrt(ratios[i]);
            heights.push_back(scales[i] / r);
            widths.push_back(scales[i] * r);
        }
        const int stride = o.strides[layer];
        const int fh = static_cast<int>(std::ceil(1.0f * o.input_h / stride));
        const int fw = static_cast<int>(std::ceil(1.0f * o.input_w / stride));
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x)
                for (size_t a = 0; a < heights.size(); ++a) {
                    Anchor an{(x + o.offset_x) * 1.0f / fw, (y + o.offset_y) * 1.0f / fh, 1.f, 1.f};
                    if (!o.fixed_anchor_size) {
                        an.w = widths[a];
                        an.h = heights[a];
                    }
                    anchors.push_back(an);
                }
        layer = last;
    }
    return anchors;
}

// ---- TensorsToDetectionsCalculator -----------------------------------------------------------
std::vector<Detection> decode_detections(const float* raw, const float* raw_scores,
                                         const std::vector<Anchor>& anchors, const DecodeOptions& o) {
    std::vector<Detection> out;
    for (int i = 0; i < o.num_boxes; ++i) {
        float score = raw_scores[i];
        score = std::clamp(score, -o.score_clipping_thresh, o.score_clipping_thresh);
        score = 1.0f / (1.0f + std::exp(-score));
        if (score < o.min_score_thresh) continue;
        const float* b = raw + i * o.num_coords + o.box_coord_offset;
        const Anchor& an = anchors[i];
        float xc, yc, w, h;
        if (o.reverse_output_order) {
            xc = b[0]; yc = b[1]; w = b[2]; h = b[3];
        } else {
            yc = b[0]; xc = b[1]; h = b[2]; w = b[3];
        }
        xc = xc / o.x_scale * an.w + an.xc;
        yc = yc / o.y_scale * an.h + an.yc;
        h = h / o.h_scale * an.h;
        w = w / o.w_scale * an.w;
        const float ymin = yc - h / 2.f, xmin = xc - w / 2.f, ymax = yc + h / 2.f, xmax = xc + w / 2.f;
        Detection d;
        d.score = score;
        d.xmin = xmin;
        d.ymin = ymin;
        d.w = xmax - xmin;
        d.h = ymax - ymin;
        if (d.w < 0 || d.h < 0 || std::isnan(d.w) || std::isnan(d.h)) continue;
        const float* raw_box = raw + i * o.num_coords;
        for (int k = 0; k < o.num_keypoints; ++k) {
            const float* kp = raw_box + o.keypoint_coord_offset + k * o.values_per_keypoint;
            float kx, ky;
            if (o.reverse_output_order) {
                kx = kp[0]; ky = kp[1];
            } else {
                ky = kp[0]; kx = kp[1];
            }
            d.kps.emplace_back(kx / o.x_scale * an.w + an.xc, ky / o.y_scale * an.h + an.yc);
        }
        out.push_back(std::move(d));
    }
    return out;
}

// ---- NonMaxSuppressionCalculator (WEIGHTED) --------------------------------------------------
static float iou_boxes(const Detection& a, const Detection& b) {
    const float ix0 = std::max(a.xmin, b.xmin), iy0 = std::max(a.ymin, b.ymin);
    const float ix1 = std::min(a.xmin + a.w, b.xmin + b.w), iy1 = std::min(a.ymin + a.h, b.ymin + b.h);
    if (ix1 < ix0 || iy1 < iy0) return 0.f;   // Rectangle_f::Intersects: touching counts, area 0
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float norm = a.w * a.h + b.w * b.h - inter;
    return norm > 0.f ? inter / norm : 0.f;
}

std::vector<Detection> weighted_nms(const std::vector<Detection>& dets, float thr) {
    std::vector<std::pair<int, float>> remained;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) remained.emplace_back(i, dets[i].score);
    std::stable_sort(remained.begin(), remained.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::vector<Detection> out;
    while (!remained.empty()) {
        const size_t original = remained.size();
        const Detection& det = dets[remained[0].first];
        std::vector<std::pair<int, float>> rest, cand;
        for (auto& is : remained) {
            if (iou_boxes(dets[is.first], det) > thr) cand.push_back(is);
            else rest.push_back(is);
        }
        Detection w = det;
        if (!cand.empty()) {
            const size_t nk = det.kps.size();
            std::vector<float> kps(nk * 2, 0.f);
            float wx0 = 0, wy0 = 0, wx1 = 0, wy1 = 0, total = 0;
            for (auto& c : cand) {
                const Detection& d = dets[c.first];
                total += c.second;
                wx0 += d.xmin * c.second;
                wy0 += d.ymin * c.second;
                wx1 += (d.xmin + d.w) * c.second;
                wy1 += (d.ymin + d.h) * c.second;
                for (size_t k = 0; k < nk; ++k) {
                    kps[2 * k] += d.kps[k].x * c.second;
                    kps[2 * k + 1] += d.kps[k].y * c.second;
                }
            }
            w.xmin = wx0 / total;
            w.ymin = wy0 / total;
            w.w = wx1 / total - w.xmin;
            w.h = wy1 / total - w.ymin;
            for (size_t k = 0; k < nk; ++k) w.kps[k] = {kps[2 * k] / total, kps[2 * k + 1] / total};
        }
        out.push_back(std::move(w));
        if (original == rest.size()) break;
        remained = std::move(rest);
    }
    return out;
}

// ---- ImageToTensorCalculator -----------------------------------------------------------------
void get_rotated_subrect_to_rect_matrix(const RotatedRoi& r, int w, int h, std::array<float, 16>& m) {
    const float a = r.w, b = r.h, c = std::cos(r.rot), d = std::sin(r.rot), e = r.xc, f = r.yc;
    const float g = 1.0f / w, hh = 1.0f / h;
    m = {a * c * g, -b * d * g, 0.f, (-0.5f * a * c + 0.5f * b * d + e) * g,
         a * d * hh, b * c * hh, 0.f, (-0.5f * b * c - 0.5f * a * d + f) * hh,
         0.f, 0.f, a * g, 0.f,
         0.f, 0.f, 0.f, 1.f};
}

bool image_to_tensor(const cv::Mat& rgb, const NormRect* rect, int tw, int th, bool keep_aspect, bool border_zero,
                     float range_min, float range_max, float* dst, TensorInput* info) {
    const int W = rgb.cols, H = rgb.rows;
    RotatedRoi roi{0.5f * W, 0.5f * H, static_cast<float>(W), static_cast<float>(H), 0.f};
    if (rect) {
        if (rect->w == 0 && rect->h == 0) return false;   // sentinel rect
        roi = {rect->xc * W, rect->yc * H, rect->w * W, rect->h * H, rect->rot};
    }
    std::array<float, 4> padding{0, 0, 0, 0};
    if (keep_aspect) {
        if (!(roi.w > 0 && roi.h > 0)) return false;
        const float tensor_ar = static_cast<float>(th) / tw;
        const float roi_ar = roi.h / roi.w;
        float vpad = 0, hpad = 0, nw, nh;
        if (tensor_ar > roi_ar) {
            nw = roi.w;
            nh = roi.w * tensor_ar;
            vpad = (1.0f - roi_ar / tensor_ar) / 2.0f;
        } else {
            nw = roi.h / tensor_ar;
            nh = roi.h;
            hpad = (1.0f - tensor_ar / roi_ar) / 2.0f;
        }
        roi.w = nw;
        roi.h = nh;
        padding = {hpad, vpad, hpad, vpad};
    }
    if (std::isnan(roi.xc) || std::isnan(roi.yc) || std::isnan(roi.w) || std::isnan(roi.h) || std::isnan(roi.rot) ||
        roi.w <= 0 || roi.h <= 0)
        return false;
    if (info) {
        info->padding = padding;
        info->roi = roi;
        get_rotated_subrect_to_rect_matrix(roi, W, H, info->matrix);
    }
    const cv::RotatedRect rr(cv::Point2f(roi.xc, roi.yc), cv::Size2f(roi.w, roi.h),
                             roi.rot * 180.f / static_cast<float>(M_PI));
    cv::Mat src_points;
    cv::boxPoints(rr, src_points);
    const float dw = static_cast<float>(tw), dh = static_cast<float>(th);
    float corners[8] = {0.0f, dh, 0.0f, 0.0f, dw, 0.0f, dw, dh};
    cv::Mat dst_points(4, 2, CV_32F, corners);
    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat transformed;
    cv::warpPerspective(rgb, transformed, M, cv::Size(tw, th), cv::INTER_LINEAR,
                        border_zero ? cv::BORDER_CONSTANT : cv::BORDER_REPLICATE);
    const float scale = (range_max - range_min) / 255.0f;
    const float offset = range_min;
    cv::Mat out(th, tw, CV_32FC3, dst);
    transformed.convertTo(out, CV_32FC3, scale, offset);
    return true;
}

// ---- detection projection / letterbox --------------------------------------------------------
void project_detection(Detection& d, const std::array<float, 16>& m) {
    auto proj = [&](cv::Point2f p) {
        return cv::Point2f(p.x * m[0] + p.y * m[1] + m[3], p.x * m[4] + p.y * m[5] + m[7]);
    };
    for (auto& k : d.kps) k = proj(k);
    const cv::Point2f c[4] = {proj({d.xmin, d.ymin}), proj({d.xmin + d.w, d.ymin}),
                              proj({d.xmin + d.w, d.ymin + d.h}), proj({d.xmin, d.ymin + d.h})};
    float x0 = std::numeric_limits<float>::max(), y0 = x0;
    float x1 = std::numeric_limits<float>::lowest(), y1 = x1;
    for (auto& p : c) {
        x0 = std::min(x0, p.x);
        y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x);
        y1 = std::max(y1, p.y);
    }
    d.xmin = x0;
    d.ymin = y0;
    d.w = x1 - x0;
    d.h = y1 - y0;
}

void remove_letterbox(Detection& d, const std::array<float, 4>& p) {
    const float left = p[0], top = p[1], lr = p[0] + p[2], tb = p[1] + p[3];
    d.xmin = (d.xmin - left) / (1.0f - lr);
    d.ymin = (d.ymin - top) / (1.0f - tb);
    d.w = d.w / (1.0f - lr);
    d.h = d.h / (1.0f - tb);
    for (auto& k : d.kps) k = {(k.x - left) / (1.0f - lr), (k.y - top) / (1.0f - tb)};
}

// ---- DetectionsToRects / AlignmentPointsRects / RectTransformation ---------------------------
static float compute_rotation(const Detection& d, int W, int H, int s, int e, float target) {
    const float x0 = d.kps[s].x * W, y0 = d.kps[s].y * H;
    const float x1 = d.kps[e].x * W, y1 = d.kps[e].y * H;
    return normalize_radians(target - std::atan2(-(y1 - y0), x1 - x0));
}

NormRect detection_to_rect(const Detection& d, int W, int H, int s, int e, float target) {
    NormRect r{d.xmin + d.w / 2, d.ymin + d.h / 2, d.w, d.h, 0.f};
    if (s >= 0) r.rot = compute_rotation(d, W, H, s, e, target);
    return r;
}

NormRect alignment_points_to_rect(const Detection& d, int W, int H, int s, int e, float target) {
    const float xc = d.kps[s].x * W, yc = d.kps[s].y * H;
    const float xs = d.kps[e].x * W, ys = d.kps[e].y * H;
    const float box = std::sqrt((xs - xc) * (xs - xc) + (ys - yc) * (ys - yc)) * 2.0f;
    NormRect r{xc / W, yc / H, box / W, box / H, 0.f};
    r.rot = compute_rotation(d, W, H, s, e, target);
    return r;
}

NormRect transform_rect(NormRect r, int W, int H, float sx, float sy, float shx, float shy, bool square_long) {
    float width = r.w, height = r.h;
    const float rot = r.rot;
    if (rot == 0.f) {
        r.xc += width * shx;
        r.yc += height * shy;
    } else {
        const float xs = (W * width * shx * std::cos(rot) - H * height * shy * std::sin(rot)) / W;
        const float ys = (W * width * shx * std::sin(rot) + H * height * shy * std::cos(rot)) / H;
        r.xc += xs;
        r.yc += ys;
    }
    if (square_long) {
        const float long_side = std::max(width * W, height * H);
        width = long_side / W;
        height = long_side / H;
    }
    r.w = width * sx;
    r.h = height * sy;
    return r;
}

// ---- landmarks -------------------------------------------------------------------------------
void remove_letterbox(Landmarks& lms, const std::array<float, 4>& p) {
    const float left = p[0], top = p[1], lr = p[0] + p[2], tb = p[1] + p[3];
    for (auto& l : lms) {
        l.x = (l.x - left) / (1.0f - lr);
        l.y = (l.y - top) / (1.0f - tb);
        l.z = l.z / (1.0f - lr);
    }
}

void project_landmarks_rect(Landmarks& lms, const NormRect& r) {
    const float c = std::cos(r.rot), s = std::sin(r.rot);
    for (auto& l : lms) {
        const float x = l.x - 0.5f, y = l.y - 0.5f;
        float nx = c * x - s * y, ny = s * x + c * y;
        l.x = nx * r.w + r.xc;
        l.y = ny * r.h + r.yc;
        l.z = l.z * r.w;
    }
}

void project_landmarks_matrix(Landmarks& lms, const NormRect& r, int W, int H) {
    RotatedRoi roi{r.xc * W, r.yc * H, r.w * W, r.h * H, r.rot};
    std::array<float, 16> m;
    get_rotated_subrect_to_rect_matrix(roi, W, H, m);
    const float ax = m[3], ay = m[7], bx = m[0] + m[3], by = m[4] + m[7];
    const float zscale = std::sqrt(std::pow(bx - ax, 2.f) + std::pow(by - ay, 2.f));
    for (auto& l : lms) {
        const float x = l.x * m[0] + l.y * m[1] + l.z * m[2] + m[3];
        const float y = l.x * m[4] + l.y * m[5] + l.z * m[6] + m[7];
        l.x = x;
        l.y = y;
        l.z = zscale * l.z;
    }
}

Detection landmarks_to_detection(const Landmarks& lms) {
    Detection d;
    float x0 = std::numeric_limits<float>::max(), x1 = std::numeric_limits<float>::min();
    float y0 = std::numeric_limits<float>::max(), y1 = std::numeric_limits<float>::min();
    for (auto& l : lms) {
        x0 = std::min(x0, l.x);
        x1 = std::max(x1, l.x);
        y0 = std::min(y0, l.y);
        y1 = std::max(y1, l.y);
        d.kps.emplace_back(l.x, l.y);
    }
    d.xmin = x0;
    d.ymin = y0;
    d.w = x1 - x0;
    d.h = y1 - y0;
    return d;
}

// ---- association -----------------------------------------------------------------------------
float rect_iou(const NormRect& a, const NormRect& b) {
    const float ax0 = a.xc - a.w / 2.0f, ay0 = a.yc - a.h / 2.0f, bx0 = b.xc - b.w / 2.0f, by0 = b.yc - b.h / 2.0f;
    const float ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    const float ix1 = std::min(ax0 + a.w, bx0 + b.w), iy1 = std::min(ay0 + a.h, by0 + b.h);
    if (ix1 < ix0 || iy1 < iy0) return 0.f;
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float norm = a.w * a.h + b.w * b.h - inter;
    return norm > 0.f ? inter / norm : 0.f;
}

std::vector<NormRect> hand_association(const std::vector<NormRect>& base, const std::vector<NormRect>& added,
                                       float thr) {
    std::vector<NormRect> out = base;
    for (const auto& r : added) {
        bool overlap = false;
        for (const auto& e : out)
            if (rect_iou(e, r) > thr) {
                overlap = true;
                break;
            }
        if (!overlap) out.push_back(r);
    }
    return out;
}

std::vector<NormRect> norm_rect_association(const std::vector<NormRect>& base, const std::vector<NormRect>& added,
                                            float thr) {
    std::vector<NormRect> out;
    auto add = [&](const NormRect& r) {
        out.erase(std::remove_if(out.begin(), out.end(), [&](const NormRect& e) { return rect_iou(r, e) > thr; }),
                  out.end());
        out.push_back(r);
    };
    // The first non-empty input seeds the list with its first element as-is, the rest go through add().
    bool seeded = false;
    for (const auto* v : {&base, &added}) {
        for (size_t i = 0; i < v->size(); ++i) {
            if (!seeded) {
                out.push_back((*v)[i]);
                seeded = true;
            } else {
                add((*v)[i]);
            }
        }
    }
    return out;
}

void sort_largest_first(std::vector<NormRect>& rects, int W, int H) {
    std::stable_sort(rects.begin(), rects.end(),
                     [&](const NormRect& a, const NormRect& b) { return a.w * W * a.h * H > b.w * W * b.h * H; });
}

std::optional<NormRect> bigger_instance(const NormRect& cur, const std::vector<NormRect>& found, int W, int H,
                                        float min_ratio) {
    const NormRect* same = nullptr;
    for (const auto& r : found)
        if (rect_iou(r, cur) > 0.1f && (!same || r.w * r.h > same->w * same->h)) same = &r;
    if (!same) return std::nullopt;   // the detector doesn't see the tracked one right now: don't guess
    for (const auto& r : found)       // largest first
        if (rect_iou(r, cur) <= 0.1f && r.w * W > min_ratio * same->w * W) return r;
    return std::nullopt;
}

}  // namespace memecam::vision
