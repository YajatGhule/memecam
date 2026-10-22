#include "overlays.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <tuple>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>   // OpenCV 5: getPerspectiveTransform moved out of imgproc
#endif
#ifdef MEMECAM_HAVE_FREETYPE
#include <opencv2/freetype.hpp>
#endif

// cv::imreadanimation (frames + per-frame durations + alpha) arrived in OpenCV 4.11; older OpenCV (e.g.
// Ubuntu / Debian packages) falls back to reading the GIF as a video at its nominal frame rate.
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 11)
#define MEMECAM_HAVE_ANIMATION 1
#endif

namespace fs = std::filesystem;

namespace memecam {

namespace {

int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }   // Python // for b > 0

cv::Mat to_bgra(const cv::Mat& in) {
    cv::Mat m = in;
    if (m.depth() == CV_16U) m.convertTo(m, CV_8U, 1.0 / 257);
    cv::Mat out;
    switch (m.channels()) {
        case 1: cv::cvtColor(m, out, cv::COLOR_GRAY2BGRA); break;
        case 3: cv::cvtColor(m, out, cv::COLOR_BGR2BGRA); break;
        default: out = m.clone(); break;
    }
    return out;
}

// Pillow Image.thumbnail((max_side, max_side)) target size: only shrinks, aspect kept with Pillow's rounding.
std::optional<cv::Size> thumbnail_size(cv::Size s, int max_side) {
    int x = max_side, y = max_side;
    if (x >= s.width && y >= s.height) return std::nullopt;
    double aspect = double(s.width) / s.height;
    auto round_aspect = [](double number, auto key) {
        double lo = std::floor(number), hi = std::ceil(number);
        return std::max(int(key(hi) < key(lo) ? hi : lo), 1);   // min(floor, ceil, key=key): ties -> floor
    };
    if (double(x) / y >= aspect)
        x = round_aspect(y * aspect, [&](double n) { return std::abs(aspect - n / y); });
    else
        y = round_aspect(x / aspect, [&](double n) { return n == 0 ? 0.0 : std::abs(aspect - x / n); });
    if (x == s.width && y == s.height) return std::nullopt;
    return cv::Size(x, y);
}

// Small LRU of derived images keyed on (source pixels, extra key); holds the source so its pointer stays valid.
class Cache {
public:
    explicit Cache(size_t max_bytes = size_t(96) << 20) : max_bytes_(max_bytes) {}
    template <class Make>
    cv::Mat get(const cv::Mat& src, std::tuple<int, int, int, int> key, Make make) {
        std::lock_guard lock(mu_);
        Key k{src.data, key};
        for (auto it = items_.begin(); it != items_.end(); ++it) {
            if (it->key == k && it->src.data == src.data && it->src.size() == src.size()) {
                items_.splice(items_.end(), items_, it);   // most recently used last
                return it->out;
            }
        }
        cv::Mat out = make();
        items_.push_back({k, src, out});
        bytes_ += out.total() * out.elemSize();
        while (bytes_ > max_bytes_ && items_.size() > 1) {
            bytes_ -= items_.front().out.total() * items_.front().out.elemSize();
            items_.pop_front();
        }
        return out;
    }
private:
    struct Key {
        const uchar* data;
        std::tuple<int, int, int, int> extra;
        bool operator==(const Key&) const = default;
    };
    struct Item { Key key; cv::Mat src, out; };
    std::mutex mu_;
    std::list<Item> items_;   // linear scan: a few hundred entries at most, far cheaper than the resize
    size_t max_bytes_, bytes_ = 0;
};

Cache& fit_cache() { static Cache c; return c; }
Cache& panel_cache() { static Cache c; return c; }

// dst[roi] = src * a + dst * (255 - a), in saturating uint8 ops like the Python version (±1 vs float).
void blend_roi(cv::Mat roi, const cv::Mat& bgr, const cv::Mat& a) {
    cv::Mat a3, inv, fg, bg;
    cv::merge(std::vector<cv::Mat>{a, a, a}, a3);
    cv::multiply(bgr, a3, fg, 1.0 / 255);
    cv::bitwise_not(a3, inv);
    cv::multiply(roi, inv, bg, 1.0 / 255);
    cv::add(fg, bg, roi);
}

// ---- bold TTF text ---------------------------------------------------------------------------------

// Arch, then Debian / Ubuntu, then Fedora font locations.
const char* BOLD_FONTS[] = {"/usr/share/fonts/noto/NotoSans-Black.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                            "/usr/share/fonts/truetype/noto/NotoSans-Black.ttf",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                            "/usr/share/fonts/google-noto/NotoSans-Black.ttf",
                            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"};

#ifdef MEMECAM_HAVE_FREETYPE
cv::Ptr<cv::freetype::FreeType2> bold_font() {
    static cv::Ptr<cv::freetype::FreeType2> ft = [] {
        cv::Ptr<cv::freetype::FreeType2> f;
        for (const char* path : BOLD_FONTS) {
            if (!fs::exists(path)) continue;
            try {
                f = cv::freetype::createFreeType2();
                f->loadFontData(path, 0);
                return f;
            } catch (const cv::Exception&) { f.release(); }
        }
        return f;
    }();
    return ft;
}
#endif

std::mutex text_mu;

// Render `text` into 1-channel coverage masks (fill, and fill+stroke) on a padded canvas; returns the canvas
// origin offset of the fill's ink box.
struct TextMasks { cv::Mat fill, outer; cv::Rect ink; };

cv::Rect ink_box(const cv::Mat& mask) {
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    if (nz.empty()) return {};
    int x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
    for (const auto& p : nz) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
    return {x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

TextMasks text_masks(const std::string& text, int size, int stroke) {
    std::lock_guard lock(text_mu);
    TextMasks m;
    int pad = stroke + 4;
#ifdef MEMECAM_HAVE_FREETYPE
    if (auto ft = bold_font()) {
        int base = 0;
        cv::Size sz = ft->getTextSize(text, size, -1, &base);
        cv::Size canvas(sz.width + 2 * size + 2 * pad, 2 * size + 2 * pad);
        cv::Point org(size / 2 + pad, int(1.4 * size) + pad);   // baseline (bottomLeftOrigin = true)
        m.fill = cv::Mat::zeros(canvas, CV_8UC1);
        ft->putText(m.fill, text, org, size, cv::Scalar(255), -1, cv::LINE_AA, true);
        m.outer = m.fill.clone();
        if (stroke > 0) {   // Pillow's stroke_width extends `stroke` px outwards: an outline 2*stroke thick
            cv::Mat o = cv::Mat::zeros(canvas, CV_8UC1);
            ft->putText(o, text, org, size, cv::Scalar(255), 2 * stroke, cv::LINE_AA, true);
            cv::max(m.outer, o, m.outer);
        }
        m.ink = ink_box(m.fill);
        return m;
    }
#endif
    // Fallback: OpenCV's built-in font, scaled so its cap height roughly matches the TTF.
    double scale = size / 30.0;
    int th = std::max(1, size / 10), base = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, th, &base);
    cv::Size canvas(sz.width + 2 * pad + 2 * th, sz.height + base + 2 * pad + 2 * th);
    cv::Point org(pad + th, pad + th + sz.height);
    m.fill = cv::Mat::zeros(canvas, CV_8UC1);
    cv::putText(m.fill, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255), th, cv::LINE_AA);
    m.outer = m.fill.clone();
    if (stroke > 0)
        cv::putText(m.outer, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255), th + 2 * stroke,
                    cv::LINE_AA);
    m.ink = ink_box(m.fill);
    return m;
}

}  // namespace

void draw_bold_text(cv::Mat& bgra, const std::string& text, int cx, int top, int size, int stroke) {
    CV_Assert(bgra.type() == CV_8UC4);
    if (text.empty()) return;
    TextMasks t = text_masks(text, size, stroke);
    if (t.ink.empty()) return;
    // Canvas -> destination offset: the fill ink's top-centre lands on (cx, top).
    int dx = cx - (t.ink.x + t.ink.width / 2), dy = top - t.ink.y;
    for (int y = 0; y < t.outer.rows; y++) {
        int Y = y + dy;
        if (Y < 0 || Y >= bgra.rows) continue;
        const uchar* fo = t.outer.ptr<uchar>(y);
        const uchar* ff = t.fill.ptr<uchar>(y);
        auto* d = bgra.ptr<cv::Vec4b>(Y);
        for (int x = 0; x < t.outer.cols; x++) {
            int X = x + dx;
            if (X < 0 || X >= bgra.cols || !fo[x]) continue;
            // Source: white fill over a black stroke, alpha = stroke coverage (premultiplied colour = fill).
            float a = fo[x] / 255.f, c = ff[x] / 255.f;   // c = premultiplied white
            cv::Vec4b& px = d[X];
            float da = px[3] / 255.f, oa = a + da * (1 - a);
            for (int k = 0; k < 3; k++) {
                float v = (c * 255.f + px[k] * da * (1 - a)) / std::max(oa, 1e-6f);
                px[k] = cv::saturate_cast<uchar>(v);
            }
            px[3] = cv::saturate_cast<uchar>(oa * 255.f);
        }
    }
}

cv::Mat text_card(const std::string& label) {
    std::string text = label;
    std::replace(text.begin(), text.end(), '_', ' ');
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return std::toupper(ch); });
    cv::Mat img(260, 420, CV_8UC4, cv::Scalar(20, 20, 20, 255));
    // Pillow multiline_text(anchor="mm", align="center"): each line centred, the block centred on (210, 130).
    std::vector<std::string> lines;
    size_t s = 0;
    for (size_t e; (e = text.find('\n', s)) != std::string::npos; s = e + 1) lines.push_back(text.substr(s, e - s));
    lines.push_back(text.substr(s));
    int line_h = 34 + 4 + 10, top = 130 - int(lines.size()) * line_h / 2 + 10;
    for (size_t i = 0; i < lines.size(); i++) draw_bold_text(img, lines[i], 210, top + int(i) * line_h, 34);
    return img;
}

// ---- Media -----------------------------------------------------------------------------------------

Media::Media(const std::string& path, int max_side, bool key_black, const std::string& label,
             std::optional<std::pair<int, int>> trim, std::optional<int> loop_from) {
    std::vector<cv::Mat> raw;
    std::vector<int> durs;   // ms
    if (fs::exists(path)) {
        std::string ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        bool animated = false;
        if (ext == ".gif" || ext == ".webp" || ext == ".apng") {
#ifdef MEMECAM_HAVE_ANIMATION
            cv::Animation anim;
            try {
                animated = cv::imreadanimation(path, anim) && !anim.frames.empty();
            } catch (const cv::Exception&) { animated = false; }
            if (animated) {
                raw = std::move(anim.frames);
                durs = std::move(anim.durations);
            }
#else
            cv::VideoCapture vc(path);
            const double fps = vc.get(cv::CAP_PROP_FPS);
            const int dur = fps >= 1 && fps <= 100 ? int(1000 / fps) : 80;
            for (cv::Mat f; vc.read(f) && !f.empty();) {
                raw.push_back(f.clone());
                durs.push_back(dur);
            }
            animated = raw.size() > 1;
            if (!animated) raw.clear(), durs.clear();
#endif
        }
        if (!animated) {
            cv::Mat im = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (!im.empty()) {
                raw.push_back(im);
                durs.push_back(80);   // Pillow: no "duration" in a still image -> default 80 ms
            }
        }
    }
    if (raw.empty()) {
        std::printf("[memecam] missing asset %s, using a text card\n", fs::path(path).filename().c_str());
        missing_ = true;
        frames_.push_back(text_card(label.empty() ? fs::path(path).stem().string() : label));
        ends_.push_back(1.0);
        return;
    }
    auto [first, last] = trim.value_or(std::pair<int, int>{0, 1 << 30});
    double t = 0.0;
    for (int k = 0; k < int(raw.size()); k++) {
        if (k < first || k > last) continue;
        if (loop_from && k == *loop_from) loop_start_ = t;
        cv::Mat a = to_bgra(raw[k]);
        if (auto sz = thumbnail_size(a.size(), max_side)) cv::resize(a, a, *sz, 0, 0, cv::INTER_AREA);
        if (key_black) {
            // Footage on pure black (scuba cat): turn the black into transparency.
            for (int y = 0; y < a.rows; y++) {
                auto* p = a.ptr<cv::Vec4b>(y);
                for (int x = 0; x < a.cols; x++) {
                    int lum = std::max({p[x][0], p[x][1], p[x][2]});
                    p[x][3] = cv::saturate_cast<uchar>(std::clamp((lum - 18) * 8, 0, 255));
                }
            }
        }
        frames_.push_back(a);
        int d = k < int(durs.size()) ? durs[k] : 80;
        t += std::max(d, 20) / 1000.0;
        ends_.push_back(t);
    }
    if (frames_.empty()) {   // trim outside the clip: keep the first frame rather than crash
        frames_.push_back(to_bgra(raw[0]));
        ends_.push_back(0.08);
        loop_start_.reset();
    }
}

const cv::Mat& Media::at(double t) const {
    double dur = duration();
    if (loop_start_ && t >= dur) {
        double span = dur - *loop_start_;
        if (span > 0) {
            double phase = std::fmod(t - dur, 2 * span);
            if (phase < 0) phase += 2 * span;
            t = phase < span ? dur - phase : *loop_start_ + (phase - span);
            t = std::min(t, dur - 1e-6);
        }
    }
    t = std::fmod(t, dur);
    if (t < 0) t += dur;
    size_t i = std::upper_bound(ends_.begin(), ends_.end(), t) - ends_.begin();
    return frames_[std::min(i, frames_.size() - 1)];
}

// ---- compositing -----------------------------------------------------------------------------------

void blend(cv::Mat& dst, const cv::Mat& src, int x, int y, double alpha) {
    int h = src.rows, w = src.cols, H = dst.rows, W = dst.cols;
    int x0 = std::max(x, 0), y0 = std::max(y, 0), x1 = std::min(x + w, W), y1 = std::min(y + h, H);
    if (x0 >= x1 || y0 >= y1) return;
    cv::Mat s = src(cv::Rect(x0 - x, y0 - y, x1 - x0, y1 - y0));
    cv::Mat bgr, a;
    cv::extractChannel(s, a, 3);
    if (alpha < 1.0) cv::convertScaleAbs(a, a, alpha);
    cv::cvtColor(s, bgr, cv::COLOR_BGRA2BGR);
    blend_roi(dst(cv::Rect(x0, y0, x1 - x0, y1 - y0)), bgr, a);
}

cv::Mat fit(const cv::Mat& img, int height) {
    return fit_cache().get(img, {height, 0, 0, 0}, [&] {
        cv::Mat out;
        int w = std::max(1, int(double(img.cols) * height / img.rows));
        cv::resize(img, out, cv::Size(w, height), 0, 0, cv::INTER_AREA);
        return out;
    });
}

void panel(cv::Mat& dst, const cv::Mat& img, int x, int y, const std::string& title, double alpha,
           cv::Scalar accent) {
    cv::Mat framed = panel_cache().get(img, {int(accent[0]), int(accent[1]), int(accent[2]), 1}, [&] {
        cv::Mat f(img.rows + 6, img.cols + 6, CV_8UC4, cv::Scalar(accent[0], accent[1], accent[2], 255));
        img.copyTo(f(cv::Rect(3, 3, img.cols, img.rows)));
        return f;
    });
    blend(dst, framed, x, y, alpha);
    if (!title.empty() && alpha > 0.5) {
        int base = 0;
        cv::Size ts = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(dst, cv::Point(x, y - ts.height - 10), cv::Point(x + ts.width + 10, y), accent, -1);
        cv::putText(dst, title, cv::Point(x + 5, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(15, 15, 15), 1,
                    cv::LINE_AA);
    }
}

cv::Point place_beside(cv::Size frame, const Box& box, int w, int h, int margin) {
    int W = frame.width, H = frame.height;
    int x0 = box.x, y0 = box.y, x1 = x1_of(box), y1 = y1_of(box);
    int x = W - x1 > x0 ? x1 + margin : x0 - margin - w;
    int y = floordiv(y0 + y1, 2) - floordiv(h, 2);
    return {std::clamp(x, 0, std::max(0, W - w)), std::clamp(y, 30, std::max(30, H - h))};
}

void over_face(cv::Mat& dst, const cv::Mat& img, const Box& head, double alpha, double grow) {
    int x0 = head.x, y0 = head.y, x1 = x1_of(head), y1 = y1_of(head);
    double target_w = std::max(16.0, (x1 - x0) * grow);
    int h = std::max(16, int(target_w * img.rows / img.cols) / 8 * 8);   // quantised so the fit cache hits
    cv::Mat small = fit(img, h);
    int w = small.cols;
    int cx = floordiv(x0 + x1, 2), cy = floordiv(y0 + y1, 2) - int(0.08 * (y1 - y0));
    blend(dst, small, cx - floordiv(w, 2), cy - floordiv(h, 2), alpha);
}

namespace {
// Feathered oval mask, cached per meme size (the big blur is the expensive part).
const cv::Mat& oval(int h, int w) {
    static std::mutex mu;
    static std::map<std::pair<int, int>, cv::Mat> cache;
    std::lock_guard lock(mu);
    auto& m = cache[{h, w}];
    if (m.empty()) {
        m = cv::Mat::zeros(h, w, CV_8UC1);
        cv::ellipse(m, cv::Point(w / 2, h / 2), cv::Size(int(w * 0.46), int(h * 0.46)), 0, 0, 360, cv::Scalar(255),
                    -1);
        cv::GaussianBlur(m, m, cv::Size(0, 0), std::max(w, h) * 0.04);
    }
    return m;
}
}  // namespace

void warp_on_face(cv::Mat& dst, const cv::Mat& img, const std::vector<Vec3>& face_pts, double alpha, double grow) {
    if (face_pts.size() <= 365) return;
    std::array<cv::Point2f, 4> quad;
    cv::Point2f c(0, 0);
    for (int i = 0; i < 4; i++) {
        quad[i] = xy(face_pts[FACE_QUAD[i]]);
        c += quad[i];
    }
    c *= 0.25f;
    float mnx = 1e9f, mny = 1e9f, mxx = -1e9f, mxy = -1e9f;
    for (auto& q : quad) {
        q = (q - c) * float(grow) + c;
        mnx = std::min(mnx, q.x); mny = std::min(mny, q.y);
        mxx = std::max(mxx, q.x); mxy = std::max(mxy, q.y);
    }
    // Only warp/blend the quad's bounding box: doing it full-frame is far slower.
    int H = dst.rows, W = dst.cols;
    int x0 = std::clamp(int(std::floor(mnx)), 0, W), y0 = std::clamp(int(std::floor(mny)), 0, H);
    int x1 = std::clamp(int(std::ceil(mxx)), 0, W), y1 = std::clamp(int(std::ceil(mxy)), 0, H);
    if (x0 >= x1 || y0 >= y1) return;
    float h = float(img.rows), w = float(img.cols);
    cv::Point2f src[4] = {{0, 0}, {w, 0}, {w, h}, {0, h}};
    cv::Point2f to[4];
    for (int i = 0; i < 4; i++) to[i] = quad[i] - cv::Point2f(float(x0), float(y0));
    cv::Mat M = cv::getPerspectiveTransform(src, to);
    cv::Size size(x1 - x0, y1 - y0);
    cv::Mat warped, mask, wa, a, bgr;
    cv::warpPerspective(img, warped, M, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::warpPerspective(oval(img.rows, img.cols), mask, M, size);
    cv::extractChannel(warped, wa, 3);
    cv::multiply(mask, wa, a, alpha / 255);
    cv::cvtColor(warped, bgr, cv::COLOR_BGRA2BGR);
    blend_roi(dst(cv::Rect(x0, y0, size.width, size.height)), bgr, a);
}

const cv::Mat& CinemaEffect::caption(int width) {
    auto& cap = cache_[width];
    if (cap.empty()) {
        cap = cv::Mat::zeros(width / 5, width, CV_8UC4);
        draw_bold_text(cap, "ABSOLUTE", width / 2, width / 32, width / 22, 2);
        draw_bold_text(cap, "CINEMA", width / 2, width / 13, width / 9, 3);
    }
    return cap;
}

void CinemaEffect::apply(cv::Mat& dst, double alpha) {
    cv::Mat gray, g3;
    cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, g3, cv::COLOR_GRAY2BGR);
    cv::convertScaleAbs(g3, g3, 1.25, -25);   // punchier contrast
    cv::addWeighted(g3, alpha, dst, 1 - alpha, 0, dst);
    const cv::Mat& cap = caption(int(dst.cols * 0.6));
    blend(dst, cap, (dst.cols - cap.cols) / 2, dst.rows - cap.rows - 30, alpha);
}

}  // namespace memecam
