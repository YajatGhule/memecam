// Tracker::process — memecam/tracking.py Tracker.process for a single person, on top of the LiteRT pipelines.
#include "vision/tracker.h"

#include <cstdio>
#include <cstdlib>
#include <future>
#include <sstream>
#include <thread>
#include <tuple>

#include <opencv2/imgproc.hpp>

#include "tracking.h"
#include "vision/pipelines.h"

namespace memecam {

using namespace vision;

namespace {
int model_threads() {
    if (const char* e = std::getenv("MEMECAM_VISION_THREADS")) {
        const int n = std::atoi(e);
        if (n > 0) return n;
    }
    return 2;   // three pipelines run concurrently; 2 XNNPACK threads each fills 6-8 cores
}

constexpr int RESCAN_FACE = 15;   // frames between checks for a bigger face than the tracked one (~4 ms each)
constexpr int RESCAN_POSE = 30;   // same for the body (~24 ms each, on the async pose thread)
}  // namespace

// Single-person build: one hand task (2 hands), one face mesh and one body pose, tracked frame to frame.
// MediaPipe only re-runs its palm / face / person detectors when it tracks fewer instances than requested;
// the face and pose trackers additionally re-scan now and then for a bigger (closer) person, so the
// tracker never stays locked on someone in the background.
struct Tracker::Impl {
    bool pose_async;
    double min_face_width;
    Identities ids;
    std::string device = "cpu";

    std::unique_ptr<HandModels> hand_models;
    std::unique_ptr<FaceModels> face_models;
    std::unique_ptr<PoseModels> pose_models;
    std::unique_ptr<HandTracker> hands;
    std::unique_ptr<FaceTracker> face;
    std::unique_ptr<PoseTracker> pose;

    std::optional<std::future<std::vector<Landmarks>>> pose_job;
    int pose_w = 0, pose_h = 0;
    std::vector<std::shared_ptr<const Pose>> last_poses;
    cv::Mat prev_gray;
    TrackerRaw raw;

    Impl(const Config& cfg, bool use_pose, bool pose_async_)
        : pose_async(pose_async_), min_face_width(cfg.min_face_width), ids(ID_MAX_JUMP, cfg.forget) {
        const std::string dir = project_path("models");
        const int threads = model_threads();
        hand_models = std::make_unique<HandModels>(dir, threads);
        face_models = std::make_unique<FaceModels>(dir, threads);
        // MEMECAM_HAND_MP=1: MediaPipe's own hand graph, without the pose-seeded crops, prediction, smoothing
        // and pose handedness (for comparisons with the Python app).
        // MEMECAM_HAND_OFF=predict,smooth,...: switch single additions off.
        const char* mp = std::getenv("MEMECAM_HAND_MP");
        HandOptions ho = mp && *mp == '1' ? HandOptions::mediapipe() : HandOptions{};
        if (const char* off = std::getenv("MEMECAM_HAND_OFF")) {
            const std::string o = off;
            if (o.find("seeds") != std::string::npos) ho.pose_seeds = false;
            if (o.find("predict") != std::string::npos) ho.predict = false;
            if (o.find("smooth") != std::string::npos) ho.smooth = false;
            if (o.find("handed") != std::string::npos) ho.pose_handedness = false;
        }
        hands = std::make_unique<HandTracker>(*hand_models, 2, ho);
        face = std::make_unique<FaceTracker>(*face_models, 1, RESCAN_FACE);
        if (use_pose) {
            pose_models = std::make_unique<PoseModels>(dir, threads);
            pose = std::make_unique<PoseTracker>(*pose_models, 1, RESCAN_POSE);
        }
        std::string mode = cfg.device;
        for (auto& c : mode) c = static_cast<char>(std::tolower(c));
        const char* why = mode == "cpu" ? "forced by config"
                          : mode == "gpu" ? "GPU requested, but this build has no LiteRT GPU delegate"
                                          : "LiteRT XNNPACK; no GPU delegate in this build";
        std::printf("[memecam] tracking on CPU (%s)\n", why);
        std::fflush(stdout);
    }

    ~Impl() {
        if (pose_job) pose_job->wait();
    }

    Frame process(const cv::Mat& rgb, int64_t ts_ms) {
        const int h = rgb.rows, w = rgb.cols;
        const double now = ts_ms / 1000.0;
        raw = TrackerRaw{};

        // The body pose of the latest finished pose job (usually the previous frame) guides the hand tracker.
        std::vector<PoseHand> arms;
        if (!last_poses.empty()) {
            const Pose& p = *last_poses[0];
            auto pt = [&](int i) { return cv::Point2f(float(p[i].x), float(p[i].y)); };
            const float sw = float(cv::norm(pt(11) - pt(12)));
            // Pose landmarks 15/17/19 are the subject's left wrist / pinky / index: in the mirrored frame that
            // is the hand the hand model calls "Left" (checked on the recorded clips: 96 % agreement).
            for (auto [w, pk, ix, label] : {std::tuple{15, 17, 19, "Left"}, std::tuple{16, 18, 20, "Right"}})
                arms.push_back({pt(w), pt(pk), pt(ix),
                                float(std::min({p[w].visibility, p[pk].visibility, p[ix].visibility})), sw, label});
        }
        auto hands_job = std::async(std::launch::async, [&] { return hands->process(rgb, ts_ms, arms); });
        auto face_job = std::async(std::launch::async, [&] { return face->process(rgb, ts_ms); });
        if (pose && !pose_job) {
            // The pose job may outlive this frame: give it its own reference to the pixels.
            cv::Mat img = pose_async ? rgb.clone() : rgb;
            PoseTracker* p = pose.get();
            pose_job = std::async(std::launch::async, [p, img, ts_ms] { return p->process(img, ts_ms); });
            pose_w = w;
            pose_h = h;
        }

        for (auto& r : hands_job.get()) {
            Hand hand;
            for (int i = 0; i < 21; ++i)
                hand.pts[i] = Vec3(double(r.lms[i].x) * w, double(r.lms[i].y) * h, double(r.lms[i].z) * w);
            hand.label = r.label;
            hand.gesture = r.gesture;
            hand.gesture_score = r.gesture_score;
            raw.hands.push_back(std::move(hand));
        }
        for (auto& l : face_job.get()) {
            Face f;
            f.pts.reserve(l.size());
            for (auto& p : l) f.pts.emplace_back(double(p.x) * w, double(p.y) * h, double(p.z) * w);
            raw.faces.push_back(std::move(f));
        }
        if (pose_job && (!pose_async ||
                         pose_job->wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
            auto res = pose_job->get();
            pose_job.reset();
            last_poses.clear();
            for (auto& l : res) {
                auto p = std::make_shared<Pose>();
                for (int i = 0; i < 33; ++i)
                    (*p)[i] = {double(l[i].x) * pose_w, double(l[i].y) * pose_h, double(l[i].z) * pose_w,
                               double(l[i].visibility)};
                last_poses.push_back(std::move(p));
            }
            raw.pose_fresh = true;
        }
        raw.poses = last_poses;

        std::vector<Frame> people = group_people(raw.faces, last_poses, raw.hands, w, 1, min_face_width,
                                                 ids.recent(now));
        ids.assign(people, now);

        // Frame differencing: fast-moving hands blur and drop out of the hand tracker, but they
        // still light up the difference image (used for "talking to a wall").
        cv::Mat gray = motion_gray(rgb);
        std::shared_ptr<const cv::Mat> motion;
        if (!prev_gray.empty()) {
            cv::Mat diff, mask;
            cv::absdiff(gray, prev_gray, diff);
            mask = diff > 20;
            motion = std::make_shared<const cv::Mat>(std::move(mask));
        }
        prev_gray = gray;
        for (auto& q : people) {
            q.motion = motion;
            q.width = w;
            q.height = h;
        }
        Frame out;
        if (!people.empty()) {
            out.hands = people[0].hands;
            out.face = people[0].face;
            out.pose = people[0].pose;
            out.pid = people[0].pid;
        }
        out.width = w;
        out.height = h;
        out.motion = motion;
        out.renamed = ids.renamed;
        out.people = std::move(people);
        return out;
    }
};

Tracker::Tracker(const Config& cfg, bool use_pose, bool pose_async)
    : impl_(std::make_unique<Impl>(cfg, use_pose, pose_async)) {}
Tracker::~Tracker() = default;
Frame Tracker::process(const cv::Mat& rgb, int64_t ts_ms) { return impl_->process(rgb, ts_ms); }
std::string Tracker::device() const { return impl_->device; }
const TrackerRaw& Tracker::last_raw() const { return impl_->raw; }

std::string Tracker::stats() const {
    std::ostringstream s;
    auto line = [&](const char* name, const StageStats& st, const char* det, const char* extra) {
        const long calls = st.calls.load();
        if (!calls) return;
        const long dr = std::max(1L, st.det_runs.load()), lr = std::max(1L, st.lm_runs.load());
        s.setf(std::ios::fixed);
        s.precision(1);
        s << name << ": " << st.total_ms.load() / calls << " ms/frame (" << det << " " << st.det_ms.load() / dr
          << " ms x" << st.det_runs.load() << ", landmarks " << st.lm_ms.load() / calls << " ms/frame, "
          << st.lm_ms.load() / lr << " ms/run x" << st.lm_runs.load();
        if (extra) s << ", " << extra << " " << st.extra_ms.load() / calls << " ms/frame";
        s << ", " << calls << " frames)\n";
    };
    line("hands", impl_->hand_models->stats, "palm", "gesture");
    line("face", impl_->face_models->stats, "detector", nullptr);
    if (impl_->pose_models) line("pose", impl_->pose_models->stats, "detector", nullptr);
    return s.str();
}

}  // namespace memecam
