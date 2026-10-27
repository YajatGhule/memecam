// Meme Cam: webcam + mocap markers + gesture-triggered memes (port of memecam/main.py main()).
// Built twice: `memecam` and `memecamv2` (MEMECAM_V2: config_v2.toml layered over config.toml).
//
// Keys: d = toggle markers/bounding boxes, f = toggle meme overlays, m = mute meme sounds, i = gesture debug scores, r = record,
//       c = recalibrate head pose, q / Esc = quit
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#define fdopen _fdopen
static constexpr const char* NULL_DEVICE = "NUL";
#else
#include <fcntl.h>
#include <unistd.h>
static constexpr const char* NULL_DEVICE = "/dev/null";
#endif

#include <CLI/CLI.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "audio.h"
#include "compose.h"
#include "config.h"
#include "crowd.h"
#include "mocap.h"
#include "overlays.h"
#include "vision/tracker.h"

namespace fs = std::filesystem;
using namespace memecam;
using Clock = std::chrono::steady_clock;

#ifdef MEMECAM_V2
static constexpr const char* TITLE = "Meme Cam v2";
#else
static constexpr const char* TITLE = "Meme Cam";
#endif

namespace {

FILE* g_err = stderr;                      // our own errors; stays visible when native logs are hidden
std::atomic<bool> g_interrupted{false};

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Hide the native log spam (LiteRT/XNNPACK, OpenCV, Qt/GTK) that goes straight to the stderr file
// descriptor, while keeping our own error messages visible (`--verbose` keeps everything).
void quiet_native_logs() {
    set_env("TF_CPP_MIN_LOG_LEVEL", "3");
    set_env("GLOG_minloglevel", "3");
    set_env("OPENCV_LOG_LEVEL", "SILENT");
    set_env("OPENCV_FFMPEG_LOGLEVEL", "-8");
    set_env("QT_LOGGING_RULES", "*=false");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    int keep = dup(2);
    FILE* null_file = std::fopen(NULL_DEVICE, "w");
    if (keep < 0 || !null_file) return;
    dup2(fileno(null_file), 2);
    std::fclose(null_file);
    if (FILE* f = fdopen(keep, "w")) {
#ifdef _WIN32
        setvbuf(f, nullptr, _IONBF, 0);   // MSVC has no line buffering
#else
        setvbuf(f, nullptr, _IOLBF, 0);
#endif
        g_err = f;
    }
}

// False once the user closed the window with its X button.
bool window_open(const std::string& name) {
    try {
        return cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1;
    } catch (const cv::Exception&) {
        return false;
    }
}

// Timestamps for a video file: the container's frame times, so holds/min_show behave like real time.
// Falls back to frame_idx / fps when the backend doesn't report per-frame times.
class VideoClock {
public:
    explicit VideoClock(cv::VideoCapture& cap) : cap_(cap) {
        double f = cap.get(cv::CAP_PROP_FPS);
        fps = (f >= 1 && f <= 240) ? f : 30.0;
    }
    double next() {
        double t = cap_.get(cv::CAP_PROP_POS_MSEC) / 1000;
        if (!(t > last_)) t = std::max(idx_ / fps, last_ + 1 / fps);
        idx_++;
        last_ = t;
        return t;
    }
    double fps;
private:
    cv::VideoCapture& cap_;
    long idx_ = 0;
    double last_ = -1.0;
};

// Reads the camera (or a video file paced to its frame rate) on a background thread and keeps only the
// newest mirrored frame, so JPEG decoding never blocks the display loop and frames never queue up.
class Grabber {
public:
    struct Got { cv::Mat frame; double t; long seq; };   // `frame` is shared read-only with other threads

    Grabber(cv::VideoCapture& cap, bool mirror, Clock::time_point t0, std::optional<double> pace, bool loop)
        : cap_(cap), mirror_(mirror), t0_(t0), pace_(pace), loop_(loop), thread_([this] { run(); }) {}
    ~Grabber() { stop(); }

    // Newest frame with seq > `after`; nullopt on timeout or once the source ended.
    std::optional<Got> wait(long after, double timeout = 1.0) {
        std::unique_lock lock(mu_);
        cv_.wait_for(lock, std::chrono::duration<double>(timeout),
                     [&] { return seq_ > after || ended_ || stopping_; });
        if (seq_ <= after) return std::nullopt;
        return Got{frame_, t_, seq_};
    }
    bool ended() {
        std::lock_guard lock(mu_);
        return ended_;
    }
    void stop() {
        {
            std::lock_guard lock(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        try {
            long n = 0;
            while (!stopping_) {
                if (pace_) {   // video file: deliver frames like a webcam would
                    auto due = t0_ + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::duration<double>((n + 1) * *pace_));
                    std::this_thread::sleep_until(due);
                }
                cv::Mat raw;   // a fresh buffer every time: the previous frame may still be in use
                bool ok = cap_.read(raw);
                if (!ok && loop_ && cap_.set(cv::CAP_PROP_POS_FRAMES, 0)) ok = cap_.read(raw);
                if (!ok || raw.empty()) break;
                double t = since(t0_);
                cv::Mat frame;
                if (mirror_) cv::flip(raw, frame, 1);
                else frame = raw;
                {
                    std::lock_guard lock(mu_);
                    frame_ = frame, t_ = t, seq_++;
                }
                n++;
                cv_.notify_all();
            }
        } catch (const std::exception& e) {
            std::fprintf(g_err, "[memecam] capture failed: %s\n", e.what());
        }
        {
            std::lock_guard lock(mu_);
            ended_ = true;
        }
        cv_.notify_all();
    }

    cv::VideoCapture& cap_;
    bool mirror_;
    Clock::time_point t0_;
    std::optional<double> pace_;
    bool loop_;
    std::mutex mu_;
    std::condition_variable cv_;
    cv::Mat frame_;
    double t_ = 0;
    long seq_ = 0;
    bool ended_ = false;
    std::atomic<bool> stopping_{false};
    std::thread thread_;   // last: started after every other member is initialised
};

// One tracking result, published as a whole: the frame's tracking, the person's detections, and a
// consistent copy of their memes (taken on the tracking thread).
struct Published {
    Frame fr;
    std::map<int, std::map<std::string, Detection>> dets;
    std::vector<Shown> shown;
    long count = 0;
};

// Runs Tracker::process + Crowd::update (the tracked person's detectors and memes) on a background thread, always on the newest grabbed frame,
// independent of the display. Meme trigger state is only ever touched by this thread.
class LiveTracker {
public:
    LiveTracker(Grabber& grabber, Tracker& tracker, Crowd& crowd, int width, int height,
                std::vector<std::string> force)
        : grabber_(grabber), tracker_(tracker), crowd_(crowd), force_(std::move(force)) {
        auto p = std::make_shared<Published>();
        p->fr.width = width, p->fr.height = height;
        p->shown = crowd.snapshot();
        latest_ = p;
        thread_ = std::thread([this] { run(); });
    }
    ~LiveTracker() { stop(); }

    std::shared_ptr<const Published> latest() {
        if (error_) std::rethrow_exception(error_);
        std::lock_guard lock(mu_);
        return latest_;
    }
    void calibrate() { want_calibrate_ = true; }
    void stop() {
        stopping_ = true;
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        long seq = 0, count = 0;
        int64_t last_ts = -1;
        try {
            cv::Mat rgb;
            while (!stopping_) {
                auto got = grabber_.wait(seq, 0.2);
                if (!got) {
                    if (grabber_.ended()) break;
                    continue;
                }
                seq = got->seq;
                if (want_calibrate_.exchange(false)) crowd_.calibrate();
                int64_t ts = std::max<int64_t>(int64_t(got->t * 1000), last_ts + 1);
                last_ts = ts;
                cv::cvtColor(got->frame, rgb, cv::COLOR_BGR2RGB);
                auto p = std::make_shared<Published>();
                p->fr = tracker_.process(rgb, ts);
                p->dets = crowd_.update(p->fr, got->t, force_);
                p->shown = crowd_.snapshot();
                p->count = ++count;
                std::lock_guard lock(mu_);
                latest_ = std::move(p);
            }
        } catch (...) {
            error_ = std::current_exception();   // re-raised on the display thread
        }
    }

    Grabber& grabber_;
    Tracker& tracker_;
    Crowd& crowd_;
    std::vector<std::string> force_;
    std::mutex mu_;
    std::shared_ptr<const Published> latest_;
    std::exception_ptr error_;
    std::atomic<bool> want_calibrate_{false}, stopping_{false};
    std::thread thread_;
};

std::string join_names(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? ", '" : "'") + v[i] + "'";
    return s + "]";
}

std::string recording_path() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    fs::path dir = fs::path(home ? home : ".") / "Videos";
    fs::create_directories(dir);
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
    return (dir / (std::string("memecam-") + stamp + ".mp4")).string();
}

struct Args {
    std::optional<int> camera;
    std::string video, snapshot, config;
    bool realtime = false, no_pose = false, no_mirror = false, verbose = false, events = false, mute = false;
    int frames = 60;
    std::optional<double> bench;
    std::vector<std::string> force;
};

constexpr int MAX_CAMERA = 4;   // fallback / [n] switching tries camera indexes 0..4

// Opens camera `index` and reads one frame into `first`; leaves `cap` closed on failure.
bool open_camera(cv::VideoCapture& cap, int index, const Config& cfg, cv::Mat& first) {
    cap.release();
#ifdef _WIN32
    cap.open(index, cv::CAP_DSHOW);
#else
    cap.open(index, cv::CAP_V4L2);
#endif
    if (!cap.isOpened()) return false;
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    cap.set(cv::CAP_PROP_FPS, 30);
    // No CAP_PROP_BUFFERSIZE=1: with a single V4L2 buffer the camera skips every other frame (15 FPS).
    // Latency stays low anyway because the Grabber thread always drains to the newest frame.
    if (cap.read(first) && !first.empty()) return true;
    cap.release();
    return false;
}

int run(const Args& args) {
    Config cfg;
#ifdef MEMECAM_V2
    cfg = load_config(args.config, project_path("config/config_v2.toml"));
#else
    cfg = load_config(args.config);
#endif
    cv::VideoCapture cap;
    std::optional<VideoClock> clock;
    int W = 0, H = 0, camera = -1;   // camera: index in use (webcam mode)
    if (!args.video.empty()) {
        cap.open(args.video);
        if (!cap.isOpened()) {
            std::fprintf(g_err, "could not open video %s\n", args.video.c_str());
            return 1;
        }
        clock.emplace(cap);
        W = int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), H = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    } else {
        // MEMECAM_CAMERA=N overrides the config (a double-clicked .exe can't pass --camera)
        if (const char* e = std::getenv("MEMECAM_CAMERA"); e && *e && !args.camera) camera = std::atoi(e);
        else camera = args.camera.value_or(cfg.camera_index);
        // If that camera doesn't open, try the others (0..4) before giving up.
        cv::Mat first;
        if (!open_camera(cap, camera, cfg, first)) {
            for (int i = 0; i <= MAX_CAMERA && !cap.isOpened(); i++)
                if (i != camera && open_camera(cap, i, cfg, first)) {
                    std::printf("[memecam] camera %d did not open, using camera %d\n", camera, i);
                    camera = i;
                }
        }
        if (!cap.isOpened()) {
            std::fprintf(g_err, "could not read from the webcam — is another app using it?\n");
            return 1;
        }
        W = first.cols, H = first.rows;
    }
    const bool mirror = cfg.mirror && !args.no_mirror;

    // Live path: a grabber thread reads frames, a tracking thread runs the tracker + memes on the newest one,
    // and this loop shows every camera frame with the latest tracking result drawn on it.
    // --video (without --realtime) and --snapshot run synchronously: every frame is tracked, deterministic.
    const bool live = args.bench || (args.snapshot.empty() && !args.events && (args.video.empty() || args.realtime));
    const bool headless = !args.snapshot.empty() || args.bench || args.events;

    std::printf("[memecam] loading models + memes ...\n");
    std::fflush(stdout);
    // Offline (--video without --realtime, --snapshot, --events): pose runs in step with every frame, deterministic.
    auto tracker = std::make_unique<Tracker>(cfg, !args.no_pose, live);
    Crowd crowd(cfg);
    MemeMedia media = load_meme_media(cfg, project_path("assets"));
    const bool on_face = cfg.meme_on_face;
    CinemaEffect cinema;

    bool show_markers = true, show_debug = false, show_memes = true;
    cv::VideoWriter writer;
    double fps = 0.0, det_fps = 0.0, draw_time = 0.0;
    auto last = Clock::now(), det_last = Clock::now();
    long det_count = 0, n = 0, seq = 0;
    int64_t last_ts = -1;
    const auto t0 = Clock::now();
    cv::Mat out_frame;
    std::shared_ptr<const Published> cur = std::make_shared<Published>();
    std::vector<std::string> active, was_active;
    std::map<std::string, int> fired;   // --events: how often each meme switched on

    std::unique_ptr<Grabber> grabber;
    std::unique_ptr<LiveTracker> live_tracker;
    if (live) {
        std::optional<double> pace;
        if (clock) pace = 1 / clock->fps;
        grabber = std::make_unique<Grabber>(cap, mirror, t0, pace, args.bench.has_value());
        live_tracker = std::make_unique<LiveTracker>(*grabber, *tracker, crowd, W, H, args.force);
    }

    auto save_snapshot = [&](const cv::Mat& img) {
        cv::imwrite(args.snapshot, img);
        std::printf("[memecam] %ld frames, %.1f FPS, face=%s hands=%zu pose=%s active=%s -> %s\n", n, fps,
                    cur->fr.face ? "True" : "False", cur->fr.hands.size(), cur->fr.pose ? "True" : "False",
                    join_names(active).c_str(), args.snapshot.c_str());
    };

    std::unique_ptr<Sounds> sounds;   // meme sounds only in the window (not for --snapshot / --bench / --events)
    if (!headless) {
        sounds = std::make_unique<Sounds>(cfg, project_path("assets"));
        sounds->muted = args.mute;
    }
    if (!headless) {
        // GUI_NORMAL: no Qt toolbar (its save/zoom buttons open dialogs).
        cv::namedWindow(TITLE, cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
        cv::resizeWindow(TITLE, W, H);
    }

    int status = 0;
    try {
        cv::Mat rgb;
        while (!g_interrupted) {
            cv::Mat frame;
            double now;
            if (live) {
                auto got = grabber->wait(seq);
                if (!got) {
                    if (grabber->ended()) break;
                    continue;
                }
                frame = got->frame.clone();   // the grabbed frame is shared with the tracking thread
                now = got->t;
                seq = got->seq;
                cur = live_tracker->latest();
                if (cur->count != det_count) {
                    auto t = Clock::now();
                    double dt = std::max(std::chrono::duration<double>(t - det_last).count(), 1e-3);
                    det_fps = 0.9 * det_fps + 0.1 * (cur->count - det_count) / dt;
                    det_count = cur->count, det_last = t;
                }
            } else {
                cv::Mat raw;
                if (!cap.read(raw) || raw.empty()) {
                    if (!args.snapshot.empty() && !out_frame.empty()) save_snapshot(out_frame);   // ended early
                    break;
                }
                if (mirror) cv::flip(raw, frame, 1);
                else frame = raw;
                now = clock ? clock->next() : since(t0);
                int64_t ts = std::max<int64_t>(int64_t(now * 1000), last_ts + 1);
                last_ts = ts;
                cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
                auto p = std::make_shared<Published>();
                p->fr = tracker->process(rgb, ts);
                p->dets = crowd.update(p->fr, now, args.force);
                p->shown = crowd.snapshot();
                cur = std::move(p);
            }

            auto d0 = Clock::now();
            // [f] hides the meme overlays (detection keeps running, markers stay as toggled by [d])
            static const std::vector<Shown> none;
            active = compose(frame, cur->fr, show_memes ? cur->shown : none, now, cfg, media, cinema, show_markers,
                             on_face);
            if (sounds) sounds->update(active);
            auto t = Clock::now();
            fps = 0.9 * fps + 0.1 / std::max(std::chrono::duration<double>(t - last).count(), 1e-3);
            last = t;
            if (show_debug) {
                auto it = cur->dets.find(cur->fr.pid);
                static const std::map<std::string, Detection> empty;
                debug_panel(frame, it != cur->dets.end() ? it->second : empty, cfg);
            }
            std::optional<double> track_fps;
            if (live) track_fps = det_fps;
            std::optional<bool> sound_on;
            if (sounds && sounds->ok()) sound_on = !sounds->muted;
            hud(frame, fps, active, writer.isOpened(), show_markers, track_fps, show_memes, sound_on);
            draw_time += std::chrono::duration<double>(Clock::now() - d0).count();

            if (writer.isOpened()) writer.write(frame);
            n++;

            if (args.bench) {
                double el = since(t0);
                if (el >= *args.bench) {
                    long count = live_tracker->latest()->count;
                    std::printf("[memecam] bench %.1fs: display %.1f FPS, tracking %.1f FPS, compose+hud %.1f ms/frame\n",
                                el, n / el, count / el, draw_time / n * 1000);
                    std::string stats = tracker->stats();
                    if (!stats.empty()) std::printf("[memecam] %s\n", stats.c_str());
                    break;
                }
                continue;
            }
            if (args.events) {   // print every meme switching on / off, then a summary at the end of the clip
                for (const auto& m : active)
                    if (std::find(was_active.begin(), was_active.end(), m) == was_active.end()) {
                        std::printf("%7.2fs  + %s\n", now, m.c_str());
                        fired[m]++;
                    }
                for (const auto& m : was_active)
                    if (std::find(active.begin(), active.end(), m) == active.end())
                        std::printf("%7.2fs  - %s\n", now, m.c_str());
                was_active = active;
                continue;
            }
            if (!args.snapshot.empty()) {
                out_frame = frame;
                if (n >= args.frames) {
                    save_snapshot(frame);
                    break;
                }
                continue;
            }

            cv::imshow(TITLE, frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27 || !window_open(TITLE)) break;
            if (key == 'f') show_memes = !show_memes;
            else if (key == 'd') show_markers = !show_markers;
            else if (key == 'm' && sounds) sounds->muted = !sounds->muted;
            else if (key == 'i') show_debug = !show_debug;
            else if (key == 'n' && camera >= 0 && live && !writer.isOpened()) {
                // next camera that opens (wraps around); the capture thread is restarted around the switch
                live_tracker.reset();
                grabber.reset();
                cv::Mat first;
                for (int k = 1; k <= MAX_CAMERA + 1; k++) {
                    int i = (camera + k) % (MAX_CAMERA + 1);
                    if (open_camera(cap, i, cfg, first)) {
                        camera = i;
                        break;
                    }
                }
                if (!cap.isOpened()) open_camera(cap, camera, cfg, first);   // nothing else opened: back to this one
                std::printf("[memecam] camera %d\n", camera);
                std::fflush(stdout);
                grabber = std::make_unique<Grabber>(cap, mirror, t0, std::nullopt, false);
                live_tracker = std::make_unique<LiveTracker>(*grabber, *tracker, crowd, W, H, args.force);
                seq = 0;
            }
            else if (key == 'c') {
                if (live_tracker) live_tracker->calibrate();   // detector state belongs to the tracking thread
                else crowd.calibrate();
            } else if (key == 'r') {
                if (!writer.isOpened()) {
                    std::string out = recording_path();
                    writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, cv::Size(W, H));
                    if (writer.isOpened()) std::printf("[memecam] recording to %s\n", out.c_str());
                    else std::fprintf(g_err, "[memecam] could not open %s for recording\n", out.c_str());
                } else {
                    writer.release();
                    std::printf("[memecam] recording stopped\n");
                }
                std::fflush(stdout);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(g_err, "[memecam] %s%s\n", live ? "tracking thread failed: " : "", e.what());
        status = 1;
    }
    if (args.events) {
        std::string summary;
        for (const auto& [m, k] : fired) summary += (summary.empty() ? "" : ", ") + m + " x" + std::to_string(k);
        std::printf("[memecam] %ld frames, fired: %s\n", n, summary.empty() ? "nothing" : summary.c_str());
    }
    // Order matters: stop the tracking thread before the models go away, and the grabber before releasing
    // the capture it reads from.
    sounds.reset();
    live_tracker.reset();
    grabber.reset();
    cap.release();
    tracker.reset();
    if (writer.isOpened()) writer.release();
    if (!headless) cv::destroyAllWindows();
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{std::string(TITLE) +
                 ": webcam + mocap markers + gesture-triggered memes.\n"
                 "Keys: d = toggle markers/bounding boxes, f = toggle meme overlays, m = mute sounds, i = gesture debug scores,\n"
                 "      n = next camera, r = record, c = recalibrate head pose, q / Esc = quit"};
    Args args;
    args.config = project_path("config/config.toml");
    app.add_option("--camera", args.camera, "camera index (default from config.toml)");
    app.add_flag("--no-pose", args.no_pose, "skip body tracking (faster)");
    app.add_option("--config", args.config, "config file")->capture_default_str();
    app.add_option("--snapshot", args.snapshot, "headless: run --frames frames, save last output here, exit");
    app.add_option("--frames", args.frames, "frames for --snapshot")->capture_default_str();
    app.add_option("--force", args.force, "force a meme on (testing), repeatable")->allow_extra_args(false);
    app.add_option("--video", args.video, "read frames from this video file instead of the webcam");
    app.add_flag("--no-mirror", args.no_mirror, "don't mirror the input (already-mirrored video)");
    app.add_flag("--realtime", args.realtime,
                 "with --video: play the clip at its frame rate through the live path (background tracking), "
                 "like a webcam");
    app.add_option("--bench", args.bench, "headless: run the live path this long (seconds), print display FPS vs "
                                          "tracking FPS, exit");
    app.add_flag("--events", args.events,
                 "headless, with --video: track every frame, print each meme switching on/off and a summary");
    app.add_flag("--mute", args.mute, "start with the meme sounds muted (toggle: m)");
    app.add_flag("--verbose", args.verbose, "show LiteRT/OpenCV/Qt log output");
    CLI11_PARSE(app, argc, argv);

    if (!args.verbose) quiet_native_logs();
    std::signal(SIGINT, [](int) { g_interrupted = true; });
    std::signal(SIGTERM, [](int) { g_interrupted = true; });
    try {
        return run(args);
    } catch (const std::exception& e) {
        std::fprintf(g_err, "[memecam] error: %s\n", e.what());
        return 1;
    }
}
