// Android glue (JNI): the camera frames come from the Kotlin side (CameraX), everything else is the desktop
// code: Tracker (LiteRT) on a tracking thread, Crowd (gestures + meme triggers), compose (markers + memes),
// Sounds (miniaudio). The rendered frame is written into an android.graphics.Bitmap.
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "audio.h"
#include "compose.h"
#include "config.h"
#include "crowd.h"
#include "mocap.h"
#include "vision/tracker.h"

namespace memecam {
namespace {

using Clock = std::chrono::steady_clock;

struct Published {
    Frame fr;
    std::vector<Shown> shown;
    long count = 0;
};

struct Engine {
    Config cfg;
    std::unique_ptr<Tracker> tracker;
    std::unique_ptr<Crowd> crowd;
    MemeMedia media;
    CinemaEffect cinema;
    std::unique_ptr<Sounds> sounds;
    Clock::time_point t0 = Clock::now();

    // newest camera frame for the tracking thread
    std::mutex mu;
    std::condition_variable cv;
    cv::Mat rgb;
    double rgb_t = 0;
    long seq = 0;
    std::shared_ptr<const Published> pub = std::make_shared<Published>();
    std::atomic<bool> stopping{false}, calibrate{false};
    std::thread thread;

    std::atomic<bool> markers{true}, memes{true}, debug{false};
    double fps = 0, track_fps = 0;
    Clock::time_point last = Clock::now(), track_last = Clock::now();
    long track_count = 0;

    ~Engine() {
        stopping = true;
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }

    void run() {
        long done = 0, count = 0;
        int64_t last_ts = -1;
        while (!stopping) {
            cv::Mat in;
            double t;
            {
                std::unique_lock lock(mu);
                cv.wait(lock, [&] { return seq > done || stopping; });
                if (stopping) break;
                in = rgb, t = rgb_t, done = seq;
            }
            if (calibrate.exchange(false)) crowd->calibrate();
            int64_t ts = std::max<int64_t>(int64_t(t * 1000), last_ts + 1);
            last_ts = ts;
            auto p = std::make_shared<Published>();
            try {
                p->fr = tracker->process(in, ts);
            } catch (const std::exception& e) {
                __android_log_print(ANDROID_LOG_ERROR, "memecam", "tracking failed: %s", e.what());
                continue;
            }
            crowd->update(p->fr, t);
            p->shown = crowd->snapshot();
            p->count = ++count;
            std::lock_guard lock(mu);
            pub = std::move(p);
        }
    }
};

std::unique_ptr<Engine> g_engine;

std::string jstr(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c;
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace
}  // namespace memecam

using namespace memecam;

extern "C" {

// Native.init(root, v2): null on success, else the error message.
JNIEXPORT jstring JNICALL Java_dev_yajatghule_memecam_Native_init(JNIEnv* env, jclass, jstring root, jboolean v2) {
    try {
        g_engine.reset();
        set_project_root(jstr(env, root));
        auto e = std::make_unique<Engine>();
        e->cfg = v2 ? load_config(project_path("config/config.toml"), project_path("config/config_v2.toml"))
                    : load_config(project_path("config/config.toml"));
        e->tracker = std::make_unique<Tracker>(e->cfg, true, true);
        e->crowd = std::make_unique<Crowd>(e->cfg);
        e->media = load_meme_media(e->cfg, project_path("assets"));
        e->sounds = std::make_unique<Sounds>(e->cfg, project_path("assets"));
        e->thread = std::thread([p = e.get()] { p->run(); });
        g_engine = std::move(e);
        return nullptr;
    } catch (const std::exception& ex) {
        return env->NewStringUTF(ex.what());
    }
}

// Native.process(rgba, width, height, rowStride, rotation, mirror, out): one camera frame in (RGBA_8888, as the
// sensor delivers it), the finished upright frame out. Returns the active meme names, comma separated.
JNIEXPORT jstring JNICALL Java_dev_yajatghule_memecam_Native_process(JNIEnv* env, jclass, jobject buf, jint w, jint h,
                                                                      jint stride, jint rotation, jboolean mirror,
                                                                      jobject bitmap) {
    Engine* e = g_engine.get();
    if (!e) return nullptr;
    auto* data = static_cast<uint8_t*>(env->GetDirectBufferAddress(buf));
    cv::Mat rgba(h, w, CV_8UC4, data, size_t(stride));
    cv::Mat up;
    switch (rotation) {
        case 90: cv::rotate(rgba, up, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(rgba, up, cv::ROTATE_180); break;
        case 270: cv::rotate(rgba, up, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: up = rgba;
    }
    if (mirror) cv::flip(up, up, 1);
    const double now = std::chrono::duration<double>(Clock::now() - e->t0).count();

    cv::Mat rgb;
    cv::cvtColor(up, rgb, cv::COLOR_RGBA2RGB);
    {
        std::lock_guard lock(e->mu);
        e->rgb = rgb, e->rgb_t = now, e->seq++;
    }
    e->cv.notify_one();

    std::shared_ptr<const Published> cur;
    {
        std::lock_guard lock(e->mu);
        cur = e->pub;
    }
    if (cur->count != e->track_count) {
        auto t = Clock::now();
        double dt = std::max(std::chrono::duration<double>(t - e->track_last).count(), 1e-3);
        e->track_fps = 0.9 * e->track_fps + 0.1 * (cur->count - e->track_count) / dt;
        e->track_count = cur->count, e->track_last = t;
    }

    cv::Mat frame;
    cv::cvtColor(up, frame, cv::COLOR_RGBA2BGR);
    static const std::vector<Shown> none;
    auto active = compose(frame, cur->fr, e->memes ? cur->shown : none, now, e->cfg, e->media, e->cinema, e->markers,
                          e->cfg.meme_on_face);
    if (e->sounds) e->sounds->update(active);
    auto t = Clock::now();
    e->fps = 0.9 * e->fps + 0.1 / std::max(std::chrono::duration<double>(t - e->last).count(), 1e-3);
    e->last = t;
    std::optional<bool> sound_on;
    if (e->sounds && e->sounds->ok()) sound_on = !e->sounds->muted;
    hud(frame, e->fps, active, false, e->markers, e->track_fps, e->memes, sound_on, false);

    AndroidBitmapInfo info;
    void* px = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) == 0 && int(info.width) == frame.cols &&
        int(info.height) == frame.rows && AndroidBitmap_lockPixels(env, bitmap, &px) == 0) {
        cv::Mat dst(frame.rows, frame.cols, CV_8UC4, px, info.stride);
        cv::cvtColor(frame, dst, cv::COLOR_BGR2RGBA);
        AndroidBitmap_unlockPixels(env, bitmap);
    }
    std::string names;
    for (const auto& a : active) names += (names.empty() ? "" : ",") + a;
    return env->NewStringUTF(names.c_str());
}

JNIEXPORT void JNICALL Java_dev_yajatghule_memecam_Native_setOptions(JNIEnv*, jclass, jboolean markers, jboolean memes,
                                                                      jboolean mute) {
    if (!g_engine) return;
    g_engine->markers = markers, g_engine->memes = memes;
    if (g_engine->sounds) g_engine->sounds->muted = mute;
}

JNIEXPORT void JNICALL Java_dev_yajatghule_memecam_Native_calibrate(JNIEnv*, jclass) {
    if (g_engine) g_engine->calibrate = true;
}

JNIEXPORT void JNICALL Java_dev_yajatghule_memecam_Native_release(JNIEnv*, jclass) { g_engine.reset(); }

}  // extern "C"
