// Dev tool: draw a frame dumped by tools/dump_frame.py (Python tracking state + raw frame) with the C++
// renderer and compare it with the Python rendering.
//
//   uv run --project ~/Projects/meme-cam python tools/dump_frame.py ~/Videos/Webcam/Neutral.webm \
//       --frames 30 --force nerd --out /tmp/rc/nerd
//   build/render_check /tmp/rc/nerd          # reads nerd.json + nerd_raw.png, writes nerd_cpp.png,
//                                            # nerd_cmp.png (python | c++ | diff x4) and prints the diff
#include <cstdio>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "compose.h"
#include "config.h"
#include "mocap.h"
#include "overlays.h"

using namespace memecam;
using json = nlohmann::json;

static Box box_of(const json& j) {   // Python (x0, y0, x1, y1)
    int x0 = j[0], y0 = j[1], x1 = j[2], y1 = j[3];
    return Box(x0, y0, x1 - x0, y1 - y0);
}

static Frame person_of(const json& j, int W, int H) {
    Frame f;
    f.width = W, f.height = H;
    f.pid = j["pid"];
    if (!j["face"].is_null()) {
        Face face;
        for (const auto& p : j["face"]) face.pts.push_back(Vec3(p[0], p[1], p[2]));
        f.face = std::move(face);
    }
    if (!j["pose"].is_null()) {
        auto pose = std::make_shared<Pose>();
        for (int i = 0; i < 33; i++) {
            const auto& p = j["pose"][i];
            (*pose)[i] = PoseLm{p[0], p[1], p[2], p[3]};
        }
        f.pose = pose;
    }
    for (const auto& jh : j["hands"]) {
        Hand h;
        for (int i = 0; i < 21; i++) h.pts[i] = Vec3(jh["pts"][i][0], jh["pts"][i][1], jh["pts"][i][2]);
        h.label = jh["label"];
        h.gesture = jh["gesture"];
        h.gesture_score = jh["score"];
        f.hands.push_back(h);
    }
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: render_check <prefix> [more prefixes...]\n");
        return 2;
    }
    for (int ai = 1; ai < argc; ai++) {
        std::string prefix = argv[ai];
        std::ifstream in(prefix + ".json");
        if (!in) { std::fprintf(stderr, "cannot read %s.json\n", prefix.c_str()); return 1; }
        json d = json::parse(in);
        cv::Mat frame = cv::imread(prefix + "_raw.png");
        cv::Mat py = cv::imread(prefix + ".png");
        CV_Assert(!frame.empty());

        Config cfg = load_config(project_path("config/config.toml"),
                                 d["v2"].get<bool>() ? project_path("config/config_v2.toml") : "");
        if (d.contains("placements"))
            for (auto& [name, where] : d["placements"].items())
                for (auto& m : cfg.memes)
                    if (m.name == name) m.placement = where.get<std::string>();
        MemeMedia media = load_meme_media(cfg, project_path("assets"));
        const int W = d["width"], H = d["height"];

        Frame fr = person_of(d["top"], W, H);
        for (const auto& jp : d["people"]) fr.people.push_back(person_of(jp, W, H));
        std::vector<Shown> shown;
        for (const auto& js : d["shown"]) {
            Shown s{js["pid"], std::nullopt, {}};
            if (!js["head"].is_null()) s.head = box_of(js["head"]);
            for (const auto& jm : js["memes"]) {
                const MemeSpec* spec = cfg.meme(jm["name"]);
                if (!spec) continue;
                MemeState m(spec);
                m.started = jm["started"].get<double>();
                if (!jm["anchor"].is_null()) m.anchor = box_of(jm["anchor"]);
                s.memes.emplace(spec->name, std::move(m));
            }
            shown.push_back(std::move(s));
        }

        CinemaEffect cinema;
        bool markers = d["markers"], on_face = d["on_face"];
        double now = d["now"];
        auto t0 = cv::getTickCount();
        auto active = compose(frame, fr, shown, now, cfg, media, cinema, markers, on_face);
        if (d["debug"].get<bool>()) {
            std::map<std::string, Detection> dets;
            auto key = std::to_string(fr.pid);
            if (d["dets"].contains(key))
                for (auto& [name, jd] : d["dets"][key].items()) dets[name] = Detection{jd["raw"], jd["score"], {}};
            debug_panel(frame, dets, cfg);
        }
        hud(frame, d["fps"], active, false, markers, std::nullopt, true);
        double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
        cv::imwrite(prefix + "_cpp.png", frame);

        std::string act;
        for (auto& a : active) act += a + " ";
        std::string py_act;
        for (auto& a : d["active"]) py_act += a.get<std::string>() + " ";
        if (!py.empty() && py.size() == frame.size()) {
            cv::Mat diff, gray;
            cv::absdiff(py, frame, diff);
            cv::Scalar mean = cv::mean(diff);
            cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
            double frac = cv::countNonZero(gray > 24) / double(gray.total());
            cv::Mat d4, cmp;
            diff.convertTo(d4, -1, 4.0);
            cv::hconcat(std::vector<cv::Mat>{py, frame, d4}, cmp);
            cv::resize(cmp, cmp, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::imwrite(prefix + "_cmp.png", cmp);
            std::printf("%s: compose+hud %.1f ms, active [%s] (python [%s]), mean |diff| %.2f, pixels > 24: %.2f%%\n",
                        prefix.c_str(), ms, act.c_str(), py_act.c_str(), (mean[0] + mean[1] + mean[2]) / 3,
                        frac * 100);
        } else {
            std::printf("%s: compose+hud %.1f ms, active [%s]\n", prefix.c_str(), ms, act.c_str());
        }
    }
    return 0;
}
