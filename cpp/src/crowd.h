// Per-person meme state (Crowd / Person / Shown in memecam/main.py), without any rendering.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.h"
#include "gestures.h"

namespace memecam {

struct MemeState {                 // Python Meme minus the media
    const MemeSpec* spec = nullptr;
    Trigger trigger;
    std::optional<double> started; // when it switched on (gif restarts from frame 0)
    std::optional<Box> anchor;

    MemeState() = default;
    explicit MemeState(const MemeSpec* s);   // fresh trigger from the spec (hold, min_show, linger, cooldown)
    bool update(const Detection& det, double now);
};

struct Shown {                      // what the display thread draws for one person (a consistent copy)
    int pid;
    std::optional<Box> head;
    std::map<std::string, MemeState> memes;
};

// Per-person state: their own detectors (EMAs, head-pose baseline) and meme triggers.
struct Person {
    Person(int pid, const Config& cfg);
    int pid;
    Detectors detectors;
    std::map<std::string, MemeState> memes;
    double last_seen = 0.0;
    std::optional<Box> head;        // head box in the current frame (none if absent)
};

// Everyone in view: one Person per tracked person id, so each person's memes fire independently.
// Only people in the current tracking result are run and shown; someone who drops out keeps their
// frozen state for `forget` seconds, then it is dropped.
class Crowd {
public:
    explicit Crowd(const Config& cfg);   // keeps a reference: `cfg` must outlive the Crowd
    void calibrate();
    // Run every present person's detectors + triggers. Returns detections per person id.
    // `force` turns those memes' detections on for the primary (first) person.
    std::map<int, std::map<std::string, Detection>> update(const Frame& fr, double now,
                                                           const std::vector<std::string>& force = {});
    std::vector<Shown> snapshot() const;   // only people present in the last update, sorted by pid
    double forget() const { return cfg_.forget; }
    const std::map<int, std::unique_ptr<Person>>& people() const { return people_; }
    std::map<int, std::unique_ptr<Person>>& people() { return people_; }
    const std::vector<int>& present() const { return present_; }
private:
    const Config& cfg_;
    std::map<int, std::unique_ptr<Person>> people_;
    std::vector<int> present_;     // pids detected in the latest update, in frame order
};

}  // namespace memecam
