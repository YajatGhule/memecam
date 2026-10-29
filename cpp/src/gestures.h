// Gesture detectors and the trigger state machine — 1:1 port of memecam/gestures.py.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "config.h"
#include "tracking.h"

namespace memecam {

struct Detection {
    bool raw = false;
    double score = 0.0;             // for the debug overlay
    std::optional<Box> anchor;      // bbox the overlay should sit next to
};

// gestures.py DEFAULTS (merged under the [thresholds] of the config).
const std::map<std::string, double>& detector_defaults();

struct FingerStates { bool index = false, middle = false, ring = false, pinky = false; int count() const; };
FingerStates finger_states(const Hand& hand, double ratio);
std::optional<HeadRef> head_ref(const Frame& fr);   // gestures.py head_ref: nose/mouth point + face width

class Detectors {
public:
    explicit Detectors(const std::map<std::string, double>& thresholds = {});   // merged with DEFAULTS
    ~Detectors();
    Detectors(Detectors&&) noexcept;
    Detectors& operator=(Detectors&&) noexcept;

    void calibrate();
    // Detections by meme name ("dafoe" only when a face or a body pose is present, like Python).
    std::map<std::string, Detection> run(const Frame& fr, double now);
    double cfg(const std::string& key, double def = 0.0) const;
    double motion_level(const Frame& fr) const;   // fraction of changed pixels around this person
    struct State;                    // all EMA / baseline / history state (defined in gestures.cpp)
private:
    std::map<std::string, double> cfg_;
    std::unique_ptr<State> st_;
};

class Trigger {   // idle --(raw held `hold`)--> active --(raw gone, min_show & linger elapsed)--> cooldown
public:
    // Same defaults as the Python class (Meme passes its own: hold 0.3, min_show 0, linger 0.3, cooldown 0).
    Trigger(double hold = 0.3, double min_show = 2.0, double linger = 0.6, double cooldown = 0.8);
    bool update(bool raw, double now);
    bool active() const { return started_.has_value(); }
    double hold, min_show, linger, cooldown;
private:
    std::optional<double> raw_since_, started_;
    double last_raw_ = 0, cool_until_ = 0;
};

}  // namespace memecam
