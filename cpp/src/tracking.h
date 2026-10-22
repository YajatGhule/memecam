// People grouping, identities and helpers — 1:1 port of memecam/tracking.py (no model inference here).
#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "types.h"

namespace memecam {

constexpr double EAR_TO_FACE = 1.35;      // pose ear-to-ear distance × this ≈ face-mesh width
constexpr double POSE_FACE_MATCH = 1.5;   // a pose's nose within this many face-widths of a face's nose = same person
constexpr double HAND_WRIST_MATCH = 1.2;  // hand wrist within this many face-widths of a pose wrist = that person's hand
constexpr double HAND_FACE_MATCH = 3.5;   // ...else within this many face-widths of a face = that person's hand
constexpr double WRIST_STRAY = 2.5;       // a pose wrist this far (face-widths, sideways) from its own head is suspect
constexpr double GHOST_DIST = 2.0;        // a face-less pose this close (face-widths) to another person's head is a duplicate
constexpr double ID_MAX_JUMP = 1.5;       // identity: a head moves at most this many face-widths between frames
constexpr int ID_CONFIRM = 2;             // identity: frames a person must be seen before a dropout of theirs claims hands
constexpr double ID_FORGET = 1.0;         // seconds an unseen person keeps their id (bridges tracking dropouts only)

struct HeadRef { cv::Point2d nose; double width; };   // a head position (nose, or nose/mouth point) + face width

double pose_face_width(const Pose& pose);
std::optional<HeadRef> head_of(const Frame& person);   // nose + face width: face mesh, else body pose
std::optional<Box> head_box(const Frame& person);
std::optional<Box> hands_box(const Frame& person);

using Wrists = std::array<cv::Point2d, 2>;   // pose wrists 15, 16

struct Seen {                          // where a tracked person was last seen (Python Seen)
    std::optional<cv::Point2d> nose;   // head position (none: hands only)
    double fw = 1.0;                   // face width
    double t = 0.0;                    // when
    std::optional<Wrists> wrists;      // last pose wrists, if seen recently
    double wrists_t = 0.0;
    int hits = 1;                      // frames seen in a row (with gaps under `forget`)
};

// Group raw detections into people, largest first (ids assigned later by Identities).
std::vector<Frame> group_people(const std::vector<Face>& faces, const std::vector<std::shared_ptr<const Pose>>& poses,
                                const std::vector<Hand>& hands, int width, int max_people = 3,
                                double min_face_width = 0.08, const std::vector<Seen>& recent = {});

// Persistent person ids: nearest-neighbour matching of head positions between frames (Python Identities).
class Identities {
public:
    explicit Identities(double max_jump = ID_MAX_JUMP, double forget = ID_FORGET);
    std::vector<Seen> recent(double now) const;
    void assign(std::vector<Frame>& people, double now);   // sets pid, fills `renamed`
    std::map<int, int> renamed;                             // old -> new, from the latest assign()
    // pid -> last seen, in insertion order (Python dict order matters for tie-breaks)
    const std::vector<std::pair<int, Seen>>& tracks() const { return tracks_; }
private:
    double max_jump_, forget_;
    std::vector<std::pair<int, Seen>> tracks_;
    Seen* find(int pid);
};

// 160x90 blurred grayscale for frame differencing (input: RGB, any size — like Python's motion_gray).
cv::Mat motion_gray(const cv::Mat& rgb);

}  // namespace memecam
