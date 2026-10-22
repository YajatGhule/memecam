#include "tracking.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace memecam {

namespace {

double dist(const cv::Point2d& a, const cv::Point2d& b) { return norm2(a.x - b.x, a.y - b.y); }
cv::Point2d lm_xy(const Pose& p, int i) { return {p[i].x, p[i].y}; }

}  // namespace

double pose_face_width(const Pose& pose) {
    double ears = dist(lm_xy(pose, 7), lm_xy(pose, 8));
    double shoulders = dist(lm_xy(pose, 11), lm_xy(pose, 12));
    return std::max(EAR_TO_FACE * ears, 0.45 * shoulders) + 1e-6;
}

std::optional<HeadRef> head_of(const Frame& person) {
    if (person.face) return HeadRef{xy(person.face->pts[1]), person.face->width()};
    if (person.pose) return HeadRef{lm_xy(*person.pose, 0), pose_face_width(*person.pose)};
    return std::nullopt;
}

std::optional<Box> head_box(const Frame& person) {
    if (person.face) return person.face->bbox();
    if (person.pose) {
        double x = (*person.pose)[0].x, y = (*person.pose)[0].y, r = 0.5 * pose_face_width(*person.pose);
        return box_from_corners(int(x - r), int(y - 1.25 * r), int(x + r), int(y + 1.25 * r));
    }
    return std::nullopt;
}

std::optional<Box> hands_box(const Frame& person) {
    if (person.hands.empty()) return std::nullopt;
    int x0 = std::numeric_limits<int>::max(), y0 = x0, x1 = std::numeric_limits<int>::min(), y1 = x1;
    for (const auto& h : person.hands) {
        Box b = h.bbox();
        x0 = std::min(x0, b.x); y0 = std::min(y0, b.y);
        x1 = std::max(x1, b.x + b.width); y1 = std::max(y1, b.y + b.height);
    }
    return box_from_corners(x0, y0, x1, y1);
}

namespace {

// The recently seen people (with a head) that none of the current heads continues.
std::vector<Seen> absent_of(const std::vector<std::optional<HeadRef>>& heads, const std::vector<Seen>& recent) {
    std::vector<std::tuple<double, int, int>> pairs;
    for (int i = 0; i < int(heads.size()); ++i) {
        if (!heads[i]) continue;
        for (int j = 0; j < int(recent.size()); ++j) {
            if (!recent[j].nose) continue;
            pairs.emplace_back(dist(heads[i]->nose, *recent[j].nose) / std::max(heads[i]->width, recent[j].fw), i, j);
        }
    }
    std::sort(pairs.begin(), pairs.end());
    std::set<int> used_i, used_j;
    for (auto& [d, i, j] : pairs) {
        if (d < ID_MAX_JUMP && !used_i.count(i) && !used_j.count(j)) {
            used_i.insert(i);
            used_j.insert(j);
        }
    }
    std::vector<Seen> out;
    for (int j = 0; j < int(recent.size()); ++j)
        if (recent[j].nose && !used_j.count(j)) out.push_back(recent[j]);
    return out;
}

}  // namespace

std::vector<Frame> group_people(const std::vector<Face>& faces_in, const std::vector<std::shared_ptr<const Pose>>& poses_in,
                                const std::vector<Hand>& hands_in, int width, int max_people, double min_face_width,
                                const std::vector<Seen>& recent) {
    const double min_w = min_face_width * width;
    std::vector<const Face*> faces;
    for (const auto& f : faces_in)
        if (f.width() >= min_w) faces.push_back(&f);
    std::stable_sort(faces.begin(), faces.end(), [](const Face* a, const Face* b) { return -a->width() < -b->width(); });
    if (int(faces.size()) > max_people) faces.resize(std::max(0, max_people));
    std::vector<std::shared_ptr<const Pose>> poses;
    for (const auto& p : poses_in)
        if (pose_face_width(*p) >= min_w) poses.push_back(p);

    std::vector<Frame> people;
    for (const Face* f : faces) {
        Frame q;
        q.face = *f;
        people.push_back(std::move(q));
    }
    std::vector<std::tuple<double, int, int>> pairs;
    for (int i = 0; i < int(faces.size()); ++i)
        for (int j = 0; j < int(poses.size()); ++j)
            pairs.emplace_back(dist(lm_xy(*poses[j], 0), xy(faces[i]->pts[1])) / faces[i]->width(), i, j);
    std::sort(pairs.begin(), pairs.end());
    std::set<int> used_f, used_p;
    for (auto& [d, i, j] : pairs) {
        if (d < POSE_FACE_MATCH && !used_f.count(i) && !used_p.count(j)) {
            used_f.insert(i);
            used_p.insert(j);
            people[i].pose = poses[j];
        }
    }
    // Leftover poses are people whose face is hidden, unless they sit on someone already found (a ghost).
    std::vector<int> left;
    for (int j = 0; j < int(poses.size()); ++j)
        if (!used_p.count(j)) left.push_back(j);
    std::stable_sort(left.begin(), left.end(),
                     [&](int a, int b) { return -pose_face_width(*poses[a]) < -pose_face_width(*poses[b]); });
    for (int j : left) {
        const Pose& p = *poses[j];
        double fw = pose_face_width(p);
        bool ok = true;
        for (const auto& q : people) {
            auto hd = head_of(q);
            if (!hd) continue;
            if (!(dist(lm_xy(p, 0), hd->nose) > GHOST_DIST * std::max(fw, hd->width))) { ok = false; break; }
        }
        if (ok) {
            Frame q;
            q.pose = poses[j];
            people.push_back(std::move(q));
        }
    }
    std::stable_sort(people.begin(), people.end(),
                     [](const Frame& a, const Frame& b) { return -head_of(a)->width < -head_of(b)->width; });
    if (int(people.size()) > max_people) people.resize(std::max(0, max_people));

    if (people.empty()) {
        if (!hands_in.empty()) {
            std::vector<int> idx(hands_in.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(),
                             [&](int a, int b) { return -hands_in[a].size() < -hands_in[b].size(); });
            Frame q;
            for (size_t k = 0; k < idx.size() && k < 2; ++k) q.hands.push_back(hands_in[idx[k]]);
            people.push_back(std::move(q));
        }
        return people;
    }
    std::vector<std::optional<HeadRef>> heads;
    for (const auto& q : people) heads.push_back(head_of(q));
    std::vector<Seen> absent = absent_of(heads, recent);
    if (people.size() == 1 && absent.empty()) {   // one person: every hand is theirs, no matching needed
        if (hands_in.size() > 2) {
            cv::Point2d nose = heads[0]->nose;
            std::vector<int> idx(hands_in.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
                return dist(xy(hands_in[a].pts[0]), nose) < dist(xy(hands_in[b].pts[0]), nose);
            });
            std::set<int> near(idx.begin(), idx.begin() + 2);
            for (int i = 0; i < int(hands_in.size()); ++i)
                if (near.count(i)) people[0].hands.push_back(hands_in[i]);
        } else {
            people[0].hands = hands_in;
        }
        return people;
    }

    // Everyone who can own a hand: the people found now, then the recently seen but missing ones.
    struct Slot { cv::Point2d nose; double fw; std::optional<Wrists> wrists; };
    std::vector<Slot> slots;
    for (size_t k = 0; k < people.size(); ++k) {
        std::optional<Wrists> w;
        if (people[k].pose) w = Wrists{lm_xy(*people[k].pose, 15), lm_xy(*people[k].pose, 16)};
        slots.push_back({heads[k]->nose, heads[k]->width, w});
    }
    for (const auto& s : absent) slots.push_back({*s.nose, s.fw, s.wrists});
    const bool alone = people.size() == 1;
    const int n_people = int(people.size());
    std::vector<std::tuple<double, int, int>> cands;   // (cost, hand index, slot index)
    for (int i = 0; i < int(hands_in.size()); ++i) {
        cv::Point2d w = xy(hands_in[i].pts[0]);
        std::vector<double> near;
        for (const auto& s : slots) near.push_back(dist(w, s.nose) / s.fw);
        int nearest = int(std::min_element(near.begin(), near.end()) - near.begin());
        for (int k = 0; k < int(slots.size()); ++k) {
            const Slot& s = slots[k];
            if (s.wrists) {
                double d = std::min(dist(w, (*s.wrists)[0]), dist(w, (*s.wrists)[1])) / s.fw;
                bool stray = std::abs(w.x - s.nose.x) / s.fw > WRIST_STRAY && near[nearest] < 0.75 * near[k];
                if (d < HAND_WRIST_MATCH && !stray) {
                    cands.emplace_back(d, i, k);
                    continue;
                }
            }
            if (k == nearest && (near[k] < HAND_FACE_MATCH || (alone && k == 0)))
                cands.emplace_back(100 + near[k], i, k);
            else if (alone && k == 0 && nearest >= n_people && near[nearest] >= HAND_FACE_MATCH)
                cands.emplace_back(100 + near[k], i, k);
        }
    }
    std::sort(cands.begin(), cands.end());
    std::map<int, int> owner;
    std::vector<int> count(slots.size(), 0);
    for (auto& [c, i, k] : cands) {
        if (!owner.count(i) && count[k] < 2) {
            owner[i] = k;
            count[k] += 1;
        }
    }
    for (int i = 0; i < int(hands_in.size()); ++i) {   // keep MediaPipe's order within a person
        auto it = owner.find(i);
        if (it != owner.end() && it->second < n_people) people[it->second].hands.push_back(hands_in[i]);
    }
    return people;
}

// ---- Identities ---------------------------------------------------------------------------

Identities::Identities(double max_jump, double forget) : max_jump_(max_jump), forget_(forget) {}

Seen* Identities::find(int pid) {
    for (auto& [k, v] : tracks_)
        if (k == pid) return &v;
    return nullptr;
}

std::vector<Seen> Identities::recent(double now) const {
    std::vector<Seen> out;
    for (const auto& [pid, s] : tracks_) {
        if (now - s.t <= forget_ && s.hits >= ID_CONFIRM) {
            bool fresh = s.wrists.has_value() && now - s.wrists_t <= forget_;
            Seen c = s;
            if (!fresh) c.wrists.reset();
            out.push_back(c);
        }
    }
    return out;
}

void Identities::assign(std::vector<Frame>& people, double now) {
    std::vector<std::pair<int, Seen>> kept;
    for (auto& kv : tracks_)
        if (now - kv.second.t <= forget_) kept.push_back(kv);
    tracks_ = std::move(kept);
    renamed.clear();
    if (tracks_.size() == 1 && tracks_[0].first != 1) {     // alone again: back to PERSON_01
        renamed[tracks_[0].first] = 1;
        tracks_[0].first = 1;
    }
    std::vector<std::optional<HeadRef>> heads;
    for (const auto& q : people) heads.push_back(head_of(q));
    std::vector<std::tuple<double, int, int>> pairs;
    for (int i = 0; i < int(heads.size()); ++i) {
        if (!heads[i]) continue;
        for (const auto& [pid, s] : tracks_) {
            if (!s.nose) continue;
            double d = dist(heads[i]->nose, *s.nose) / std::max(heads[i]->width, s.fw);
            if (d < max_jump_) pairs.emplace_back(d, i, pid);
        }
    }
    std::sort(pairs.begin(), pairs.end());
    std::map<int, int> done;   // person index -> pid
    auto taken = [&](int pid) {
        for (auto& [i, p] : done)
            if (p == pid) return true;
        return false;
    };
    for (auto& [d, i, pid] : pairs)
        if (!done.count(i) && !taken(pid)) done[i] = pid;
    for (int i = 0; i < int(people.size()); ++i) {
        Frame& q = people[i];
        if (!done.count(i)) {
            // hands only: whoever was seen most recently; a face for a hands-only track
            int best = -1;
            double best_t = 0;
            for (const auto& [pid, s] : tracks_) {
                if (taken(pid) || !(!heads[i] || !s.nose)) continue;
                if (best < 0 || s.t > best_t) { best = pid; best_t = s.t; }
            }
            if (best >= 0) {
                done[i] = best;
            } else {
                int limit = int(tracks_.size() + people.size() + 2);
                for (int n = 1; n < limit; ++n) {
                    if (!find(n) && !taken(n)) { done[i] = n; break; }
                }
            }
        }
        q.pid = done[i];
        Seen* prev = find(q.pid);
        Seen s;
        if (heads[i]) {
            s.nose = heads[i]->nose;
            s.fw = heads[i]->width;
        } else if (prev) {
            s.nose = prev->nose;
            s.fw = prev->fw;
        } else {
            s.nose.reset();
            s.fw = 1.0;
        }
        if (q.pose) {
            s.wrists = Wrists{lm_xy(*q.pose, 15), lm_xy(*q.pose, 16)};
            s.wrists_t = now;
        } else if (prev) {
            s.wrists = prev->wrists;
            s.wrists_t = prev->wrists_t;
        } else {
            s.wrists.reset();
            s.wrists_t = 0.0;
        }
        s.t = now;
        s.hits = prev ? prev->hits + 1 : 1;
        if (prev) *prev = s;
        else tracks_.emplace_back(q.pid, s);
    }
}

cv::Mat motion_gray(const cv::Mat& rgb) {
    cv::Mat small, gray, out;
    cv::resize(rgb, small, cv::Size(MOTION_W, MOTION_H), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray, out, cv::Size(5, 5), 0);
    return out;
}

}  // namespace memecam
