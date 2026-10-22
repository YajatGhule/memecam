#include "crowd.h"

#include <algorithm>

namespace memecam {

MemeState::MemeState(const MemeSpec* s)
    : spec(s), trigger(s->hold, s->min_show, s->linger, s->cooldown) {}

bool MemeState::update(const Detection& det, double now) {
    bool was = started.has_value();
    if (trigger.update(det.raw, now)) {
        if (!was) started = now;
        if (det.anchor && (det.raw || !anchor)) anchor = det.anchor;
    } else {
        started.reset();
        anchor.reset();
    }
    return started.has_value();
}

Person::Person(int pid_, const Config& cfg) : pid(pid_), detectors(cfg.thresholds) {
    for (const auto& m : cfg.memes) memes.emplace(m.name, MemeState(&m));
}

Crowd::Crowd(const Config& cfg) : cfg_(cfg) {}

void Crowd::calibrate() {
    for (auto& [pid, p] : people_) p->detectors.calibrate();
}

std::map<int, std::map<std::string, Detection>> Crowd::update(const Frame& fr, double now,
                                                              const std::vector<std::string>& force) {
    std::vector<const Frame*> people;
    if (!fr.people.empty())
        for (const auto& q : fr.people) people.push_back(&q);
    else if (fr.pid)
        people.push_back(&fr);
    for (auto it = people_.begin(); it != people_.end();) {
        if (now - it->second->last_seen > cfg_.forget) it = people_.erase(it);
        else ++it;
    }
    for (const auto& [old_id, new_id] : fr.renamed) {    // ids renumbered by the tracker
        auto it = people_.find(old_id);
        if (it == people_.end()) continue;
        std::unique_ptr<Person> st = std::move(it->second);
        people_.erase(it);
        st->pid = new_id;
        people_[new_id] = std::move(st);
    }
    for (auto& [pid, st] : people_) st->head.reset();
    std::map<int, std::map<std::string, Detection>> out;
    present_.clear();
    for (const Frame* q : people) {
        auto& slot = people_[q->pid];
        if (!slot) slot = std::make_unique<Person>(q->pid, cfg_);
        Person& st = *slot;
        st.last_seen = now;
        st.head = head_box(*q);
        if (!st.head) st.head = hands_box(*q);
        auto dets = st.detectors.run(*q, now);
        if (q == people.front()) {
            for (const auto& name : force) {
                std::optional<Box> a;
                if (!q->hands.empty()) a = q->hands[0].bbox();
                else if (q->face) a = q->face->bbox();
                dets[name] = Detection{true, 1.0, a};
            }
        }
        for (auto& [name, meme] : st.memes) {
            auto it = dets.find(name);
            meme.update(it != dets.end() ? it->second : Detection{}, now);
        }
        out[q->pid] = std::move(dets);
        present_.push_back(q->pid);
    }
    return out;
}

std::vector<Shown> Crowd::snapshot() const {
    std::vector<int> pids = present_;
    std::sort(pids.begin(), pids.end());
    std::vector<Shown> out;
    for (int pid : pids) {
        const Person& p = *people_.at(pid);
        out.push_back(Shown{pid, p.head, p.memes});
    }
    return out;
}

}  // namespace memecam
