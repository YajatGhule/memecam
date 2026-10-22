// Multi-person grouping, identities and per-person meme state — port of tests/test_people.py.
#include <catch2/catch_test_macros.hpp>

#include "builders.h"
#include "config.h"
#include "crowd.h"

using namespace mt;

namespace {

const int W = 1280, H = 720;
const double LEFT = 330.0, RIGHT = 950.0;   // the two people's face centres (x)

Face face_at(double x, double w = 200.0, double y = FACE_C.y) { return make_face(0.6, {x, y}, w, 1.3 * w); }

// make_pose() moved so its nose sits under a face centred at x (wrists relative to that).
std::shared_ptr<const Pose> pose_at(double x, double wl_dx = -150, double wr_dx = 150, double wy = 560, double dy = 0.0) {
    Pose p = make_pose_arr({FACE_C.x + wl_dx, wy}, {FACE_C.x + wr_dx, wy});
    for (auto& l : p) {
        l.x += x - FACE_C.x;
        l.y += dy;
    }
    return std::make_shared<const Pose>(p);
}

Hand peace_hand(double x, double y = 420) { return make_hand(PEACE, 0, {x, y}, 90); }
Hand thumbs_hand(double x, double y = 420) { return make_hand(FIST, 0, {x, y}, 90, "Right", "Thumb_Up"); }

using Poses = std::vector<std::shared_ptr<const Pose>>;

const Frame& by_face_x(const std::vector<Frame>& people, bool max) {
    const Frame* best = &people[0];
    for (const auto& q : people)
        if (max ? q.face->pts[1][0] > best->face->pts[1][0] : q.face->pts[1][0] < best->face->pts[1][0]) best = &q;
    return *best;
}

std::map<long, int> pid_by_x(const std::vector<Frame>& people) {
    std::map<long, int> out;
    for (const auto& q : people) out[std::lround(q.face->pts[1][0])] = q.pid;
    return out;
}

const Config& v1_config() {
    static Config cfg = load_config(project_path("config/config.toml"));
    return cfg;
}

Frame person(const Face& f, int pid) {
    Frame q;
    q.face = f;
    q.pid = pid;
    q.width = W;
    q.height = H;
    return q;
}

Frame crowd_frame(std::vector<Frame> people) {
    Frame top;
    top.width = W;
    top.height = H;
    if (!people.empty()) top.pid = people[0].pid;
    top.people = std::move(people);
    return top;
}

double seen_together(Identities& ids, int frames = 3) {
    for (int k = 0; k < frames; ++k) {
        auto p = group_people({face_at(LEFT), face_at(RIGHT)}, {}, {}, W, 3, 0.08, ids.recent(k / 30.0));
        ids.assign(p, k / 30.0);
    }
    return frames / 30.0;
}

}  // namespace

TEST_CASE("hands go to the person whose wrist is nearest", "[people]") {
    Hand h_left = make_hand(OPEN, 0, {LEFT + 150, 560}, 90);
    Hand h_right = make_hand(OPEN, 0, {RIGHT - 150, 560}, 90);
    auto people = group_people({face_at(LEFT), face_at(RIGHT)}, {pose_at(RIGHT), pose_at(LEFT)}, {h_right, h_left}, W);
    REQUIRE(people.size() == 2);
    const Frame& l = by_face_x(people, false);
    const Frame& r = by_face_x(people, true);
    CHECK((*l.pose)[0].x == LEFT);
    CHECK((*r.pose)[0].x == RIGHT);
    CHECK(same_hands(l.hands, {h_left}));
    CHECK(same_hands(r.hands, {h_right}));
}

TEST_CASE("hands fall back to the nearest face and far hands are dropped", "[people]") {
    Hand near = make_hand(PEACE, 0, {RIGHT + 110, 360}, 60);
    Hand far = make_hand(OPEN, 0, {640, 700}, 60);
    auto people = group_people({face_at(LEFT, 120), face_at(RIGHT, 120)}, {}, {near, far}, W);
    CHECK(same_hands(by_face_x(people, true).hands, {near}));
    CHECK(by_face_x(people, false).hands.empty());
}

TEST_CASE("single person keeps every hand", "[people]") {
    Hand far = make_hand(OPEN, 0, {1200, 700}, 90);
    auto people = group_people({face_at(LEFT)}, {}, {far}, W);
    REQUIRE(people.size() == 1);
    CHECK(same_hands(people[0].hands, {far}));
}

TEST_CASE("hands without anyone form a person", "[people]") {
    Hand h = make_hand(OPEN, 0, {600, 500});
    auto people = group_people({}, {}, {h}, W);
    REQUIRE(people.size() == 1);
    CHECK(same_hands(people[0].hands, {h}));
    CHECK_FALSE(people[0].face);
}

TEST_CASE("background faces ignored and largest kept", "[people]") {
    auto people = group_people({face_at(1100, 70), face_at(300, 150), face_at(640, 260), face_at(950, 180)}, {}, {}, W, 2);
    REQUIRE(people.size() == 2);
    CHECK(std::lround(people[0].face->width()) == 260);
    CHECK(std::lround(people[1].face->width()) == 180);
}

TEST_CASE("lagging pose still pairs with its face", "[people]") {
    auto people = group_people({face_at(LEFT)}, {pose_at(LEFT + 100)}, {}, W);
    REQUIRE(people.size() == 1);
    CHECK(people[0].pose);
}

TEST_CASE("pose without face is its own person", "[people]") {
    auto people = group_people({face_at(LEFT)}, {pose_at(LEFT), pose_at(RIGHT)}, {}, W);
    REQUIRE(people.size() == 2);
    CHECK(int(!people[0].face) + int(!people[1].face) == 1);
}

TEST_CASE("ids stable when people move and swap order", "[people]") {
    Identities ids;
    auto a = group_people({face_at(LEFT), face_at(RIGHT)}, {}, {}, W);
    ids.assign(a, 0.0);
    auto first = pid_by_x(a);
    CHECK(((first[long(LEFT)] == 1 && first[long(RIGHT)] == 2) || (first[long(LEFT)] == 2 && first[long(RIGHT)] == 1)));
    for (int k = 1; k < 20; ++k) {
        auto b = group_people({face_at(RIGHT - 8 * k), face_at(LEFT + 8 * k, 200 + k)}, {}, {}, W);
        ids.assign(b, k / 30.0);
        std::map<int, double> got;
        for (const auto& q : b) got[q.pid] = q.face->pts[1][0];
        CHECK(got[first[long(LEFT)]] == LEFT + 8 * k);
        CHECK(got[first[long(RIGHT)]] == RIGHT - 8 * k);
    }
}

TEST_CASE("new person gets new id and ids expire", "[people]") {
    Identities ids(ID_MAX_JUMP, 0.4);
    auto a = group_people({face_at(LEFT)}, {}, {}, W);
    ids.assign(a, 0.0);
    CHECK(a[0].pid == 1);
    auto b = group_people({face_at(LEFT), face_at(RIGHT)}, {}, {}, W);
    ids.assign(b, 0.1);
    CHECK(pid_by_x(b) == std::map<long, int>{{long(LEFT), 1}, {long(RIGHT), 2}});
    auto c = group_people({face_at(RIGHT)}, {}, {}, W);
    ids.assign(c, 0.3);
    CHECK(c[0].pid == 2);
    CHECK(ids.renamed.empty());
    auto d = group_people({face_at(RIGHT), face_at(LEFT)}, {}, {}, W);
    ids.assign(d, 0.4);
    CHECK(pid_by_x(d) == std::map<long, int>{{long(LEFT), 1}, {long(RIGHT), 2}});
    for (double t : {0.6, 0.7}) {
        auto e = group_people({face_at(RIGHT)}, {}, {}, W);
        ids.assign(e, t);
        CHECK(e[0].pid == 2);
    }
    auto e = group_people({face_at(RIGHT)}, {}, {}, W);
    ids.assign(e, 0.9);
    CHECK(e[0].pid == 1);
    CHECK(ids.renamed == std::map<int, int>{{2, 1}});
    auto f = group_people({face_at(RIGHT), face_at(LEFT)}, {}, {}, W);
    ids.assign(f, 1.0);
    CHECK(pid_by_x(f) == std::map<long, int>{{long(RIGHT), 1}, {long(LEFT), 2}});
}

TEST_CASE("face after hands only keeps the id", "[people]") {
    Identities ids;
    auto a = group_people({}, {}, {make_hand(OPEN, 0, {600, 500})}, W);
    ids.assign(a, 0.0);
    auto b = group_people({face_at(LEFT)}, {}, {}, W);
    ids.assign(b, 0.05);
    CHECK(a[0].pid == 1);
    CHECK(b[0].pid == 1);
}

TEST_CASE("hands of a person who dropped out are not given to the other", "[people]") {
    Identities ids(ID_MAX_JUMP, 1.0);
    double t = seen_together(ids);
    Hand mine = make_hand(OPEN, 0, {LEFT + 60, 520}, 90);
    Hand theirs = make_hand(OPEN, 0, {RIGHT - 40, 500}, 90);
    auto people = group_people({face_at(LEFT)}, {}, {mine, theirs}, W, 3, 0.08, ids.recent(t));
    ids.assign(people, t);
    REQUIRE(people.size() == 1);
    CHECK(people[0].pid == 1);
    CHECK(same_hands(people[0].hands, {mine}));
    // Same with poses: the missing person's last wrists claim the hand too.
    Identities ids2(ID_MAX_JUMP, 1.0);
    for (int k = 0; k < 3; ++k) {
        auto p = group_people({face_at(LEFT), face_at(RIGHT)}, {pose_at(LEFT), pose_at(RIGHT)}, {}, W, 3, 0.08,
                              ids2.recent(k / 30.0));
        ids2.assign(p, k / 30.0);
    }
    Hand on_wrist = make_hand(OPEN, 0, {RIGHT - 150, 560}, 90);
    auto p2 = group_people({face_at(LEFT)}, {pose_at(LEFT)}, {on_wrist}, W, 3, 0.08, ids2.recent(0.1));
    CHECK(p2[0].hands.empty());
    // Once the missing person is forgotten, the one left gets every hand again.
    auto p3 = group_people({face_at(LEFT)}, {}, {mine, theirs}, W, 3, 0.08, ids.recent(t + 1.5));
    CHECK(same_hands(p3[0].hands, {mine, theirs}));
}

TEST_CASE("recent tracks remember head and wrists", "[people]") {
    Identities ids(ID_MAX_JUMP, 1.0);
    for (int k = 0; k < 3; ++k) {
        auto p = group_people({face_at(LEFT)}, {pose_at(LEFT)}, {}, W);
        ids.assign(p, k / 30.0);
    }
    auto p = group_people({face_at(LEFT + 10)}, {}, {}, W);
    ids.assign(p, 0.1);
    auto rec = ids.recent(0.2);
    REQUIRE(rec.size() == 1);
    CHECK(rec[0].nose->x == LEFT + 10);
    CHECK(std::abs(rec[0].fw - 200) < 1);
    REQUIRE(rec[0].wrists);
    CHECK((*rec[0].wrists)[0].x == LEFT - 150);
    CHECK((*rec[0].wrists)[1].x == LEFT + 150);
    CHECK(ids.recent(1.2).empty());
}

TEST_CASE("one frame false face does not claim hands", "[people]") {
    Identities ids(ID_MAX_JUMP, 1.0);
    for (int k = 0; k < 5; ++k) {
        std::vector<Face> faces{face_at(LEFT)};
        if (k == 3) faces.push_back(face_at(RIGHT));
        auto p = group_people(faces, {}, {}, W, 3, 0.08, ids.recent(k / 30.0));
        ids.assign(p, k / 30.0);
    }
    Hand far = make_hand(OPEN, 0, {RIGHT, 500}, 90);
    auto people = group_people({face_at(LEFT)}, {}, {far}, W, 3, 0.08, ids.recent(5 / 30.0));
    CHECK(same_hands(people[0].hands, {far}));
}

TEST_CASE("stray pose wrist loses to the face next to the hand", "[people]") {
    Hand hand = make_hand(OPEN, 0, {RIGHT - 60, 450}, 90);
    auto stray = pose_at(LEFT, -150, RIGHT - 60 - LEFT, 450);
    auto people = group_people({face_at(LEFT), face_at(RIGHT)}, {stray}, {hand}, W);
    const Frame& l = by_face_x(people, false);
    const Frame& r = by_face_x(people, true);
    CHECK(l.pose);
    CHECK(same_hands(r.hands, {hand}));
    CHECK(l.hands.empty());
    Hand reach = make_hand(OPEN, 0, {LEFT + 300, 450}, 90);
    auto p2 = group_people({face_at(LEFT), face_at(RIGHT)}, {pose_at(LEFT, -150, 300, 450)}, {reach}, W);
    CHECK(same_hands(by_face_x(p2, false).hands, {reach}));
}

TEST_CASE("third hand of a full person is not given to the neighbour", "[people]") {
    Hand h1 = make_hand(OPEN, 0, {RIGHT - 150, 560}, 90);
    Hand h2 = make_hand(OPEN, 0, {RIGHT + 150, 560}, 90);
    Hand dup = make_hand(OPEN, 0, {RIGHT - 140, 540}, 90);
    auto people = group_people({face_at(LEFT), face_at(RIGHT)}, {pose_at(RIGHT)}, {h1, dup, h2}, W);
    CHECK(same_hands(by_face_x(people, true).hands, {h1, h2}));
    CHECK(by_face_x(people, false).hands.empty());
}

TEST_CASE("two thumbs up fire two memes with their own anchors", "[crowd]") {
    Crowd crowd(v1_config());
    Identities ids;
    auto motion = motion_frac(0.0);
    std::map<int, std::map<std::string, Detection>> dets;
    std::vector<Frame> people;
    for (int k = 0; k < 30; ++k) {
        double now = k / 30.0;
        people = group_people({face_at(LEFT), face_at(RIGHT)}, {}, {thumbs_hand(LEFT + 180), thumbs_hand(RIGHT - 180)}, W);
        ids.assign(people, now);
        for (auto& q : people) { q.motion = motion; q.width = W; q.height = H; }
        Frame fr;
        fr.hands = people[0].hands;
        fr.face = people[0].face;
        fr.pid = people[0].pid;
        fr.motion = motion;
        fr.width = W;
        fr.height = H;
        fr.people = people;
        dets = crowd.update(fr, now);
    }
    REQUIRE(dets.size() == 2);
    REQUIRE((dets.count(1) && dets.count(2)));
    CHECK(dets[1]["thumbs_up"].raw);
    CHECK(dets[2]["thumbs_up"].raw);
    CHECK(dets[1]["thumbs_up"].anchor != dets[2]["thumbs_up"].anchor);
    auto shown = crowd.snapshot();
    std::map<int, Box> anchors;
    for (const auto& s : shown) {
        CHECK(s.memes.at("thumbs_up").started);
        anchors[s.pid] = *s.memes.at("thumbs_up").anchor;
    }
    REQUIRE(anchors.size() == 2);
    CHECK(anchors[1] != anchors[2]);
    std::map<int, double> centre;
    for (const auto& q : people) centre[q.pid] = q.face->pts[1][0];
    for (auto [pid, b] : anchors) {
        double mid = (b.x + (b.x + b.width)) / 2.0;
        CHECK(std::abs(mid - centre[pid]) < std::abs(mid - centre[3 - pid]));
    }
}

TEST_CASE("one person's meme does not fire for the other", "[crowd]") {
    Crowd crowd(v1_config());
    Identities ids;
    auto motion = motion_frac(0.0);
    std::vector<Frame> people;
    for (int k = 0; k < 30; ++k) {
        people = group_people({face_at(LEFT), face_at(RIGHT)}, {}, {thumbs_hand(RIGHT - 180)}, W);
        ids.assign(people, k / 30.0);
        for (auto& q : people) { q.motion = motion; q.width = W; q.height = H; }
        Frame fr;
        fr.motion = motion;
        fr.width = W;
        fr.height = H;
        fr.people = people;
        crowd.update(fr, k / 30.0);
    }
    int right = 0;
    for (const auto& q : people)
        if (q.face->pts[1][0] == RIGHT) right = q.pid;
    std::map<int, bool> on;
    for (const auto& s : crowd.snapshot()) on[s.pid] = s.memes.at("thumbs_up").started.has_value();
    CHECK(on[right]);
    CHECK_FALSE(on[3 - right]);
}

TEST_CASE("snapshot is a copy", "[crowd]") {
    Crowd crowd(v1_config());
    Frame fr = person(face_at(LEFT), 1);
    fr.people = {fr};
    for (int k = 0; k < 16; ++k) crowd.update(fr, k / 30.0, {"thumbs_up"});
    auto shot = crowd.snapshot();
    CHECK(shot[0].memes.at("thumbs_up").started);
    crowd.people().at(1)->memes.at("thumbs_up").started.reset();
    CHECK(shot[0].memes.at("thumbs_up").started);
}

TEST_CASE("absent person is not shown or run and dropped after grace", "[crowd]") {
    Crowd crowd(v1_config());
    double forget = crowd.forget();
    CHECK(forget <= 1.0);
    Frame a = person(face_at(LEFT), 1), b = person(face_at(RIGHT), 2);
    for (int k = 0; k < 16; ++k) crowd.update(crowd_frame({a, b}), k / 30.0, {"thumbs_up"});
    std::set<int> pids;
    for (const auto& s : crowd.snapshot()) pids.insert(s.pid);
    CHECK(pids == std::set<int>{1, 2});
    double t = 0.5 + forget / 2;
    auto dets = crowd.update(crowd_frame({b}), t);
    CHECK((dets.size() == 1 && dets.count(2)));
    auto shot = crowd.snapshot();
    REQUIRE(shot.size() == 1);
    CHECK(shot[0].pid == 2);
    CHECK_FALSE(crowd.people().at(1)->head);
    CHECK(crowd.people().size() == 2);
    crowd.update(crowd_frame({b}), 0.5 + forget * 1.5);
    CHECK((crowd.people().size() == 1 && crowd.people().count(2)));
}

TEST_CASE("nobody in view shows nothing", "[crowd]") {
    Crowd crowd(v1_config());
    Frame a = person(face_at(LEFT), 1);
    for (int k = 0; k < 20; ++k) crowd.update(crowd_frame({a}), k / 30.0, {"thumbs_up"});
    CHECK(crowd.snapshot()[0].memes.at("thumbs_up").started);
    Frame empty;
    empty.width = W;
    empty.height = H;
    CHECK(crowd.update(empty, 0.7).empty());
    CHECK(crowd.snapshot().empty());
}

TEST_CASE("renamed ids carry their state", "[crowd]") {
    Crowd crowd(v1_config());
    crowd.update(crowd_frame({person(face_at(RIGHT), 2)}), 0.0);
    Person* st = crowd.people().at(2).get();
    Frame top = crowd_frame({person(face_at(RIGHT), 1)});
    top.renamed = {{2, 1}};
    crowd.update(top, 0.1);
    CHECK(crowd.people().size() == 1);
    CHECK(crowd.people().at(1).get() == st);
    CHECK(st->pid == 1);
}

// ---------------------------------------------------------------------------------------
// config (toml + v2 overlay, Python merge())
// ---------------------------------------------------------------------------------------

TEST_CASE("config loads v1 and merges v2 overlay", "[config]") {
    const Config& v1 = v1_config();
    CHECK(v1.memes.size() == 11);
    CHECK(v1.memes.front().name == "higuruma");
    CHECK(v1.memes.back().name == "sixty_seven");
    CHECK(v1.thr("extra_gestures", 0.0) == 0.0);
    CHECK(v1.thr("bounce_speed", 0.0) == 1.0);
    CHECK(v1.meme_on_face);
    CHECK(v1.min_face_width == 0.08);
    CHECK(v1.forget == 1.0);
    const MemeSpec* hig = v1.meme("higuruma");
    REQUIRE(hig);
    CHECK(hig->frames == std::make_pair(114, 176));
    CHECK(hig->loop_from == 140);
    CHECK(v1.meme("dafoe")->hold == 0.45);
    CHECK(v1.meme("nerd")->title.empty());
    Config v2 = load_config("config/config.toml", "config/config_v2.toml");
    CHECK(v2.memes.size() == 13);
    CHECK(v2.memes[11].name == "godfather");
    CHECK(v2.memes[12].name == "looksmaxxing");
    CHECK(v2.thr("extra_gestures", 0.0) == 1.0);
    CHECK(v2.thr("jaw_dist", 0.0) == 0.12);
    CHECK(v2.thr("cover_dist", 0.0) == 0.75);   // base thresholds kept
    CHECK_THROWS(load_config("config/does_not_exist.toml"));
}
