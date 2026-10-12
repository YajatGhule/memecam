"""Multi-person grouping, identities and per-person meme state, on synthetic landmarks."""
from __future__ import annotations

import tomllib

import numpy as np
from test_gestures import FACE_C, OPEN, PEACE, make_face, make_hand, make_pose

from memecam.main import CONFIG, Crowd
from memecam.tracking import MOTION_H, MOTION_W, Frame, Identities, group_people

W, H = 1280, 720
LEFT, RIGHT = 330.0, 950.0          # the two people's face centres (x)


def face_at(x, w=200.0, y=FACE_C[1]):
    return make_face(center=(x, y), w=w, h=1.3 * w)


def pose_at(x, wl_dx=-150, wr_dx=150, wy=560, dy=0.0):
    """make_pose() moved so its nose sits under a face centred at x (wrists relative to that)."""
    p = make_pose((FACE_C[0] + wl_dx, wy), (FACE_C[0] + wr_dx, wy))
    p[:, 0] += x - FACE_C[0]
    p[:, 1] += dy
    return p


def peace_hand(x, y=420):
    return make_hand(PEACE, wrist=(x, y), size=90)


def test_hands_go_to_the_person_whose_wrist_is_nearest():
    faces = [face_at(LEFT), face_at(RIGHT)]
    poses = [pose_at(RIGHT), pose_at(LEFT)]
    h_left = make_hand(OPEN, wrist=(LEFT + 150, 560), size=90)      # on the left person's right wrist
    h_right = make_hand(OPEN, wrist=(RIGHT - 150, 560), size=90)    # on the right person's left wrist
    people = group_people(faces, poses, [h_right, h_left], W)
    assert len(people) == 2
    by_x = sorted(people, key=lambda q: q.face.pts[1, 0])
    assert by_x[0].pose[0, 0] == LEFT and by_x[1].pose[0, 0] == RIGHT
    assert by_x[0].hands == [h_left] and by_x[1].hands == [h_right]


def test_hands_fall_back_to_the_nearest_face_and_far_hands_are_dropped():
    faces = [face_at(LEFT, w=120), face_at(RIGHT, w=120)]
    near = make_hand(PEACE, wrist=(RIGHT + 110, 360), size=60)
    far = make_hand(OPEN, wrist=(640, 700), size=60)              # nobody's (> 3.5 face-widths away)
    people = group_people(faces, [], [near, far], W)
    right = max(people, key=lambda q: q.face.pts[1, 0])
    left = min(people, key=lambda q: q.face.pts[1, 0])
    assert right.hands == [near] and left.hands == []


def test_single_person_keeps_every_hand():
    """With one person (the old single-person path) a hand far from the face is still theirs."""
    far = make_hand(OPEN, wrist=(1200, 700), size=90)
    people = group_people([face_at(LEFT)], [], [far], W)
    assert len(people) == 1 and people[0].hands == [far]


def test_hands_without_anyone_form_a_person():
    h = make_hand(OPEN, wrist=(600, 500))
    people = group_people([], [], [h], W)
    assert len(people) == 1 and people[0].hands == [h] and people[0].face is None


def test_background_faces_ignored_and_largest_kept():
    small = face_at(1100, w=70)                 # 0.055 of the frame width: someone in the back row
    faces = [small, face_at(300, w=150), face_at(640, w=260), face_at(950, w=180)]
    people = group_people(faces, [], [], W, max_people=2)
    assert [round(q.face.width) for q in people] == [260, 180]


def test_lagging_pose_still_pairs_with_its_face():
    """Async pose is a frame old: the nose can be half a face-width off."""
    people = group_people([face_at(LEFT)], [pose_at(LEFT + 100)], [], W)
    assert len(people) == 1 and people[0].pose is not None


def test_pose_without_face_is_its_own_person():
    """Hands over the face: face mesh gone, the body pose still knows where the person is."""
    people = group_people([face_at(LEFT)], [pose_at(LEFT), pose_at(RIGHT)], [], W)
    assert len(people) == 2
    assert sum(q.face is None for q in people) == 1


def test_ids_stable_when_people_move_and_swap_order():
    ids = Identities()
    a = group_people([face_at(LEFT), face_at(RIGHT)], [], [], W)
    ids.assign(a, 0.0)
    first = {round(q.face.pts[1, 0]): q.pid for q in a}
    assert sorted(first.values()) == [1, 2]
    for k in range(1, 20):              # both drift towards each other, list order reversed
        b = group_people([face_at(RIGHT - 8 * k), face_at(LEFT + 8 * k, w=200 + k)], [], [], W)
        ids.assign(b, k / 30)
        got = {q.pid: q.face.pts[1, 0] for q in b}
        assert got[first[round(LEFT)]] == LEFT + 8 * k
        assert got[first[round(RIGHT)]] == RIGHT - 8 * k


def test_new_person_gets_new_id_and_ids_expire():
    ids = Identities(forget=2.0)
    a = group_people([face_at(LEFT)], [], [], W)
    ids.assign(a, 0.0)
    b = group_people([face_at(LEFT), face_at(RIGHT)], [], [], W)
    ids.assign(b, 0.1)
    assert sorted(q.pid for q in b) == [1, 2]
    ids.assign(group_people([face_at(RIGHT)], [], [], W), 0.2)
    c = group_people([face_at(RIGHT)], [], [], W)
    ids.assign(c, 2.15)
    assert c[0].pid == 2                        # person 2 kept their id...
    d = group_people([face_at(RIGHT), face_at(LEFT)], [], [], W)
    ids.assign(d, 2.2)
    assert {q.pid for q in d} == {1, 2}         # ...and id 1 (gone > 2 s) is free again


def test_two_peace_signs_fire_two_memes_with_their_own_anchors():
    cfg = tomllib.loads(CONFIG.read_text())
    crowd = Crowd(cfg, load_media=False)
    ids = Identities()
    motion = np.zeros((MOTION_H, MOTION_W), bool)
    for k in range(30):
        now = k / 30
        hands = [peace_hand(LEFT + 180), peace_hand(RIGHT - 180)]
        people = group_people([face_at(LEFT), face_at(RIGHT)], [], hands, W)
        ids.assign(people, now)
        for q in people:
            q.motion, q.width, q.height = motion, W, H
        fr = Frame(hands=people[0].hands, face=people[0].face, pid=people[0].pid, motion=motion,
                   width=W, height=H, people=people)
        dets = crowd.update(fr, now)
    assert set(dets) == {1, 2}
    assert dets[1]["peace"].raw and dets[2]["peace"].raw
    assert dets[1]["peace"].anchor != dets[2]["peace"].anchor
    shown = crowd.snapshot()
    on = {s.pid: s.memes["peace"].started is not None for s in shown}
    assert on == {1: True, 2: True}
    anchors = {s.pid: s.memes["peace"].anchor for s in shown}
    assert anchors[1] != anchors[2]
    # each anchor sits on its own person's side
    centre = {q.pid: q.face.pts[1, 0] for q in people}
    for pid, (x0, _, x1, _) in anchors.items():
        other = 3 - pid
        assert abs((x0 + x1) / 2 - centre[pid]) < abs((x0 + x1) / 2 - centre[other])


def test_one_persons_meme_does_not_fire_for_the_other():
    cfg = tomllib.loads(CONFIG.read_text())
    crowd = Crowd(cfg, load_media=False)
    ids = Identities()
    motion = np.zeros((MOTION_H, MOTION_W), bool)
    for k in range(30):
        people = group_people([face_at(LEFT), face_at(RIGHT)], [], [peace_hand(RIGHT - 180)], W)
        ids.assign(people, k / 30)
        for q in people:
            q.motion, q.width, q.height = motion, W, H
        crowd.update(Frame(motion=motion, width=W, height=H, people=people), k / 30)
    right = next(q.pid for q in people if q.face.pts[1, 0] == RIGHT)
    on = {s.pid: s.memes["peace"].started is not None for s in crowd.snapshot()}
    assert on[right] and not on[3 - right]


def test_snapshot_is_a_copy():
    cfg = tomllib.loads(CONFIG.read_text())
    crowd = Crowd(cfg, load_media=False)
    fr = Frame(face=face_at(LEFT), pid=1, width=W, height=H)
    fr.people = [fr]
    crowd.update(fr, 0.0, force=("peace",))
    crowd.update(fr, 0.5, force=("peace",))
    shot = crowd.snapshot()
    assert shot[0].memes["peace"].started is not None
    crowd.people[1].memes["peace"].started = None
    assert shot[0].memes["peace"].started is not None


def test_state_dropped_after_person_leaves():
    cfg = tomllib.loads(CONFIG.read_text())
    crowd = Crowd(cfg, load_media=False)
    a, b = Frame(face=face_at(LEFT), pid=1, width=W, height=H), Frame(face=face_at(RIGHT), pid=2, width=W, height=H)
    crowd.update(Frame(width=W, height=H, people=[a, b]), 0.0)
    assert set(crowd.people) == {1, 2}
    crowd.update(Frame(width=W, height=H, people=[a]), 1.0)
    assert set(crowd.people) == {1, 2}
    crowd.update(Frame(width=W, height=H, people=[a]), 2.5)
    assert set(crowd.people) == {1}
