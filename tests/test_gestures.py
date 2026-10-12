"""Gesture logic tests on synthetic MediaPipe-style landmarks.

Hands are built in a local "hand frame" (u = along the fingers, v = across the palm, z = depth,
unit = wrist→middle-MCP distance), then rotated in the image plane, scaled to pixels and placed.
Rotation 0° = fingers pointing up in the image, 90° = pointing right, 180° = pointing down.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from memecam.gestures import DEFAULTS, Detectors, Trigger, finger_states
from memecam.tracking import MOTION_H, MOTION_W, Face, Frame, Hand

RATIO = DEFAULTS["finger_extended_ratio"]

# ---------------------------------------------------------------------------------------
# synthetic hand
# ---------------------------------------------------------------------------------------

# finger: (mcp index, (v, u) of the MCP, segment lengths proximal/middle/distal, spread)
_FINGERS = {
    "index": (5, (0.30, 0.95), (0.45, 0.27, 0.22), 0.25),   # spread like a real ✌ (tip gap ≈ 0.6)
    "middle": (9, (0.10, 1.00), (0.50, 0.30, 0.23), 0.0),
    "ring": (13, (-0.10, 0.95), (0.46, 0.28, 0.22), -0.05),
    "pinky": (17, (-0.28, 0.85), (0.36, 0.20, 0.20), -0.12),
}
# curled (fist, palm towards camera): proximal phalanx points at the camera, the rest folds down
_CURL_DIRS = [(0.0, 0.20, -0.98), (0.0, -0.90, -0.30), (0.0, -0.80, 0.60)]  # (v, u, z)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_hand(extended=(), angle=0.0, wrist=(640, 400), size=110.0, label="Right",
              gesture="None", thumb_out=None) -> Hand:
    """`extended` = names of fingers (index/middle/ring/pinky) that are straight.
    `angle` = degrees clockwise from 'fingers pointing up' in the image."""
    extended = set(extended)
    local = np.zeros((21, 3))  # (v, u, z)
    for name, (mcp, (v, u), lengths, spread) in _FINGERS.items():
        local[mcp] = (v, u, 0.0)
        if name in extended:
            dirs = [_unit((spread, 1.0, 0.0))] * 3
        else:
            dirs = [_unit(d) for d in _CURL_DIRS]
        p = local[mcp].copy()
        for k, (seg, d) in enumerate(zip(lengths, dirs)):
            p = p + seg * d
            local[mcp + 1 + k] = p
    # thumb: out to the side for an open hand, tucked across the fingers for a fist
    if thumb_out is None:
        thumb_out = len(extended) >= 3
    if thumb_out:
        local[1:5] = [(0.25, 0.20, 0), (0.50, 0.40, 0), (0.70, 0.55, 0), (0.85, 0.70, 0)]
    else:
        local[1:5] = [(0.25, 0.20, 0), (0.40, 0.40, -0.1), (0.30, 0.60, -0.3), (0.12, 0.70, -0.4)]

    a = math.radians(angle)
    up = np.array([math.sin(a), -math.cos(a)])        # image direction of +u
    lat = np.array([math.cos(a), math.sin(a)])        # image direction of +v
    pts = np.zeros((21, 3))
    pts[:, :2] = np.asarray(wrist, float) + size * (local[:, :1] * lat + local[:, 1:2] * up)
    pts[:, 2] = size * local[:, 2]
    return Hand(pts, label, gesture, 0.9 if gesture != "None" else 0.0)


OPEN = ("index", "middle", "ring", "pinky")
FIST = ()
PEACE = ("index", "middle")
MIDDLE = ("middle",)
INDEX = ("index",)

# ---------------------------------------------------------------------------------------
# synthetic face
# ---------------------------------------------------------------------------------------

FACE_C = (640.0, 300.0)
FACE_W, FACE_H = 200.0, 260.0


def make_face(nose_ratio=0.6, blend=None, center=FACE_C, w=FACE_W, h=FACE_H, mouth=0.02) -> Face:
    """`mouth` = lip gap as a fraction of the face height (vary it between frames = talking)."""
    cx, cy = center
    top, chin = cy - 0.5 * h, cy + 0.5 * h
    rng = np.random.default_rng(0)
    # fill everything inside the face ellipse (slightly inset so the named points define the bbox)
    t = rng.uniform(0, 2 * math.pi, 478)
    r = np.sqrt(rng.uniform(0, 1, 478)) * 0.9
    pts = np.stack([cx + r * np.cos(t) * w / 2, cy + r * np.sin(t) * h / 2, rng.normal(0, 5, 478)], 1)
    pts[10] = (cx, top, 0)
    pts[152] = (cx, chin, 0)
    pts[1] = (cx, top + nose_ratio * (chin - top), -30)
    pts[13] = (cx, top + 0.78 * (chin - top), -10)     # upper lip
    pts[14] = (cx, top + (0.78 + mouth) * (chin - top), -10)   # lower lip
    pts[234] = (cx - w / 2, cy, 10)
    pts[454] = (cx + w / 2, cy, 10)
    return Face(pts, dict(blend or {}))


def run(det, hands=(), face=None, now=0.0, motion=0.0):
    """`motion` = fraction of the frame changed since the previous frame (frame differencing)."""
    m = np.zeros((MOTION_H, MOTION_W), bool)
    m.flat[:int(motion * m.size)] = True
    return det.run(Frame(hands=list(hands), face=face, motion=m, width=1280, height=720), now)


def raws(d):
    return {k for k, v in d.items() if v.raw}


ANGLES = [0, 45, -45, 90, -90, 30]

# ---------------------------------------------------------------------------------------
# finger_states
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("angle", ANGLES + [180])
@pytest.mark.parametrize("ext", [OPEN, FIST, PEACE, MIDDLE, INDEX], ids=["open", "fist", "peace", "middle", "index"])
def test_finger_states_rotation_invariant(ext, angle):
    s = finger_states(make_hand(ext, angle=angle), RATIO)
    assert s == {f: f in ext for f in OPEN}


@pytest.mark.parametrize("size", [40.0, 110.0, 300.0])
def test_finger_states_scale_invariant(size):
    s = finger_states(make_hand(PEACE, size=size), RATIO)
    assert s == {"index": True, "middle": True, "ring": False, "pinky": False}


def test_extended_ratio_has_margin():
    """Straight fingers should clear the threshold comfortably, curled ones fall well below."""
    for ext in (OPEN, FIST):
        h = make_hand(ext)
        for _, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            r = np.linalg.norm(h.pts[tip] - h.pts[0]) / np.linalg.norm(h.pts[pip] - h.pts[0])
            if ext:
                assert r > RATIO + 0.1
            else:
                assert r < RATIO - 0.3


# ---------------------------------------------------------------------------------------
# single-hand pose exclusivity
# ---------------------------------------------------------------------------------------

HAND_POSES = {"middle_finger", "thumbs_up", "nerd"}


@pytest.mark.parametrize("angle", [0, 20, -20])
def test_middle_finger_only(angle):
    d = run(Detectors({}), [make_hand(MIDDLE, angle=angle)])
    assert raws(d) & HAND_POSES == {"middle_finger"}


def thumbs_up_hand(wrist=(640, 400), size=110.0, gesture="None"):
    """Fist with the thumb sticking straight up."""
    h = make_hand(FIST, wrist=wrist, size=size, gesture=gesture)
    x, y = wrist
    h.pts[1:5, :2] = [(x - 0.35 * size, y - 0.25 * size), (x - 0.45 * size, y - 0.55 * size),
                      (x - 0.50 * size, y - 0.85 * size), (x - 0.52 * size, y - 1.10 * size)]
    return h


@pytest.mark.parametrize("angle", [0, 30, -30, 90])
def test_v_sign_fires_nothing(angle):
    """The peace sign no longer has a meme (piece of cake was removed)."""
    d = run(Detectors({}), [make_hand(PEACE, angle=angle)])
    assert raws(d) & HAND_POSES == set()


def test_thumbs_up_from_geometry():
    d = run(Detectors({}), [thumbs_up_hand()])
    assert raws(d) & HAND_POSES == {"thumbs_up"}
    assert not d["mahoraga"].raw


def test_thumbs_up_from_mediapipe_label():
    d = run(Detectors({}), [make_hand(FIST, gesture="Thumb_Up")])
    assert d["thumbs_up"].raw


def test_plain_fist_is_not_thumbs_up():
    assert not run(Detectors({}), [make_hand(FIST)])["thumbs_up"].raw


def test_fist_label_vetoes_geometric_thumbs_up():
    """MediaPipe calling it a Closed_Fist (thumb resting on top of the fist) beats the geometry."""
    assert not run(Detectors({}), [thumbs_up_hand(gesture="Closed_Fist")])["thumbs_up"].raw


@pytest.mark.parametrize("angle", [0, 30, -30, 45])
def test_index_up_is_nerd(angle):
    d = run(Detectors({}), [make_hand(INDEX, angle=angle)])
    assert raws(d) & HAND_POSES == {"nerd"}


@pytest.mark.parametrize("angle", [180, 150, -150, 135])
def test_index_pointing_down_is_not_nerd(angle):
    d = run(Detectors({}), [make_hand(INDEX, angle=angle)])
    assert not raws(d) & HAND_POSES


@pytest.mark.parametrize("angle", [90, -90, 80, -80])
def test_index_pointing_sideways_is_not_nerd(angle):
    """Pointing at something beside you (e.g. at chat on screen) shouldn't be the nerd meme."""
    d = run(Detectors({}), [make_hand(INDEX, angle=angle)])
    assert "nerd" not in raws(d)


def test_open_palm_and_fist_trigger_nothing():
    for ext in (OPEN, FIST):
        d = run(Detectors({}), [make_hand(ext)])
        assert not raws(d)


def test_victory_gesture_label_is_peace_not_nerd():
    """MediaPipe's canned gesture says Victory but our finger states only see the index finger
    (middle tip slightly occluded): it must not fire peace AND nerd at once."""
    d = run(Detectors({}), [make_hand(INDEX, gesture="Victory")])
    assert len(raws(d) & HAND_POSES) <= 1


def test_pointing_up_label_with_peace_fingers_is_one_pose():
    d = run(Detectors({}), [make_hand(PEACE, gesture="Pointing_Up")])
    assert len(raws(d) & HAND_POSES) <= 1


# ---------------------------------------------------------------------------------------
# two-hand poses
# ---------------------------------------------------------------------------------------


def cinema_hands(y=380):
    # wrists beside the head, palms open, fingers tilted slightly outwards
    return [make_hand(OPEN, angle=-15, wrist=(420, y), label="Left"),
            make_hand(OPEN, angle=15, wrist=(860, y), label="Right")]


def test_absolute_cinema_above_chin():
    d = run(Detectors({}), cinema_hands(), make_face())
    assert d["absolute_cinema"].raw
    assert not d["mahoraga"].raw


def test_absolute_cinema_below_chin_is_false():
    d = run(Detectors({}), cinema_hands(y=650), make_face())
    assert not d["absolute_cinema"].raw


def test_absolute_cinema_needs_hands_outside_face():
    hands = [make_hand(OPEN, wrist=(600, 380), label="Left"), make_hand(OPEN, wrist=(860, 380))]
    d = run(Detectors({}), hands, make_face())
    assert not d["absolute_cinema"].raw


def test_absolute_cinema_needs_open_palms():
    hands = [make_hand(FIST, wrist=(420, 380), label="Left"), make_hand(OPEN, wrist=(860, 380))]
    d = run(Detectors({}), hands, make_face())
    assert not d["absolute_cinema"].raw


def mahoraga_hands(gap=(140, 0), left_ext=FIST, right_ext=FIST, y=640):
    """Megumi's sign as filmed: two fists, forearms level, knuckles touching in front of the chest."""
    left = make_hand(left_ext, angle=90, wrist=(560, y), label="Left")      # fingers pointing right
    right = make_hand(right_ext, angle=-90, wrist=(560 + gap[0], y + gap[1]), label="Right")
    return [left, right]


def test_mahoraga_two_fists():
    d = run(Detectors({}), mahoraga_hands(), make_face())
    assert d["mahoraga"].raw
    assert not d["absolute_cinema"].raw


def test_mahoraga_without_face():
    assert run(Detectors({}), mahoraga_hands())["mahoraga"].raw


def test_mahoraga_far_apart_is_false():
    d = run(Detectors({}), mahoraga_hands(gap=(600, 0)), make_face())
    assert not d["mahoraga"].raw


def test_mahoraga_not_during_cinema():
    d = run(Detectors({}), cinema_hands(), make_face())
    assert not d["mahoraga"].raw


def test_mahoraga_needs_two_fists():
    d = run(Detectors({}), mahoraga_hands(right_ext=OPEN), make_face())
    assert not d["mahoraga"].raw


def test_mahoraga_not_at_chin():
    """Two fists under the chin (resting your head on your hands) isn't Megumi's sign."""
    d = run(Detectors({}), mahoraga_hands(y=460), make_face())
    assert not d["mahoraga"].raw


# ---------------------------------------------------------------------------------------
# hands over the face: scuba cat (one hand), higuruma (both)
# ---------------------------------------------------------------------------------------


def covering_hand(dx=0.0, ext=OPEN, label="Right"):
    """Open palm over the mouth: palm centre lands just below the nose."""
    return make_hand(ext, wrist=(640 + dx, 470), size=110.0, label=label)


def test_scuba_cat_one_hand_over_mouth():
    """One hand stuck on the face while the rest of you moves (the other hand swimming, blurred)."""
    det = Detectors({})
    res = [run(det, [covering_hand()], make_face(), now=i / 30, motion=0.05) for i in range(10)]
    assert raws(res[-1]) == {"scuba_cat"}


def test_scuba_cat_other_hand_waving_is_not_wall():
    det = Detectors({})
    res = []
    for i in range(30):
        wav = make_hand(OPEN, wrist=(150 + (i % 8) * 40, 520), label="Left")
        res.append(run(det, [covering_hand(), wav], make_face(), now=i / 30))
    assert all(r["scuba_cat"].raw for r in res[5:])
    assert not any(r["talking_to_wall"].raw for r in res)


def test_higuruma_both_hands_over_face():
    d = run(Detectors({}), [covering_hand(-30, label="Left"), covering_hand(30)], make_face())
    assert d["higuruma"].raw and not d["scuba_cat"].raw


def test_hand_far_from_face_is_not_covering():
    d = run(Detectors({}), [make_hand(OPEN, wrist=(200, 650))], make_face())
    assert not d["scuba_cat"].raw and not d["higuruma"].raw


def test_finger_sign_by_the_face_is_not_covering():
    """A finger on your nose isn't covering the face (and isn't the nerd either: that's beside it)."""
    d = run(Detectors({}), [make_hand(INDEX, wrist=(640, 470))], make_face())
    assert not d["scuba_cat"].raw and not d["higuruma"].raw and not d["nerd"].raw


def test_nerd_beside_face():
    d = run(Detectors({}), [make_hand(INDEX, wrist=(820, 420))], make_face())
    assert raws(d) == {"nerd"}


def test_one_hand_resting_on_face_still_is_not_higuruma():
    """Higuruma needs both hands on the face; one still hand never turns into it."""
    det = Detectors({})
    res = [run(det, [covering_hand()], make_face(), now=i / 30) for i in range(90)]
    assert not any(r["higuruma"].raw for r in res)


def test_scuba_never_switches_to_higuruma():
    det = Detectors({})
    res = [run(det, [covering_hand()], make_face(), now=i / 30, motion=0.05) for i in range(10)]
    res += [run(det, [covering_hand()], make_face(), now=(10 + i) / 30) for i in range(80)]
    assert res[9]["scuba_cat"].raw
    assert not any(r["higuruma"].raw for r in res)


# ---------------------------------------------------------------------------------------
# gojo: crossed index + middle
# ---------------------------------------------------------------------------------------


def crossed_hand(angle=0.0):
    h = make_hand(PEACE, angle=angle)
    # cross the fingers: swap the index and middle fingertip chains past each other
    h.pts[[7, 8, 11, 12]] = h.pts[[11, 12, 7, 8]]
    return h


@pytest.mark.parametrize("angle", [0, 20, -20])
def test_gojo_crossed_fingers(angle):
    d = run(Detectors({}), [crossed_hand(angle)], make_face())
    assert d["gojo"].raw
    assert not d["nerd"].raw


def test_v_sign_is_not_gojo():
    d = run(Detectors({}), [make_hand(PEACE)], make_face())
    assert not d["gojo"].raw and not d["nerd"].raw


def neutral(det, n=30, ratio=0.6, t0=0.0):
    for i in range(n):
        d = run(det, face=make_face(ratio), now=t0 + i / 30)
    return d


def test_dafoe_head_tilt():
    det = Detectors({})
    assert not neutral(det)["dafoe"].raw
    d = run(det, face=make_face(0.45), now=1.1)
    assert d["dafoe"].raw


def test_dafoe_small_jitter_does_not_fire():
    det = Detectors({})
    neutral(det)
    for i, r in enumerate([0.59, 0.61, 0.585, 0.6, 0.575]):
        assert not run(det, face=make_face(r), now=2 + i / 30)["dafoe"].raw


def test_dafoe_held_tilt_stays_on_for_a_while():
    det = Detectors({})
    neutral(det)
    for i in range(60):  # 2 s at 30 fps
        assert run(det, face=make_face(0.45), now=1 + i / 30)["dafoe"].raw


def test_dafoe_not_after_looking_down():
    """Looking down at the keyboard for a few seconds, then back at the screen, isn't a tilt up."""
    det = Detectors({})
    neutral(det)
    neutral(det, n=150, ratio=0.7, t0=1)        # 5 s looking down
    d = neutral(det, n=15, ratio=0.6, t0=6)     # back to neutral
    assert not d["dafoe"].raw


def test_dafoe_recovers_from_bad_initial_baseline():
    """App started while the user looked down: the first frame becomes the baseline. Once they
    look at the screen normally dafoe must not stay stuck on."""
    det = Detectors({})
    run(det, face=make_face(0.7), now=0)
    d = neutral(det, n=300, ratio=0.6, t0=0.1)   # 10 s of neutral
    assert not d["dafoe"].raw


def test_dafoe_calibrate_resets():
    det = Detectors({})
    neutral(det)
    det.calibrate()
    assert not run(det, face=make_face(0.5), now=2)["dafoe"].raw


# ---------------------------------------------------------------------------------------
# talking to a wall
# ---------------------------------------------------------------------------------------


def wave(det, ext=OPEN, frames=30, step=30.0, two=True, labels=("Left", "Right"), t0=0.0, angle=90.0):
    """Hands moving sideways while talking. `angle` 90 = fingers pointing sideways (hands down by the
    desk); 0 = hands held up with the fingers pointing up (palms out, see the palms-out tests)."""
    out = []
    for i in range(frames):
        phase = (i // 8) % 2
        dx = (i % 8) * step * (1 if phase == 0 else -1) + (0 if phase == 0 else 8 * step)
        hands = [make_hand(ext, angle=angle, wrist=(350 + dx, 550), label=labels[0])]
        if two:
            hands.append(make_hand(ext, angle=angle, wrist=(930 - dx, 550 + dx / 3), label=labels[1]))
        face = make_face(mouth=0.02 + 0.06 * (i % 2))                 # talking
        out.append(run(det, hands, face, now=t0 + i / 30, motion=min(0.3, step / 250)))
    return out


def test_talking_to_wall_fast_hands():
    res = wave(Detectors({}))
    assert not res[0]["talking_to_wall"].raw
    assert res[-1]["talking_to_wall"].raw
    assert sum(r["talking_to_wall"].raw for r in res[10:]) >= 14


def test_talking_to_wall_one_hand():
    assert wave(Detectors({}), two=False)[-1]["talking_to_wall"].raw


def test_talking_to_wall_stationary():
    res = wave(Detectors({}), step=0.0)
    assert not any(r["talking_to_wall"].raw for r in res)


def test_talking_to_wall_slow_drift():
    res = wave(Detectors({}), step=3.0)    # 90 px/s = 0.45 face widths/s
    assert not any(r["talking_to_wall"].raw for r in res)


def test_talking_to_wall_suppressed_by_middle_finger():
    res = wave(Detectors({}), ext=MIDDLE, two=False)
    assert all(r["middle_finger"].raw for r in res)
    assert not any(r["talking_to_wall"].raw for r in res)


def test_talking_to_wall_duplicate_handedness_labels():
    """MediaPipe sometimes labels both hands 'Right'. Two stationary hands must not look like
    one hand teleporting across the frame every frame."""
    res = wave(Detectors({}), step=0.0, labels=("Right", "Right"))
    assert not any(r["talking_to_wall"].raw for r in res)


def test_talking_to_wall_handedness_flip():
    """Handedness flipping between frames shouldn't create a speed spike either."""
    det = Detectors({})
    for i in range(30):
        labels = ("Left", "Right") if i % 2 else ("Right", "Left")
        hands = [make_hand(OPEN, wrist=(350, 550), label=labels[0]),
                 make_hand(OPEN, wrist=(930, 550), label=labels[1])]
        assert not run(det, hands, make_face(), now=i / 30)["talking_to_wall"].raw


def test_talking_to_wall_decays_after_stopping():
    det = Detectors({})
    wave(det)
    res = wave(det, step=0.0, t0=1.0)
    assert not res[-1]["talking_to_wall"].raw


def test_talking_to_wall_palms_out_while_talking():
    """Explaining with both open palms held up at chest height: fires while talking, even with the
    hands barely moving (no motion at all)."""
    res = wave(Detectors({}), step=0.0, angle=0.0)
    assert not res[0]["talking_to_wall"].raw
    assert all(r["talking_to_wall"].raw for r in res[3:])
    tr = Trigger(hold=0.5)
    assert any(tr.update(r["talking_to_wall"].raw, i / 30) for i, r in enumerate(res))


def test_palms_out_needs_talking():
    det = Detectors({})
    hands = [make_hand(OPEN, wrist=(350, 550), label="Left"), make_hand(OPEN, wrist=(930, 550))]
    assert not any(run(det, hands, make_face(), now=i / 30)["talking_to_wall"].raw for i in range(30))


@pytest.mark.parametrize("kw", [
    dict(xs=(560, 720)),                 # both hands in front of the face / chest, not out to the sides
    dict(y=300),                         # up beside the head (that's absolute cinema territory)
    dict(angle=90),                      # fingers sideways: hands resting, not held up
    dict(two=False),                     # just one hand
    dict(ext=FIST),                      # fists
])
def test_palms_out_negatives(kw):
    xs, y, angle, two, ext = kw.get("xs", (350, 930)), kw.get("y", 550), kw.get("angle", 0.0), \
        kw.get("two", True), kw.get("ext", OPEN)
    det = Detectors({})
    res = []
    for i in range(30):
        hands = [make_hand(ext, angle=angle, wrist=(xs[0], y), label="Left")]
        if two:
            hands.append(make_hand(ext, angle=-angle, wrist=(xs[1], y)))
        res.append(run(det, hands, make_face(mouth=0.02 + 0.06 * (i % 2)), now=i / 30))
    assert not any(r["talking_to_wall"].raw for r in res), kw


def test_talking_to_wall_hysteresis():
    """Once the hands are flying, a dip in the motion doesn't end it; stopping does."""
    det = Detectors({})
    talk = lambda i: make_face(mouth=0.02 + 0.06 * (i % 2))
    res = [run(det, [], talk(i), now=i / 30, motion=0.12) for i in range(20)]
    assert res[-1]["talking_to_wall"].raw
    res = [run(det, [], talk(i), now=(20 + i) / 30, motion=0.05) for i in range(20)]
    assert all(r["talking_to_wall"].raw for r in res)          # 0.05 < motion threshold 0.07
    res = [run(det, [], talk(i), now=(40 + i) / 30, motion=0.0) for i in range(20)]
    assert not res[-1]["talking_to_wall"].raw
    # ...but that motion level alone never starts it
    det = Detectors({})
    assert not any(run(det, [], talk(i), now=i / 30, motion=0.05)["talking_to_wall"].raw for i in range(40))


def test_talking_to_wall_ignores_motion_beside_someone_else():
    """Only the motion around this person counts (someone next to them waving doesn't)."""
    face = make_face(center=(250.0, 300.0), w=150.0, h=195.0)
    m = np.zeros((MOTION_H, MOTION_W), bool)
    m[:, int(900 * MOTION_W / 1280):] = True                    # everything right of x = 900 moving
    det = Detectors({})
    for i in range(30):
        f = make_face(center=(250.0, 300.0), w=150.0, h=195.0, mouth=0.02 + 0.06 * (i % 2))
        d = det.run(Frame(face=f, motion=m, width=1280, height=720), i / 30)
        assert not d["talking_to_wall"].raw
    assert det.motion_level(Frame(face=face, motion=m, width=1280, height=720)) == 0.0


# ---------------------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------------------


def drive(trig, pattern, dt=0.05, t0=0.0):
    """pattern: list of (raw, seconds). Returns list of (t, active)."""
    out, i = [], 0
    for raw, secs in pattern:
        for _ in range(round(secs / dt)):
            t = round(t0 + i * dt, 6)
            out.append((t, trig.update(raw, t)))
            i += 1
    return out


def first_true(trace):
    return next((t for t, a in trace if a), None)


def test_trigger_needs_hold():
    tr = Trigger(hold=0.3, min_show=2.0, linger=0.6, cooldown=0.8)
    trace = drive(tr, [(True, 0.2), (False, 0.1), (True, 0.2)])
    assert not any(a for _, a in trace)
    tr = Trigger(hold=0.3)
    assert first_true(drive(tr, [(True, 1.0)])) == pytest.approx(0.3)


def test_trigger_min_show():
    tr = Trigger(hold=0.3, min_show=2.0, linger=0.6, cooldown=0.8)
    trace = drive(tr, [(True, 0.5), (False, 3.0)])
    on = [t for t, a in trace if a]
    assert on[0] == pytest.approx(0.3)
    assert 2.25 - 1e-9 <= on[-1] <= 2.3 + 1e-9  # ends ~0.3 + 2.0 (±1 frame of float slop)
    assert not tr.active


def test_trigger_linger_bridges_dropouts():
    tr = Trigger(hold=0.3, min_show=1.0, linger=0.6, cooldown=0.8)
    trace = drive(tr, [(True, 0.5), (True, 1.0), (False, 0.3), (True, 0.5), (False, 0.3)])
    assert all(a for t, a in trace if t >= 0.3)  # brief dropouts don't end it
    trace = drive(tr, [(False, 1.0)], t0=2.6)
    on = [t for t, a in trace if a]
    assert on and on[-1] < 2.3 + 0.6 + 0.05 + 1e-9
    assert not tr.active


def test_trigger_cooldown():
    tr = Trigger(hold=0.3, min_show=1.0, linger=0.2, cooldown=0.8)
    drive(tr, [(True, 0.5), (False, 1.0)])          # fires at 0.3, ends at 1.3
    assert not tr.active
    trace = drive(tr, [(True, 2.0)], t0=1.5)
    t_on = first_true(trace)
    assert t_on is not None and t_on >= 1.3 + 0.8 - 1e-9


def test_trigger_stays_while_raw():
    tr = Trigger(hold=0.3, min_show=1.0, linger=0.2, cooldown=0.8)
    trace = drive(tr, [(True, 5.0)])
    assert all(a for t, a in trace if t >= 0.3)


def test_talking_to_wall_hand_entering_frame():
    """A second hand appearing far from the first is a new hand, not a fast-moving one."""
    det = Detectors({})
    for i in range(10):
        run(det, [make_hand(OPEN, wrist=(350, 550))], make_face(), now=i / 30)
    for i in range(10, 20):
        hands = [make_hand(OPEN, wrist=(350, 550)), make_hand(OPEN, wrist=(930, 550))]
        d = run(det, hands, make_face(), now=i / 30)
        assert d["talking_to_wall"].score < 0.2


# ---------------------------------------------------------------------------------------
# 67: open palms at chest height bouncing up and down in alternation
# ---------------------------------------------------------------------------------------

SIX7_X = (480.0, 800.0)      # wrists, image x (left, right)
SIX7_Y = 560.0               # chest height (face centre 300, chin 430, shoulders 450)


def make_pose(wl, wr, vis=0.95) -> np.ndarray:
    """(33, 4) body pose matching make_face(): nose, mouth, ears, shoulders and both wrists."""
    p = np.zeros((33, 4))
    cx, cy = FACE_C
    p[0] = (cx, cy - 10, 0, vis)                    # nose
    p[9], p[10] = (cx - 20, cy + 40, 0, vis), (cx + 20, cy + 40, 0, vis)      # mouth
    half = FACE_W / 1.35 / 2                         # ear distance × 1.35 ≈ face width
    p[7], p[8] = (cx - half, cy, 0, vis), (cx + half, cy, 0, vis)            # ears
    p[11], p[12] = (cx - 120, 450, 0, vis), (cx + 120, 450, 0, vis)           # shoulders
    p[15], p[16] = (*wl, 0, vis), (*wr, 0, vis)                               # wrists
    return p


def bounce(det, freq=2.5, amp=50.0, mode="alternate", secs=1.5, fps=30, t0=0.0, pose=True,
           hands=True, ext=OPEN, xs=SIX7_X, y=SIX7_Y, repeat_pose=1, talk=False, motion=0.0):
    """Wrists bouncing vertically. mode: alternate (opposite phase), together, left_only, clap.
    `repeat_pose` > 1 hands the same pose array to that many frames in a row (async pose)."""
    out, p = [], None
    for i in range(int(secs * fps)):
        t = t0 + i / fps
        s = amp * math.sin(2 * math.pi * freq * t)
        xl, xr = xs
        yl, yr = {"alternate": (y + s, y - s), "together": (y + s, y + s), "left_only": (y + s, y),
                  "clap": (y, y)}[mode]
        if mode == "clap":     # hands swing in and meet in the middle
            k = (1 - math.cos(2 * math.pi * freq * t)) / 2
            mid = (xl + xr) / 2
            xl, xr = xl + k * (mid - 15 - xl), xr + k * (mid + 15 - xr)
        if pose and (p is None or i % repeat_pose == 0):
            p = make_pose((xl, yl), (xr, yr))
        hs = [make_hand(ext, wrist=(xl, yl), size=90, label="Left"),
              make_hand(ext, wrist=(xr, yr), size=90, label="Right")] if hands else []
        m = np.zeros((MOTION_H, MOTION_W), bool)
        m.flat[:int(motion * m.size)] = True
        face = make_face(mouth=0.02 + 0.06 * (i % 2)) if talk else make_face()
        out.append(det.run(Frame(hands=hs, face=face, pose=p if pose else None, motion=m,
                                 width=1280, height=720), t))
    return out


def fired(res, name="sixty_seven", hold=0.4, fps=30):
    tr = Trigger(hold=hold)
    return any(tr.update(r[name].raw, i / fps) for i, r in enumerate(res))


@pytest.mark.parametrize("freq", [2.0, 2.5, 3.0])
def test_sixty_seven_alternating_bounce(freq):
    res = bounce(Detectors({}), freq=freq)
    assert fired(res)
    assert res[-1]["sixty_seven"].raw
    assert not res[0]["sixty_seven"].raw


def test_sixty_seven_pose_only():
    """Fast hands blur out of the hand tracker: the pose wrists alone are enough."""
    assert fired(bounce(Detectors({}), hands=False))


def test_sixty_seven_hands_only():
    """No body pose: fall back to the two tracked hands, matched by x-order."""
    assert fired(bounce(Detectors({}), pose=False))


def test_sixty_seven_async_pose():
    """Live mode repeats the same pose array for a few frames; repeats must not read as 'stopped'."""
    assert fired(bounce(Detectors({}), hands=False, repeat_pose=2))
    assert fired(bounce(Detectors({}), hands=False, repeat_pose=3))


def test_sixty_seven_suppresses_other_memes():
    res = bounce(Detectors({}), talk=True, motion=0.15, secs=2.0)
    assert fired(res)
    assert not fired(res, "talking_to_wall", hold=0.5)        # its config hold
    on = next(i for i, r in enumerate(res) if r["sixty_seven"].raw)
    assert not any(r["talking_to_wall"].raw for r in res[on:])
    for k in ("higuruma", "scuba_cat", "mahoraga", "absolute_cinema"):
        assert not any(r[k].raw for r in res), k


@pytest.mark.parametrize("kw", [
    dict(amp=0.0),                                  # hands still
    dict(mode="together"),                          # both hands moving the same way
    dict(mode="left_only"),                         # one hand waving
    dict(mode="clap", amp=0.0),                     # clapping (horizontal)
    dict(mode="clap", freq=2.0),
    dict(amp=4.0),                                  # tiny jitter in opposite directions
    dict(freq=0.5, amp=40.0),                       # slow see-saw, not a bounce
    dict(ext=FIST),                                 # fists, not open palms
    dict(ext=PEACE),                                # finger signs
    dict(y=300.0),                                  # hands up at face height, not the chest
    dict(xs=(620.0, 660.0)),                        # hands together in the middle
])
def test_sixty_seven_negatives(kw):
    res = bounce(Detectors({}), **kw)
    assert not fired(res), kw


@pytest.mark.parametrize("kw", [dict(mode="together"), dict(mode="left_only"), dict(amp=0.0)])
def test_sixty_seven_negatives_pose_only(kw):
    assert not fired(bounce(Detectors({}), hands=False, **kw))


def test_sixty_seven_stops_after_bouncing():
    det = Detectors({})
    bounce(det)
    res = bounce(det, amp=0.0, t0=1.5, secs=1.5)
    assert not res[-1]["sixty_seven"].raw


# 67 from the motion alone: fast bouncing hands blur out of the hand tracker (and off the bottom of a
# close-up), and the pose wrists flicker, but the moving pixels left and right of the face follow them.

def motion_blobs(centers, size=120.0) -> np.ndarray:
    """Motion mask with a square of changed pixels (image px, `size` wide) at each (x, y) centre."""
    m = np.zeros((MOTION_H, MOTION_W), bool)
    sx, sy = MOTION_W / 1280, MOTION_H / 720
    for x, y in centers:
        m[max(0, int((y - size / 2) * sy)):int((y + size / 2) * sy),
          max(0, int((x - size / 2) * sx)):int((x + size / 2) * sx)] = True
    return m


def swing(det, mode="alternate", freq=2.0, amp=100.0, secs=1.5, fps=30, t0=0.0, hands=(), talk=False,
          xs=(390.0, 890.0), y=560.0):
    out = []
    for i in range(int(secs * fps)):
        t = t0 + i / fps
        s = amp * math.sin(2 * math.pi * freq * t)
        blobs = {"alternate": [(xs[0], y + s), (xs[1], y - s)], "together": [(xs[0], y + s), (xs[1], y + s)],
                 "one_side": [(xs[0], y + s)], "still": [(xs[0], y), (xs[1], y)]}[mode]
        face = make_face(mouth=0.02 + 0.06 * (i % 2)) if talk else make_face()
        hs = hands(i) if callable(hands) else list(hands)
        out.append(det.run(Frame(hands=hs, face=face, motion=motion_blobs(blobs), width=1280, height=720), t))
    return out


@pytest.mark.parametrize("freq", [1.5, 2.0, 3.0])
def test_sixty_seven_from_motion(freq):
    res = swing(Detectors({}), freq=freq)
    assert fired(res)
    assert not res[0]["sixty_seven"].raw


@pytest.mark.parametrize("kw", [
    dict(mode="together"),              # both hands going up and down together
    dict(mode="one_side"),              # one hand waving
    dict(mode="still"),                 # moving pixels that don't go up / down (e.g. hands shaking)
    dict(amp=15.0),                     # tiny opposite jitter
    dict(freq=0.4),                     # slow see-saw
    dict(xs=(150.0, 1130.0)),           # far out to the sides (someone else next to you)
])
def test_sixty_seven_from_motion_negatives(kw):
    assert not fired(swing(Detectors({}), **kw)), kw


def test_sixty_seven_from_motion_blocked_by_fists():
    fists = [make_hand(FIST, wrist=(390, 560), label="Left"), make_hand(FIST, wrist=(890, 560))]
    assert not fired(swing(Detectors({}), hands=fists))


def test_sixty_seven_from_motion_tolerates_misread_frames():
    """A blurred hand misread as a fist or a pointing finger for a frame now and then doesn't stop it."""
    def hands(i):
        if i % 6 == 0:
            return [make_hand(FIST, wrist=(390, 560), label="Left")]
        if i % 6 == 3:
            return [make_hand(INDEX, wrist=(890, 560))]
        return [make_hand(OPEN, wrist=(390, 560), label="Left")]
    assert fired(swing(Detectors({}), hands=hands))


def test_sixty_seven_from_motion_suppresses_talking():
    res = swing(Detectors({}), talk=True, secs=2.0)
    assert fired(res)
    on = next(i for i, r in enumerate(res) if r["sixty_seven"].raw)
    assert not any(r["talking_to_wall"].raw for r in res[on:])


def test_sixty_seven_from_motion_stops():
    det = Detectors({})
    swing(det)
    res = swing(det, mode="still", t0=1.5, secs=1.5)
    assert not res[-1]["sixty_seven"].raw


# ---------------------------------------------------------------------------------------
# v2 gestures: godfather 🤌 and looksmaxxing (only with extra_gestures on)
# ---------------------------------------------------------------------------------------

V2 = {"extra_gestures": True}


def pinched_hand(wrist=(820, 520), size=100.0):
    """All five fingertips bunched together, held up above the wrist."""
    h = make_hand(OPEN, wrist=wrist, size=size)
    x, y = wrist
    tip = np.array([x + 0.05 * size, y - 1.6 * size])
    for k, off in zip((4, 8, 12, 16, 20), ((-0.08, 0.05), (0.0, -0.06), (0.06, -0.04), (0.1, 0.02), (0.07, 0.08))):
        h.pts[k, :2] = tip + np.array(off) * size
    for k in (3, 7, 11, 15, 19):                       # distal joints just below the tips
        h.pts[k, :2] = h.pts[k + 1, :2] + np.array([0, 0.2 * size])
    return h


def jaw_hand(face):
    """Index finger laid along the jawline from the side (an "L" under the jaw)."""
    jaw = face.pts[150, :2]
    h = make_hand(INDEX, angle=60, wrist=(jaw[0] - 120, jaw[1] + 90), size=90)
    shift = jaw - h.pts[8, :2]
    h.pts[:, :2] += shift
    return h


def test_godfather_pinched_fingers():
    d = run(Detectors(V2), [pinched_hand()], make_face())
    assert d["godfather"].raw
    assert not (d["thumbs_up"].raw or d["nerd"].raw or d["middle_finger"].raw or d["gojo"].raw)


def test_godfather_off_in_v1():
    assert not run(Detectors({}), [pinched_hand()], make_face())["godfather"].raw


def test_fist_and_open_hand_are_not_godfather():
    for h in (make_hand(FIST, wrist=(820, 520)), make_hand(OPEN, wrist=(820, 520)), thumbs_up_hand()):
        assert not run(Detectors(V2), [h], make_face())["godfather"].raw


def test_looksmaxxing_finger_on_jaw():
    face = make_face()
    face.pts[150, :2] = (FACE_C[0] - 0.4 * FACE_W, FACE_C[1] + 0.32 * FACE_H)
    d = run(Detectors(V2), [jaw_hand(face)], face)
    assert d["looksmaxxing"].raw
    assert not (d["nerd"].raw or d["scuba_cat"].raw or d["higuruma"].raw)


def test_chin_on_fist_is_not_looksmaxxing():
    face = make_face()
    fist = make_hand(FIST, wrist=(FACE_C[0], FACE_C[1] + 0.75 * FACE_H), size=100)
    assert not run(Detectors(V2), [fist], face)["looksmaxxing"].raw


def test_nerd_beside_face_is_not_looksmaxxing():
    d = run(Detectors(V2), [make_hand(INDEX, wrist=(820, 420))], make_face())
    assert d["nerd"].raw and not d["looksmaxxing"].raw
