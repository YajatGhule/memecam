"""Gesture / pose detectors and the hold → show → linger → cooldown trigger logic.

Finger-state approach follows the common MediaPipe hand-sign projects (e.g. reinesana/SAT0RU):
compare each fingertip to its PIP joint. We measure distance from the wrist instead of raw
y-order so a finger counts as extended whichever way the hand is rotated.

Thresholds were tuned on real clips (tools/eval_clips.py). Lessons from those clips:
- hands over the face make the face tracker drop out and the hand tracker miss hands, so the
  body pose model (which keeps tracking wrists and fingers) backs both up;
- fast-moving hands blur and vanish from the hand tracker, so "talking to a wall" also looks at
  plain frame differencing around the face, and so does the 67 (where the moving pixels are, left
  and right of the face) — on close-ups the bouncing hands also leave the frame at the bottom, and
  the pose wrists flicker between up and down;
- people talking to a wall often just hold both open palms out while talking, barely moving.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import MOTION_H, MOTION_W, Face, Frame, Hand

FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}

DEFAULTS = {
    "finger_extended_ratio": 1.15,   # |wrist→tip| / |wrist→pip| above this = extended
    "cover_dist": 0.75,              # hand within this many face-widths of nose/mouth = covering it
    "cross_dist": 0.25,              # gojo: index & middle tips closer than this (hand-sizes) = crossed
    "peace_gap": 0.35,               # V sign: index & middle tips at least this far apart (hand-sizes)
    "head_tilt_up": 0.10,            # dafoe: drop in nose position ratio vs calibrated baseline
    "motion": 0.07,                  # talking-to-wall: fraction of changed pixels around the face
    "hand_speed": 3.0,               # talking-to-wall: wrist speed in face-widths / second
    "mahoraga_dist": 1.85,           # mahoraga: knuckles within this many face-widths of each other
    "stuck_speed": 2.0,              # scuba: the hand on the face moves slower than this (face-widths/s)
    "swim_speed": 2.5,               # scuba: the other hand moves faster than this (face-widths/s)
    "mouth_move": 0.12,              # talking: lip-gap change rate (face-heights/s) that counts as talking
    "mouth_window": 1.0,             # talking: mouth must have moved within this many seconds
    "bounce_speed": 1.0,             # 67: opposite vertical wrist speed (face-widths / second, RMS)
}

BOUNCE_TAU = 0.4        # 67: smoothing time constant of the wrist-velocity statistics, seconds
BOUNCE_WINDOW = 1.2     # 67: the hands must have swapped direction (up/down) twice within this
BOUNCE_CORR = 0.5       # 67: how anti-correlated the two wrists' vertical motion must be (-1..1)
SIGN_TAU = 0.25         # 67: smoothing of "a fist / finger sign / hand over the face is showing", seconds
# 67 from frame differencing: where the motion is, left and right of the face. Fast bouncing hands
# blur out of the hand tracker and the pose wrists flicker, but the moving pixels follow them.
SWING_WINDOW = 1.0      # seconds of history the statistics look at
SWING_REACH = 2.0       # side regions span 0.45..this many face-widths left / right of the face
SWING_TOP = 0.3         # ...from this many face-widths above the nose/mouth point down to the bottom of the frame
SWING_MIN_MOTION = 0.02 # a side counts as moving when this fraction of its region changed
SWING_CORR = 0.5        # the two sides' motion heights must be this anti-correlated (-1..1)
SWING_AMP = 0.4         # ...and their height difference swing this much (std, face-widths)

MOTION_REACH = 3.0      # talking to a wall: motion counts within this many face-widths left / right of the head
FLYING_KEEP = 0.6       # talking to a wall: once flying, the hands stay "flying" down to this × the thresholds
PRESENT_SIDE = 0.7      # talking to a wall, palms out: each wrist at least this many face-widths left / right of the face
PRESENT_LOW = (0.5, 3.0)  # ...and between these many face-widths below the nose/mouth point (chest height)
PRESENT_GAP = 0.25      # ...seen within this many seconds (rides out hand-tracker dropouts)

# Pose landmark ids
P_NOSE, P_MOUTH, P_EARS, P_SHOULDERS, P_WRISTS, P_INDEX = 0, (9, 10), (7, 8), (11, 12), (15, 16), (19, 20)
POSE_HAND_ENDS = ((15, 17, 19, 21), (16, 18, 20, 22))   # wrist, pinky, index, thumb per arm
EAR_TO_FACE = 1.35                                      # ear-to-ear distance × this ≈ face-mesh width
VIS = 0.5


def finger_states(hand: Hand, ratio: float) -> dict[str, bool]:
    w = hand.pts[0]
    out = {}
    for name, (_, pip, tip) in FINGERS.items():
        out[name] = np.linalg.norm(hand.pts[tip] - w) > ratio * np.linalg.norm(hand.pts[pip] - w)
    return out


@dataclass
class Detection:
    raw: bool
    score: float                       # for the debug overlay
    anchor: tuple[int, int, int, int] | None = None   # bbox the overlay should sit next to


def head_ref(fr: Frame) -> tuple[np.ndarray, float] | None:
    """Point between nose tip and mouth, and the face width — from the face mesh, or from the
    body pose when the face is hidden (e.g. behind your hands)."""
    if fr.face is not None:
        return (fr.face.pts[1, :2] + fr.face.pts[13, :2]) / 2, fr.face.width
    p = fr.pose
    if p is not None and p[P_NOSE, 3] > VIS:
        ref = (p[P_NOSE, :2] + p[list(P_MOUTH), :2].mean(axis=0)) / 2
        return ref, EAR_TO_FACE * float(np.linalg.norm(p[P_EARS[0], :2] - p[P_EARS[1], :2])) + 1e-6
    return None


class Detectors:
    def __init__(self, cfg: dict):
        self.cfg = {**DEFAULTS, **cfg}
        self.nose_baseline: float | None = None
        self.face_frames = 0
        self.prev_wrists: list[np.ndarray] = []
        self.prev_t: float | None = None
        self.hand_speed = 0.0
        self.hand_speeds: list[float] = []   # per tracked hand (same order as frame.hands), smoothed
        self.motion = 0.0
        self.prev_pose: tuple[int, float, np.ndarray] | None = None
        self.arm_speed = [0.0, 0.0]    # pose wrists (user's left, right), face-widths / second
        self.prev_gap: tuple[float, float] | None = None
        self.mouth_vel = 0.0
        self.mouth_at = -1e9           # last time the mouth was moving
        self.covering_tracked: list[Hand] = []
        # 67: previous wrist positions (source, pose id, t, left xy, right xy) and smoothed stats
        self.bounce_prev: tuple[str, int, float, np.ndarray, np.ndarray] | None = None
        self.bounce_anti = 0.0         # EMA of -vy_left * vy_right  (> 0 = moving in opposite directions)
        self.bounce_energy = 0.0       # EMA of (vy_left² + vy_right²) / 2
        self.bounce_sign = 0           # sign of the last clear relative vertical velocity
        self.bounce_flips: list[float] = []
        self.swing: list[tuple[float, float, float]] = []   # 67: (t, left, right motion height / face width)
        self.swing_stats = (0.0, 0.0, 0)                     # corr, amp, crossings (for the debug overlay)
        self.signing_frac = 0.0        # 67: smoothed fraction of recent frames with a fist / finger sign / covered face
        self.signing_t: float | None = None
        self.flying = False            # talking to a wall: hands-flying state (with hysteresis)
        self.present_at = -1e9         # talking to a wall: last time both palms were held out

    def calibrate(self):
        self.nose_baseline = None
        self.face_frames = 0

    # ---- face -------------------------------------------------------------------------

    @staticmethod
    def nose_ratio(face: Face) -> float:
        """Where the nose tip sits between forehead and chin. Tilting the head back moves the
        protruding nose up towards the forehead in the image, so this ratio drops."""
        top, nose, chin = face.pts[10, 1], face.pts[1, 1], face.pts[152, 1]
        return float((nose - top) / (chin - top + 1e-6))

    def head_tilt(self, face: Face) -> float:
        r = self.nose_ratio(face)
        self.face_frames += 1
        if self.nose_baseline is None:
            self.nose_baseline = r
        delta = self.nose_baseline - r
        if self.face_frames <= 15:          # warm-up: settle on the neutral pose, no triggers yet
            self.nose_baseline += 0.25 * (r - self.nose_baseline)
            return 0.0
        # Track the neutral pose quickly; anything clearly off-neutral (a tilt up, or looking down
        # at the keyboard) only nudges the baseline, so poses don't become the new neutral, but a
        # bad baseline still recovers within a few seconds instead of leaving dafoe stuck on.
        rate = 0.02 if abs(delta) < self.cfg["head_tilt_up"] * 0.5 else 0.004 if delta > 0 else 0.001
        self.nose_baseline += rate * (r - self.nose_baseline)
        return delta

    # ---- hands over the face ----------------------------------------------------------

    def covering_hands(self, fr: Frame, can_cover: list[Hand], busy: list[Hand]) -> tuple[list[float], set[int]]:
        """How many hands are held over the nose / mouth. `can_cover` are tracked open-ish hands,
        `busy` are the other tracked hands (finger signs, fists) which never count as covering.
        Returns the speed of each covering hand, and which pose arms (0 = left, 1 = right) cover."""
        self.covering_tracked: list[Hand] = []      # tracked hands found over the face
        ref = head_ref(fr)
        if ref is None:
            return [], set()
        mouth, fw = ref
        reach = self.cfg["cover_dist"] * fw
        self.covering_tracked = [h for h in can_cover if min(np.linalg.norm(h.pts[[0, 5, 9, 13, 17], :2].mean(axis=0) - mouth),
                          np.linalg.norm(h.pts[9, :2] - mouth)) < reach]
        tracked = [self.speed_of(h, fr.hands) for h in self.covering_tracked]
        # The pose model keeps the arms even when the hand tracker loses the hands, but it can't
        # tell an open hand from a fist: an arm only counts if no busy tracked hand sits on it.
        from_pose, arms = [], set()
        p = fr.pose
        if p is not None:
            for arm, ends in enumerate(POSE_HAND_ENDS):
                pts = p[list(ends)]
                pts = pts[pts[:, 3] > 0.3]
                if not len(pts) or np.linalg.norm(pts[:, :2] - mouth, axis=1).min() > reach:
                    continue
                wrist = p[ends[0], :2]
                if any(np.linalg.norm(h.pts[0, :2] - wrist) < 1.5 * h.size for h in busy):
                    continue
                from_pose.append(self.arm_speed[arm])
                arms.add(arm)
        return (tracked if len(tracked) >= len(from_pose) else from_pose), arms

    # ---- motion -----------------------------------------------------------------------

    def motion_level(self, fr: Frame) -> float:
        """Fraction of the frame that changed since the previous frame, counting only the pixels
        around this person (within MOTION_REACH face-widths left / right of their head, so someone
        else waving next to them doesn't count) and outside their face."""
        if fr.motion is None or not fr.width:
            return 0.0
        m = fr.motion
        sx, sy = MOTION_W / fr.width, MOTION_H / fr.height
        ref = head_ref(fr)
        if ref is not None:
            (x, _), fw = ref
            keep = np.zeros_like(m)
            keep[:, max(0, int((x - MOTION_REACH * fw) * sx)):max(0, int((x + MOTION_REACH * fw) * sx) + 1)] = True
            m = m & keep
        if fr.face is not None:
            x0, y0, x1, y1 = fr.face.bbox
            m = m.copy()
            m[max(0, int(y0 * sy) - 2):int(y1 * sy) + 3, max(0, int(x0 * sx) - 2):int(x1 * sx) + 3] = False
        return float(m.mean())

    def update_speed(self, hands: list[Hand], scale: float, now: float) -> None:
        # Match each wrist to the nearest unclaimed wrist from the previous frame. MediaPipe's
        # handedness labels flip and are sometimes duplicated ("Right", "Right"), which made one
        # hand look like it teleported onto the other every frame.
        speeds = []
        wrists = [h.pts[0, :2].copy() for h in hands]
        per_hand = [0.0] * len(hands)
        if self.prev_t is not None and now > self.prev_t and self.prev_wrists:
            pairs = sorted((np.linalg.norm(w - p), i, j) for i, w in enumerate(wrists)
                           for j, p in enumerate(self.prev_wrists))
            used_i, used_j = set(), set()
            for dist_px, i, j in pairs:
                if i not in used_i and j not in used_j:
                    used_i.add(i)
                    used_j.add(j)
                    v = min(dist_px / scale / (now - self.prev_t), 10.0)
                    speeds.append(v)
                    per_hand[i] = self.hand_speeds[j] + 0.35 * (v - self.hand_speeds[j])
        self.prev_wrists, self.prev_t = wrists, now
        self.hand_speeds = per_hand
        inst = max(speeds) if speeds else 0.0
        self.hand_speed += 0.25 * (inst - self.hand_speed)

    def speed_of(self, hand: Hand, hands: list[Hand]) -> float:
        for h, v in zip(hands, self.hand_speeds):
            if h is hand:
                return v
        return 0.0

    def update_arms(self, fr: Frame, fw: float, now: float) -> None:
        """Smoothed wrist speed of each arm from the body pose (stable left/right identity)."""
        p = fr.pose
        if p is None or (self.prev_pose is not None and self.prev_pose[0] == id(p)):
            return                     # no pose, or the same (async) pose result as last frame
        if self.prev_pose is not None and now > self.prev_pose[1]:
            _, t, q = self.prev_pose
            for arm, i in enumerate(P_WRISTS):
                v = 0.0
                if p[i, 3] > VIS and q[i, 3] > VIS:
                    v = min(np.linalg.norm(p[i, :2] - q[i, :2]) / fw / (now - t), 10.0)
                self.arm_speed[arm] += 0.35 * (v - self.arm_speed[arm])
        self.prev_pose = (id(p), now, p)

    def update_mouth(self, face: Face | None, now: float) -> None:
        """How fast the lips open and close (talking), in face-heights per second."""
        if face is None:
            self.prev_gap = None
            return
        gap = float(np.linalg.norm(face.pts[13, :2] - face.pts[14, :2]) /
                    (np.linalg.norm(face.pts[10, :2] - face.pts[152, :2]) + 1e-6))
        if self.prev_gap is not None and now > self.prev_gap[1]:
            self.mouth_vel += 0.2 * (abs(gap - self.prev_gap[0]) / (now - self.prev_gap[1]) - self.mouth_vel)
            if self.mouth_vel > self.cfg["mouth_move"]:
                self.mouth_at = now
        self.prev_gap = (gap, now)

    def bounce_wrists(self, fr: Frame, open_hands: list[Hand]) -> tuple[str, int, np.ndarray, np.ndarray] | None:
        """The user's two wrists for the 67 detector: (source, pose id, left xy, right xy).
        Pose wrists keep a stable left/right identity; otherwise two open tracked hands by x-order."""
        p = fr.pose
        if p is not None and p[list(P_WRISTS), 3].min() > VIS:
            a, b = p[P_WRISTS[0], :2], p[P_WRISTS[1], :2]
            return "pose", id(p), *sorted((a.copy(), b.copy()), key=lambda w: w[0])
        if len(fr.hands) == 2 and len(open_hands) == 2:
            a, b = sorted((h.pts[0, :2].copy() for h in open_hands), key=lambda w: w[0])
            return "hands", 0, a, b
        return None

    def update_bounce(self, wrists, now: float, fw: float) -> None:
        """Vertical wrist velocities → how much the hands move up/down in alternation (67)."""
        if wrists is None:
            self.bounce_prev = None
            self.bounce_anti = self.bounce_energy = 0.0
            self.bounce_sign = 0
            self.bounce_flips.clear()
            return
        src, pid, lw, rw = wrists
        prev = self.bounce_prev
        if prev is not None and src == "pose" and prev[0] == "pose" and prev[1] == pid:
            return                     # same (async) pose result as last frame: nothing new
        if prev is not None and prev[0] == src and now > prev[2]:
            dt = now - prev[2]
            vl = float(np.clip((lw[1] - prev[3][1]) / fw / dt, -10, 10))
            vr = float(np.clip((rw[1] - prev[4][1]) / fw / dt, -10, 10))
            a = 1 - np.exp(-dt / BOUNCE_TAU)
            self.bounce_anti += a * (-vl * vr - self.bounce_anti)
            self.bounce_energy += a * ((vl * vl + vr * vr) / 2 - self.bounce_energy)
            rel = vl - vr
            if abs(rel) > self.cfg["bounce_speed"]:
                sign = 1 if rel > 0 else -1
                if self.bounce_sign and sign != self.bounce_sign:
                    self.bounce_flips.append(now)
                self.bounce_sign = sign
        self.bounce_flips = [t for t in self.bounce_flips if now - t <= BOUNCE_WINDOW]
        self.bounce_prev = (src, pid, now, lw, rw)

    def motion_heights(self, fr: Frame, ref: tuple[np.ndarray, float] | None) -> tuple[float, float] | None:
        """Mean height (face-widths) of the changed pixels left and right of the face, from the nose
        down. None unless both sides are moving."""
        if fr.motion is None or ref is None or not fr.width:
            return None
        (cx, cy), fw = ref
        sx, sy = MOTION_W / fr.width, MOTION_H / fr.height
        y0 = max(0, int((cy - SWING_TOP * fw) * sy))
        out = []
        for lo, hi in ((cx - SWING_REACH * fw, cx - 0.45 * fw), (cx + 0.45 * fw, cx + SWING_REACH * fw)):
            region = fr.motion[y0:, max(0, int(lo * sx)):max(0, min(MOTION_W, int(hi * sx)))]
            if region.size == 0 or region.mean() < SWING_MIN_MOTION:
                return None
            out.append((np.nonzero(region)[0].mean() + y0) / sy / fw)
        return out[0], out[1]

    def update_swing(self, fr: Frame, ref, now: float) -> bool:
        """67 from motion: over the last SWING_WINDOW seconds, the left and right moving regions go up
        and down in opposite directions (anti-correlated heights), with a big enough swing, crossing
        over (one side higher, then the other) at least twice."""
        hs = self.motion_heights(fr, ref)
        if hs is not None:
            self.swing.append((now, *hs))
        self.swing = [x for x in self.swing if now - x[0] <= SWING_WINDOW]
        if len(self.swing) < 8 or now - self.swing[0][0] < 0.6 * SWING_WINDOW:
            self.swing_stats = (0.0, 0.0, 0)
            return False
        a = np.array(self.swing)[:, 1:]
        a = a - a.mean(axis=0)
        anti = -float((a[:, 0] * a[:, 1]).mean())
        corr = anti / (float((a * a).mean()) + 1e-9)
        diff = a[:, 0] - a[:, 1]
        amp = float(diff.std())
        crossings, side = 0, 0
        for v in diff:                         # with hysteresis, so jitter around zero doesn't count
            sg = 1 if v > 0.5 * amp else -1 if v < -0.5 * amp else 0
            if sg and side and sg != side:
                crossings += 1
            side = sg or side
        self.swing_stats = (corr, amp, crossings)
        return corr > SWING_CORR and amp > SWING_AMP and crossings >= 2

    def presenting(self, fr: Frame, open_hands: list[Hand], ref) -> bool:
        """Both open hands held up at chest height, one on each side of the face (fingers pointing
        up-ish, not hands resting on a desk), not flying around (that's the other talking-to-a-wall
        cue): the "explaining with your palms" pose."""
        if len(open_hands) < 2 or ref is None:
            return False
        (x, y), fw = ref
        left = min(open_hands, key=lambda h: h.pts[0, 0])
        right = max(open_hands, key=lambda h: h.pts[0, 0])
        return (left.pts[0, 0] < x - PRESENT_SIDE * fw and right.pts[0, 0] > x + PRESENT_SIDE * fw
                and all(PRESENT_LOW[0] * fw < h.pts[0, 1] - y < PRESENT_LOW[1] * fw    # chest height
                        and h.pts[0, 1] - h.pts[9, 1] > 0.5 * h.size                 # held up, not resting
                        and self.speed_of(h, fr.hands) < self.cfg["hand_speed"] for h in (left, right)))

    # ---- main -------------------------------------------------------------------------

    def run(self, fr: Frame, now: float) -> dict[str, Detection]:
        c = self.cfg
        d: dict[str, Detection] = {}
        face = fr.face
        fbox = face.bbox if face else None
        hands = fr.hands
        states = [finger_states(h, c["finger_extended_ratio"]) for h in hands]
        n_ext = [sum(s.values()) for s in states]

        def find(pred):
            for h, s in zip(hands, states):
                if pred(h, s):
                    return h
            return None

        def two_up(s):
            return s["index"] and s["middle"] and not s["ring"] and not s["pinky"]

        def tip_gap(h):
            return np.linalg.norm(h.pts[8, :2] - h.pts[12, :2]) / h.size

        # ---- one-hand finger signs ----
        mid = find(lambda h, s: s["middle"] and not (s["index"] or s["ring"] or s["pinky"]))
        d["middle_finger"] = Detection(mid is not None, 1.0 if mid else 0.0, mid.bbox if mid else None)

        # Gojo "nah, I'd win": index and middle crossed (his Infinite Void hand sign).
        def crossed(h):
            # Tip order flipped relative to knuckle order = the fingers cross over.
            flip = np.dot(h.pts[8, :2] - h.pts[12, :2], h.pts[5, :2] - h.pts[9, :2]) < 0
            return tip_gap(h) < c["cross_dist"] or flip

        # The pinky is often misread as straight with the fingers crossed, so it isn't checked.
        gojo = find(lambda h, s: s["index"] and s["middle"] and not s["ring"] and crossed(h))
        d["gojo"] = Detection(gojo is not None, 1.0 if gojo else 0.0, gojo.bbox if gojo else fbox)

        # A V sign (two fingers clearly spread) has no meme, but it is a finger sign: it mustn't read as
        # the nerd's finger, or as a hand over the face.
        vsign = find(lambda h, s: h is not gojo and tip_gap(h) > c["peace_gap"] and (
            two_up(s) or h.gesture == "Victory"))

        # Thumbs up 👍 (the dad meme): fingers curled, thumb sticking up. MediaPipe's Thumb_Up label is
        # reliable on real thumbs-ups; the geometric fallback (for when it says nothing) is skipped
        # when it calls the hand a fist (a fist with the thumb resting on top looks the same).
        def thumb_up(h, s):
            if h.gesture == "Thumb_Up":
                return True
            if any(s.values()) or h.gesture == "Closed_Fist":
                return False
            t = h.pts[4, :2] - h.pts[2, :2]                    # thumb MCP -> tip
            return -t[1] > 0.7 * np.linalg.norm(t) and np.linalg.norm(t) > 0.45 * h.size
        thumbs = find(thumb_up)
        d["thumbs_up"] = Detection(thumbs is not None, 1.0 if thumbs else 0.0, thumbs.bbox if thumbs else None)

        # Nerd ☝️🤓: index finger pointing up beside the face ("umm, actually...").
        def pointing_up(h):
            v = h.pts[8, :2] - h.pts[5, :2]                    # index MCP -> tip
            return -v[1] > 0.5 * np.linalg.norm(v)             # within 60 deg of straight up

        def on_face(h):                                        # e.g. a finger on your nose
            if face is None:
                return False
            x0, y0, x1, y1 = fbox
            return x0 < h.pts[8, 0] < x1 and y0 < h.pts[8, 1] < y1

        point = find(lambda h, s: h is not vsign and h is not gojo and h is not thumbs and pointing_up(h) and not on_face(h)
                     and (h.gesture == "Pointing_Up" or (s["index"] and not s["middle"] and not s["ring"]
                                                         and not s["pinky"])))
        d["nerd"] = Detection(point is not None and mid is None, 1.0 if point else 0.0,
                              point.bbox if point else None)

        signing = [h for h in (mid, gojo, vsign, point, thumbs) if h is not None]
        fists = [h for h, n in zip(hands, n_ext) if (n == 0 or h.gesture == "Closed_Fist") and h is not thumbs]
        can_cover = [h for h, n in zip(hands, n_ext) if n >= 3 and all(h is not x for x in signing + fists)]
        busy = [h for h in hands if all(h is not x for x in can_cover)]

        # ---- motion ----
        scale = face.width if face is not None else (hands[0].size * 2.5 if hands else 1.0)
        self.update_speed(hands, scale, now)
        ref = head_ref(fr)
        self.update_arms(fr, ref[1] if ref else scale, now)
        self.update_mouth(face, now)
        self.motion += 0.25 * (self.motion_level(fr) - self.motion)

        # ---- hands over the face: scuba cat / higuruma ----
        # One hand over nose & mouth = scuba cat (holding your breath to dive); both hands over the
        # face = Higuruma giving up. A single hand never becomes higuruma, however still it is.
        # Scuba needs the hand on the face to stay put while the other hand swims around.
        cover_speeds, cover_arms = self.covering_hands(fr, can_cover, busy)
        covering = len(cover_speeds)
        higuruma = covering >= 2
        # any tracked hand that isn't the one on the face can be the swimming one (open or not)
        others = [v for h, v in zip(hands, self.hand_speeds) if all(h is not x for x in self.covering_tracked)]
        others += [self.arm_speed[a] for a in (0, 1) if cover_arms and a not in cover_arms]
        swimming = max(others, default=0.0) > c["swim_speed"] or self.motion > 0.02
        scuba = covering == 1 and not higuruma and cover_speeds[0] < c["stuck_speed"] and swimming
        d["higuruma"] = Detection(higuruma, covering / 2, fbox)
        d["scuba_cat"] = Detection(scuba, covering / 2, fbox)

        # ---- dafoe: head tilted back ----
        if face is not None:
            tilt = self.head_tilt(face) if covering == 0 else 0.0
            d["dafoe"] = Detection(tilt > c["head_tilt_up"], tilt / c["head_tilt_up"] * 0.5, fbox)

        # ---- two-arm poses ----
        p = fr.pose
        arms = p is not None and p[list(P_WRISTS), 3].min() > VIS and p[list(P_SHOULDERS), 3].min() > VIS
        open_hands = [h for h, n in zip(hands, n_ext) if n >= 3 or h.gesture == "Open_Palm"]

        # Absolute cinema: both arms up, open palms beside the head.
        cinema = False
        if arms:
            ew = float(np.linalg.norm(p[P_EARS[0], :2] - p[P_EARS[1], :2])) + 1e-6
            (lw, rw), (ls, rs) = p[list(P_WRISTS), :2], p[list(P_SHOULDERS), :2]
            outer_l, outer_r = (lw, ls) if ls[0] > rs[0] else (rw, rs), (rw, rs) if ls[0] > rs[0] else (lw, ls)
            cinema = (lw[1] < ls[1] - 0.6 * ew and rw[1] < rs[1] - 0.6 * ew
                      and outer_l[0][0] > outer_l[1][0] and outer_r[0][0] < outer_r[1][0] and not fists)
        elif len(open_hands) == 2 and face is not None:
            fx0, _, fx1, _ = fbox
            left, right = sorted(open_hands, key=lambda h: h.pts[0, 0])
            cinema = (all(h.pts[9, 1] < face.pts[152, 1] for h in open_hands)
                      and left.pts[9, 0] < fx0 and right.pts[9, 0] > fx1)
        d["absolute_cinema"] = Detection(cinema, 1.0 if cinema else 0.0, fbox)

        # Mahoraga: Megumi's sign, two fists knuckle to knuckle in front of you.
        mahoraga, score, box = False, 0.0, None
        if fists and not open_hands and covering == 0 and not cinema:
            if arms and p[list(P_INDEX), 3].min() > 0.3:
                ref = head_ref(fr)
                fw = ref[1] if ref else 1.0
                knuckles = np.linalg.norm(p[P_INDEX[0], :2] - p[P_INDEX[1], :2]) / fw
                below_nose = p[list(P_WRISTS), 1].min() > p[P_NOSE, 1]
                mahoraga = knuckles < c["mahoraga_dist"] and below_nose
                score = max(0.0, 1 - knuckles / 4)
            elif len(fists) == 2:
                h1, h2 = fists
                dist = np.linalg.norm(h1.pts[9, :2] - h2.pts[9, :2]) / ((h1.size + h2.size) / 2)
                # not resting your chin on your fists
                chest = face is None or min(h1.pts[9, 1], h2.pts[9, 1]) > face.pts[152, 1] + 0.25 * face.width
                mahoraga, score = dist < 1.8 and chest, max(0.0, 1 - dist / 4)        # hand-sizes
            xs = [h.bbox for h in fists]
            box = (min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
        d["mahoraga"] = Detection(mahoraga, score, box)

        # ---- 67: open hands at chest height bouncing up and down in alternation (weighing scales) ----
        ref = head_ref(fr)
        fw = ref[1] if ref else scale
        wrists = self.bounce_wrists(fr, open_hands)
        self.update_bounce(wrists, now, fw)
        amp = float(np.sqrt(max(self.bounce_anti, 0.0)))
        corr = self.bounce_anti / (self.bounce_energy + 1e-6)
        swinging = self.update_swing(fr, ref, now)
        # Open hands, not fists or finger signs (if the hands are tracked at all), and not over the
        # face. Fast-moving hands blur, single frames get misread as a fist or a pointing finger, and
        # a hand bouncing up to chin height briefly passes the mouth, so only something that shows up
        # in most recent frames blocks the 67.
        dt = now - self.signing_t if self.signing_t is not None and now > self.signing_t else 0.0
        self.signing_t = now
        self.signing_frac += (1 - np.exp(-dt / SIGN_TAU)) * (float(bool(fists or signing or covering)) - self.signing_frac)
        hands_ok = self.signing_frac < 0.5
        bouncing, box = False, None
        if wrists is not None:
            _, _, lw, rw = wrists
            low = ref is None or min(lw[1], rw[1]) > ref[0][1] + 0.5 * fw       # below the chin
            apart = rw[0] - lw[0] > 0.8 * fw                                    # not clapping
            bouncing = (low and apart and corr > BOUNCE_CORR and amp > c["bounce_speed"]
                        and len(self.bounce_flips) >= 2)
            r = int(0.3 * fw)
            box = (int(lw[0]) - r, int(min(lw[1], rw[1])) - r, int(rw[0]) + r, int(max(lw[1], rw[1])) + r)
        elif ref is not None:
            (x, y), r = ref[0], ref[1]
            box = (int(x - 2 * r), int(y), int(x + 2 * r), int(y + 2 * r))
        sixty_seven = ((bouncing or swinging) and hands_ok and not cinema and not mahoraga)
        score = max(max(corr, 0.0) * min(amp / c["bounce_speed"], 2.0) / 2,
                    max(self.swing_stats[0], 0.0) * min(self.swing_stats[1] / SWING_AMP, 2.0) / 2)
        d["sixty_seven"] = Detection(sixty_seven, score, box)

        # ---- talking to a wall: hands flying around, or both open palms held out, while talking ----
        posed = covering > 0 or any(d[k].raw for k in ("middle_finger", "thumbs_up", "nerd", "gojo",
                                                        "absolute_cinema", "mahoraga",
                                                        "sixty_seven"))
        # With hysteresis: once the hands are flying, a dip in the motion (hands slowing down at the
        # end of a swing) doesn't end it straight away.
        floor = FLYING_KEEP if self.flying else 1.0
        self.flying = (self.motion > floor * c["motion"]
                       or (self.hand_speed > floor * c["hand_speed"] and self.motion > 0.03))
        hands_flying = self.flying
        # Explaining with both hands: open palms held out on either side of you at chest height. People
        # talking like that often barely move their hands, so this doesn't need motion.
        if self.presenting(fr, open_hands, ref):
            self.present_at = now
        presenting = now - self.present_at < PRESENT_GAP
        # ...while actually talking: the mouth has to be moving too.
        talking = (hands_flying or presenting) and now - self.mouth_at < c["mouth_window"]
        d["talking_to_wall"] = Detection(talking and not posed,
                                         max(self.motion / c["motion"], self.hand_speed / c["hand_speed"],
                                             2.0 * (presenting and talking)) * 0.5,
                                         fbox)
        return d


class Trigger:
    """idle --(raw held for `hold`)--> active --(raw gone, min_show & linger elapsed)--> cooldown."""

    def __init__(self, hold=0.3, min_show=2.0, linger=0.6, cooldown=0.8):
        self.hold, self.min_show, self.linger, self.cooldown = hold, min_show, linger, cooldown
        self.raw_since: float | None = None
        self.started: float | None = None
        self.last_raw = 0.0
        self.cool_until = 0.0

    @property
    def active(self) -> bool:
        return self.started is not None

    def update(self, raw: bool, now: float) -> bool:
        if raw:
            self.last_raw = now
            if self.raw_since is None:
                self.raw_since = now
        else:
            self.raw_since = None

        if self.started is None:
            if raw and now >= self.cool_until and now - self.raw_since >= self.hold:
                self.started = now
        elif not raw and now - self.started >= self.min_show and now - self.last_raw >= self.linger:
            self.started = None
            self.cool_until = now + self.cooldown
        return self.started is not None
