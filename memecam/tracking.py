"""MediaPipe Tasks wrappers: hands (+ canned gestures), face (+ blendshapes), pose.

All three run in VIDEO mode on the same mirrored frame, concurrently (MediaPipe releases the GIL).
Pose is the slowest model, so by default it runs decoupled: a frame waits for hands + face and
carries the newest finished pose result (usually one frame old) instead of blocking on it.
`pose_async=False` waits for pose every frame (deterministic, for offline clip evaluation).
Up to `max_people` people are tracked; detections are grouped into one Frame per person (see
group_people) with ids that stay stable across frames (Identities).
Landmarks are converted to pixel-space numpy arrays so the rest of the app never touches MediaPipe types.
"""
from __future__ import annotations

import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "gesture_recognizer.task": f"{_MODEL_BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    "face_landmarker.task": f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def model_path(name: str) -> str:
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[memecam] downloading {name} ...")
        urllib.request.urlretrieve(MODELS[name], path)
    return str(path)


@dataclass
class Hand:
    pts: np.ndarray            # (21, 3) pixel x, y and relative z
    label: str                 # "Left" / "Right" (user's hand, frame is mirrored)
    gesture: str               # MediaPipe canned gesture, e.g. "Victory", "Open_Palm"
    gesture_score: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.pts[:, :2].min(0)
        x1, y1 = self.pts[:, :2].max(0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def size(self) -> float:
        """Wrist → middle-finger MCP distance: a scale that doesn't change with finger pose."""
        return float(np.linalg.norm(self.pts[9, :2] - self.pts[0, :2])) + 1e-6


@dataclass
class Face:
    pts: np.ndarray                      # (478, 3) pixel coords
    blend: dict[str, float]              # blendshape name → score

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.pts[:, :2].min(0)
        x1, y1 = self.pts[:, :2].max(0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.pts[234, :2] - self.pts[454, :2])) + 1e-6


@dataclass
class Frame:
    """One person's tracking result. The frame returned by Tracker.process() mirrors its primary
    (largest) person in the top-level fields, and lists everyone in `people`."""
    hands: list[Hand] = field(default_factory=list)
    face: Face | None = None
    pose: np.ndarray | None = None       # (33, 4) pixel x, y, z, visibility
    motion: np.ndarray | None = None     # (MOTION_H, MOTION_W) bool: pixels that changed since last frame (shared)
    width: int = 0                       # source frame size, to map `motion` back to pixels
    height: int = 0
    pid: int = 0                         # persistent person id (1, 2, ...); 0 = not assigned
    people: list[Frame] = field(default_factory=list)   # every person, primary first (top-level frame only)
    renamed: dict[int, int] = field(default_factory=dict)  # ids renumbered this frame, old -> new (top-level only)


# ---- people: grouping faces / poses / hands, and stable ids ----------------------------------

EAR_TO_FACE = 1.35          # pose ear-to-ear distance × this ≈ face-mesh width
POSE_FACE_MATCH = 1.5       # a pose's nose within this many face-widths of a face's nose = same person
HAND_WRIST_MATCH = 1.2      # hand wrist within this many face-widths of a pose wrist = that person's hand
HAND_FACE_MATCH = 3.5       # ...else within this many face-widths of a face = that person's hand
ID_MAX_JUMP = 1.5           # identity: a head moves at most this many face-widths between frames
ID_FORGET = 0.4             # seconds an unseen person keeps their id (bridges tracking dropouts only)


def pose_face_width(pose: np.ndarray) -> float:
    """Face-width estimate from the body pose (ears, or shoulders when seen in profile)."""
    ears = float(np.linalg.norm(pose[7, :2] - pose[8, :2]))
    shoulders = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
    return max(EAR_TO_FACE * ears, 0.45 * shoulders) + 1e-6


def head_of(person: Frame) -> tuple[np.ndarray, float] | None:
    """(nose xy, face width) of a person, from the face mesh or else the body pose."""
    if person.face is not None:
        return person.face.pts[1, :2], person.face.width
    if person.pose is not None:
        return person.pose[0, :2], pose_face_width(person.pose)
    return None


def head_box(person: Frame) -> tuple[int, int, int, int] | None:
    """Bounding box of a person's head: the face mesh, or nose ± ear distance from the pose."""
    if person.face is not None:
        return person.face.bbox
    if person.pose is not None:
        (x, y), r = person.pose[0, :2], 0.5 * pose_face_width(person.pose)
        return int(x - r), int(y - 1.25 * r), int(x + r), int(y + 1.25 * r)
    return None


def hands_box(person: Frame) -> tuple[int, int, int, int] | None:
    """Box around a person's hands (for a hands-only person, whose hands usually cover the face)."""
    if not person.hands:
        return None
    b = np.array([h.bbox for h in person.hands])
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def group_people(faces: list[Face], poses: list[np.ndarray], hands: list[Hand], width: int,
                 max_people: int = 3, min_face_width: float = 0.08) -> list[Frame]:
    """Group raw detections into people, largest first (no ids yet).

    Faces narrower than `min_face_width` × frame width are background (a classroom behind you) and
    ignored, as are small poses. Each face takes the pose whose nose is nearest (poses can lag a frame
    behind, hence the generous match radius); leftover poses are people whose face is hidden. A hand goes
    to the person with the nearest pose wrist, else to the nearest face; at most two hands per person.
    With a single person every hand is theirs (as before multi-person support), and hands with nobody
    around form a person of their own."""
    min_w = min_face_width * width
    faces = sorted((f for f in faces if f.width >= min_w), key=lambda f: -f.width)[:max_people]
    poses = [p for p in poses if pose_face_width(p) >= min_w]
    people = [Frame(face=f) for f in faces]
    pairs = sorted((float(np.linalg.norm(p[0, :2] - f.pts[1, :2])) / f.width, i, j)
                   for i, f in enumerate(faces) for j, p in enumerate(poses))
    used_f, used_p = set(), set()
    for d, i, j in pairs:
        if d < POSE_FACE_MATCH and i not in used_f and j not in used_p:
            used_f.add(i)
            used_p.add(j)
            people[i].pose = poses[j]
    people += [Frame(pose=p) for j, p in enumerate(poses) if j not in used_p]
    people.sort(key=lambda q: -head_of(q)[1])
    people = people[:max_people]

    if not people:
        if hands:
            people = [Frame(hands=sorted(hands, key=lambda h: -h.size)[:2])]
        return people
    if len(people) == 1:                        # one person: every hand is theirs, no matching needed
        if len(hands) > 2:
            nose = head_of(people[0])[0]
            near = sorted(hands, key=lambda h: float(np.linalg.norm(h.pts[0, :2] - nose)))[:2]
            hands = [h for h in hands if any(h is x for x in near)]
        people[0].hands = list(hands)
        return people

    heads = [head_of(q) for q in people]
    cands = []                                  # (cost, hand index, person index)
    for i, h in enumerate(hands):
        w = h.pts[0, :2]
        for k, (q, (nose, fw)) in enumerate(zip(people, heads)):
            if q.pose is not None:
                d = min(float(np.linalg.norm(w - q.pose[j, :2])) for j in (15, 16)) / fw
                if d < HAND_WRIST_MATCH or len(people) == 1:
                    cands.append((d, i, k))
                    continue
            d = float(np.linalg.norm(w - nose)) / fw
            if d < HAND_FACE_MATCH or len(people) == 1:
                cands.append((100 + d, i, k))
    taken = set()
    for _, i, k in sorted(cands):
        if i not in taken and len(people[k].hands) < 2:
            taken.add(i)
            people[k].hands.append(hands[i])
    for q in people:                            # keep MediaPipe's order within a person
        q.hands.sort(key=lambda h: next(i for i, x in enumerate(hands) if x is h))
    return people


class Identities:
    """Persistent person ids: nearest-neighbour matching of head positions between frames.

    An unseen person keeps their id for `forget` seconds (a short tracking dropout), then it is free
    again. New people take the lowest free id, and when only one person is left they become id 1
    (`renamed` lists the renumbering of the latest assign(), old -> new)."""

    def __init__(self, max_jump: float = ID_MAX_JUMP, forget: float = ID_FORGET):
        self.max_jump, self.forget = max_jump, forget
        self.tracks: dict[int, tuple[np.ndarray | None, float, float]] = {}   # pid -> (nose, face width, last seen)
        self.renamed: dict[int, int] = {}

    def assign(self, people: list[Frame], now: float) -> None:
        self.tracks = {k: v for k, v in self.tracks.items() if now - v[2] <= self.forget}
        self.renamed = {}
        if len(self.tracks) == 1 and 1 not in self.tracks:     # alone again: back to PERSON_01
            (old, v), = self.tracks.items()
            self.tracks = {1: v}
            self.renamed = {old: 1}
        heads = [head_of(q) for q in people]
        pairs = []
        for i, hd in enumerate(heads):
            if hd is None:
                continue
            for pid, (c, fw, _) in self.tracks.items():
                if c is not None:
                    d = float(np.linalg.norm(hd[0] - c)) / max(hd[1], fw)
                    if d < self.max_jump:
                        pairs.append((d, i, pid))
        done: dict[int, int] = {}
        for _, i, pid in sorted(pairs):
            if i not in done and pid not in done.values():
                done[i] = pid
        for i, q in enumerate(people):
            if i not in done:
                free = [pid for pid in self.tracks if pid not in done.values()
                        and (heads[i] is None or self.tracks[pid][0] is None)]
                if free:        # hands only: whoever was seen most recently; a face for a hands-only track
                    done[i] = max(free, key=lambda pid: self.tracks[pid][2])
                else:
                    done[i] = next(n for n in range(1, len(self.tracks) + len(people) + 2)
                                   if n not in self.tracks and n not in done.values())
            q.pid = done[i]
            c, fw = heads[i] if heads[i] is not None else (self.tracks.get(q.pid, (None, 1.0, 0))[:2])
            self.tracks[q.pid] = (None if c is None else np.array(c, float), fw, now)


MOTION_W, MOTION_H = 160, 90


def motion_gray(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)


class Tracker:
    def __init__(self, use_pose: bool = True, pose_async: bool = True, max_people: int = 3,
                 min_face_width: float = 0.08, forget: float = ID_FORGET):
        self.pose_async = pose_async
        self.max_people, self.min_face_width = max(1, int(max_people)), float(min_face_width)
        self.ids = Identities(forget=forget)
        n = self.max_people
        running = vision.RunningMode.VIDEO
        self.hands = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path("gesture_recognizer.task")),
            running_mode=running, num_hands=2 * n,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("face_landmarker.task")),
            running_mode=running, num_faces=n, output_face_blendshapes=True))
        self.pose = None
        if use_pose:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task")),
                running_mode=running, num_poses=n))
        self.pool = ThreadPoolExecutor(3, thread_name_prefix="tracker")
        self.pose_job: tuple[Future, int, int] | None = None
        self.last_poses: list[np.ndarray] = []
        self.prev_gray: np.ndarray | None = None

    def process(self, rgb: np.ndarray, ts_ms: int) -> Frame:
        h, w = rgb.shape[:2]
        scale = np.array([w, h, w])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hands: list[Hand] = []
        faces: list[Face] = []

        hands_job = self.pool.submit(self.hands.recognize_for_video, image, ts_ms)
        face_job = self.pool.submit(self.face.detect_for_video, image, ts_ms)
        if self.pose is not None and self.pose_job is None:
            self.pose_job = (self.pool.submit(self.pose.detect_for_video, image, ts_ms), w, h)

        res = hands_job.result()
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms]) * scale
            label = res.handedness[i][0].category_name if res.handedness else "?"
            g = res.gestures[i][0] if res.gestures and res.gestures[i] else None
            hands.append(Hand(pts, label, g.category_name if g else "None", g.score if g else 0.0))

        res = face_job.result()
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms]) * scale
            blend = {c.category_name: c.score for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            faces.append(Face(pts, blend))

        if self.pose_job is not None and (self.pose_job[0].done() or not self.pose_async):
            job, pw, ph = self.pose_job
            self.pose_job = None
            res = job.result()
            self.last_poses = [np.array([[p.x * pw, p.y * ph, p.z * pw, p.visibility] for p in lm])
                               for lm in res.pose_landmarks]

        people = group_people(faces, self.last_poses, hands, w, self.max_people, self.min_face_width)
        self.ids.assign(people, ts_ms / 1000)
        main = people[0] if people else Frame()
        out = Frame(hands=main.hands, face=main.face, pose=main.pose, width=w, height=h, pid=main.pid,
                    people=people, renamed=self.ids.renamed)

        # Frame differencing: fast-moving hands blur and drop out of the hand tracker, but they
        # still light up the difference image (used for "talking to a wall").
        gray = motion_gray(rgb)
        if self.prev_gray is not None:
            out.motion = cv2.absdiff(gray, self.prev_gray) > 20
        self.prev_gray = gray
        for q in people:
            q.motion, q.width, q.height = out.motion, w, h
        return out

    def close(self):
        self.pool.shutdown(wait=True)
        for t in (self.hands, self.face, self.pose):
            if t is not None:
                t.close()
