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

import os
import sys
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
WRIST_STRAY = 2.5          # a pose wrist this far (face-widths, sideways) from its own head is suspect
GHOST_DIST = 2.0           # a face-less pose this close (face-widths) to another person's head is a duplicate
ID_MAX_JUMP = 1.5           # identity: a head moves at most this many face-widths between frames
ID_CONFIRM = 2              # identity: frames a person must be seen before a dropout of theirs claims hands
SCAN_EVERY = 10            # with one person in view, look for more people every this many frames
CROWD_LINGER = 1.5         # seconds with ≤1 person before going back to the faster 1-person models
ID_FORGET = 1.0             # seconds an unseen person keeps their id (bridges tracking dropouts only)


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
                 max_people: int = 3, min_face_width: float = 0.08,
                 recent: list[Seen] | None = None) -> list[Frame]:
    """Group raw detections into people, largest first (no ids yet).

    Faces narrower than `min_face_width` × frame width are background (a classroom behind you) and
    ignored, as are small poses. Each face takes the pose whose nose is nearest (poses can lag a frame
    behind, hence the generous match radius); leftover poses are people whose face is hidden. A hand goes
    to the person with the nearest pose wrist, else to the nearest face; at most two hands per person.
    `recent` (Identities.recent()) lists where everyone seen in the last moment was: someone among them
    that isn't found now (a tracking dropout) still claims the hands near their last head / wrists, and
    those hands are dropped rather than given to whoever is left. With a single person and nobody else
    seen recently, every hand is theirs (as before multi-person support), and hands with nobody around
    form a person of their own."""
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
    # Leftover poses are people whose face is hidden, unless they sit on someone already found: the pose
    # model sometimes returns a second, shifted skeleton for the same body (a ghost PERSON_02).
    for j in sorted((j for j in range(len(poses)) if j not in used_p), key=lambda j: -pose_face_width(poses[j])):
        p, fw = poses[j], pose_face_width(poses[j])
        if all(float(np.linalg.norm(p[0, :2] - hd[0])) > GHOST_DIST * max(fw, hd[1])
               for hd in (head_of(q) for q in people) if hd is not None):
            people.append(Frame(pose=p))
    people.sort(key=lambda q: -head_of(q)[1])
    people = people[:max_people]

    if not people:
        if hands:
            people = [Frame(hands=sorted(hands, key=lambda h: -h.size)[:2])]
        return people
    heads = [head_of(q) for q in people]
    absent = _absent(heads, recent or [])
    if len(people) == 1 and not absent:         # one person: every hand is theirs, no matching needed
        if len(hands) > 2:
            nose = heads[0][0]
            near = sorted(hands, key=lambda h: float(np.linalg.norm(h.pts[0, :2] - nose)))[:2]
            hands = [h for h in hands if any(h is x for x in near)]
        people[0].hands = list(hands)
        return people

    # Everyone who can own a hand: the people found now, then the recently seen but missing ones.
    slots = [(nose, fw, None if q.pose is None else q.pose[[15, 16], :2]) for q, (nose, fw) in zip(people, heads)]
    slots += [(s.nose, s.fw, s.wrists) for s in absent]
    alone = len(people) == 1                    # the one person found takes any hand nobody else is near
    cands = []                                  # (cost, hand index, slot index)
    for i, h in enumerate(hands):
        w = h.pts[0, :2]
        near = [float(np.linalg.norm(w - nose)) / fw for nose, fw, _ in slots]
        nearest = int(np.argmin(near))
        for k, (nose, fw, wrists) in enumerate(slots):
            if wrists is not None:
                d = min(float(np.linalg.norm(w - x)) for x in wrists) / fw
                # A pose wrist on a hand far off to the side of its own head, right next to someone
                # else's face, is a misplaced wrist (the pose model mixing up two bodies): use the faces.
                stray = abs(w[0] - nose[0]) / fw > WRIST_STRAY and near[nearest] < 0.75 * near[k]
                if d < HAND_WRIST_MATCH and not stray:
                    cands.append((d, i, k))
                    continue
            # Face fallback: only the nearest head may take the hand (a third hand of a full person is
            # a duplicate detection, not someone else's hand).
            if k == nearest and (near[k] < HAND_FACE_MATCH or (alone and k == 0)):
                cands.append((100 + near[k], i, k))
            elif alone and k == 0 and nearest >= len(people) and near[nearest] >= HAND_FACE_MATCH:
                cands.append((100 + near[k], i, k))     # nearest is a missing person, but far from them too
    owner: dict[int, int] = {}
    count = [0] * len(slots)
    for _, i, k in sorted(cands):
        if i not in owner and count[k] < 2:
            owner[i] = k
            count[k] += 1
    for i, h in enumerate(hands):               # keep MediaPipe's order within a person
        if owner.get(i, len(people)) < len(people):   # hands of a missing person are dropped
            people[owner[i]].hands.append(h)
    return people


def _absent(heads: list[tuple[np.ndarray, float] | None], recent: list[Seen]) -> list[Seen]:
    """The recently seen people (with a head) that none of the current heads continues."""
    pairs = sorted((float(np.linalg.norm(hd[0] - s.nose)) / max(hd[1], s.fw), i, j)
                   for i, hd in enumerate(heads) if hd is not None
                   for j, s in enumerate(recent) if s.nose is not None)
    used_i, used_j = set(), set()
    for d, i, j in pairs:
        if d < ID_MAX_JUMP and i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
    return [s for j, s in enumerate(recent) if s.nose is not None and j not in used_j]


@dataclass
class Seen:
    """Where a tracked person was last seen."""
    nose: np.ndarray | None        # head position (None: hands only)
    fw: float                      # face width
    t: float                       # when
    wrists: np.ndarray | None = None   # (2, 2) last pose wrists, if seen recently
    wrists_t: float = 0.0
    hits: int = 1                  # frames seen in a row (with gaps under `forget`)


class Identities:
    """Persistent person ids: nearest-neighbour matching of head positions between frames.

    An unseen person keeps their id for `forget` seconds (a short tracking dropout), then it is free
    again. New people take the lowest free id, and when only one person is left they become id 1
    (`renamed` lists the renumbering of the latest assign(), old -> new). Each track remembers the last
    head position and pose wrists, so recent() can tell group_people where a missing person was."""

    def __init__(self, max_jump: float = ID_MAX_JUMP, forget: float = ID_FORGET):
        self.max_jump, self.forget = max_jump, forget
        self.tracks: dict[int, Seen] = {}       # pid -> last seen
        self.renamed: dict[int, int] = {}

    def recent(self, now: float) -> list[Seen]:
        """Everyone seen within the last `forget` seconds (wrists only if those are that recent too).
        A face seen for fewer than `ID_CONFIRM` frames (often a false detection) doesn't count."""
        out = []
        for s in self.tracks.values():
            if now - s.t <= self.forget and s.hits >= ID_CONFIRM:
                fresh = s.wrists is not None and now - s.wrists_t <= self.forget
                out.append(Seen(s.nose, s.fw, s.t, s.wrists if fresh else None, s.wrists_t, s.hits))
        return out

    def assign(self, people: list[Frame], now: float) -> None:
        self.tracks = {k: v for k, v in self.tracks.items() if now - v.t <= self.forget}
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
            for pid, s in self.tracks.items():
                if s.nose is not None:
                    d = float(np.linalg.norm(hd[0] - s.nose)) / max(hd[1], s.fw)
                    if d < self.max_jump:
                        pairs.append((d, i, pid))
        done: dict[int, int] = {}
        for _, i, pid in sorted(pairs):
            if i not in done and pid not in done.values():
                done[i] = pid
        for i, q in enumerate(people):
            if i not in done:
                free = [pid for pid in self.tracks if pid not in done.values()
                        and (heads[i] is None or self.tracks[pid].nose is None)]
                if free:        # hands only: whoever was seen most recently; a face for a hands-only track
                    done[i] = max(free, key=lambda pid: self.tracks[pid].t)
                else:
                    done[i] = next(n for n in range(1, len(self.tracks) + len(people) + 2)
                                   if n not in self.tracks and n not in done.values())
            q.pid = done[i]
            prev = self.tracks.get(q.pid)
            if heads[i] is not None:
                nose, fw = np.array(heads[i][0], float), heads[i][1]
            else:
                nose, fw = (prev.nose, prev.fw) if prev is not None else (None, 1.0)
            if q.pose is not None:
                wrists, wrists_t = np.array(q.pose[[15, 16], :2], float), now
            else:
                wrists, wrists_t = (prev.wrists, prev.wrists_t) if prev is not None else (None, 0.0)
            self.tracks[q.pid] = Seen(nose, fw, now, wrists, wrists_t, prev.hits + 1 if prev is not None else 1)


MOTION_W, MOTION_H = 160, 90


def motion_gray(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)


GPU_VENDORS = {"10de": "nvidia", "1002": "amd", "8086": "intel"}


def windows_gpus() -> list[tuple[str, str]]:
    """(vendor, name) of the display adapters in the Windows registry, e.g. ("amd", "AMD Radeon(TM) Graphics").
    Reads the same keys Device Manager shows, so integrated Radeon / Vega graphics are listed too."""
    try:
        import winreg
    except ImportError:
        return []
    out = []
    cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls)
    except OSError:
        return []
    with root:
        for i in range(64):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            try:
                with winreg.OpenKey(root, sub) as k:
                    name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                    dev = str(winreg.QueryValueEx(k, "MatchingDeviceId")[0]).lower()
            except OSError:
                continue
            ven = dev.split("ven_")[1][:4] if "ven_" in dev else ""
            vendor = GPU_VENDORS.get(ven, "")
            low = name.lower()
            if not vendor:
                vendor = next((v for k, v in (("nvidia", "nvidia"), ("radeon", "amd"), ("amd", "amd"),
                                             ("intel", "intel")) if k in low), "")
            if vendor and "basic" not in low:      # skip "Microsoft Basic Display Adapter"
                out.append((vendor, name))
    return out


def linux_gpus() -> list[str]:
    """Vendors of the DRM cards ('nvidia' / 'amd' / 'intel'), integrated GPUs included."""
    import glob
    out = []
    for vendor_file in sorted(glob.glob("/sys/class/drm/card[0-9]*/device/vendor")):
        if "-" in vendor_file.split("/")[4]:       # card0-eDP-1 etc. are connectors, not GPUs
            continue
        try:
            vendor = open(vendor_file).read().strip().lower().removeprefix("0x")
        except OSError:
            continue
        if vendor in GPU_VENDORS:
            out.append(GPU_VENDORS[vendor])
    return out


def discrete_gpu() -> str | None:
    """'nvidia' / 'amd' if an NVIDIA or AMD GPU is present (dedicated, or AMD's integrated Radeon graphics,
    which are fast enough to be worth it). Linux: sysfs; Windows: the registry; macOS: nvidia-smi only."""
    import shutil
    vendors = linux_gpus() or [v for v, _ in windows_gpus()]
    for kind in ("nvidia", "amd"):
        if kind in vendors:
            return kind
    return "nvidia" if shutil.which("nvidia-smi") else None


def use_dedicated_gpu(kind: str) -> None:
    """Route MediaPipe's OpenGL/EGL context to the dedicated GPU on hybrid-graphics laptops.
    Must run before the first MediaPipe task is created."""
    if kind == "nvidia":
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        egl = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if os.path.exists(egl):
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", egl)
    elif kind == "amd" and len(linux_gpus()) > 1:
        # Only with a second GPU: on an AMD APU alone the Radeon already is the default (DRI_PRIME=1 just warns).
        os.environ.setdefault("DRI_PRIME", "1")


def pick_delegate(mode: str) -> tuple[str, str]:
    """mode: auto | gpu | cpu  ->  (delegate, reason). auto = GPU only for NVIDIA / AMD graphics:
    on Intel integrated graphics (e.g. Intel UHD) MediaPipe's GPU path is no faster than the CPU."""
    mode = (mode or "auto").lower()
    if mode == "cpu":
        return "cpu", "forced by config"
    if sys.platform == "win32":
        # MediaPipe's Python GPU delegate exists only on Linux; on Windows it would just fail model by model.
        names = ", ".join(n for _, n in windows_gpus()) or "no GPU found"
        return "cpu", f"{names}; MediaPipe has no GPU support on Windows, the C++ build (MemeCam-cpp) uses the GPU"
    gpu = discrete_gpu()
    if gpu:
        use_dedicated_gpu(gpu)
        return "gpu", f"{gpu} GPU"
    if mode == "gpu":
        return "gpu", "forced by config (integrated GPU)"
    return "cpu", "no NVIDIA / AMD GPU found"


def create_task(factory, options_cls, model: str, delegate: str, **kw):
    """Create a MediaPipe task on the GPU, falling back to the CPU if the GPU delegate fails."""
    if delegate == "gpu":
        try:
            return factory(options_cls(base_options=BaseOptions(
                model_asset_path=model_path(model), delegate=BaseOptions.Delegate.GPU), **kw)), "gpu"
        except Exception as e:  # noqa: BLE001 - any driver/EGL error -> run on CPU instead
            print(f"[memecam] GPU unavailable for {model} ({e}); using CPU")
    return factory(options_cls(base_options=BaseOptions(model_asset_path=model_path(model)), **kw)), "cpu"


class Tracker:
    def __init__(self, use_pose: bool = True, pose_async: bool = True, max_people: int = 3,
                 min_face_width: float = 0.08, forget: float = ID_FORGET, device: str = "auto"):
        self.pose_async = pose_async
        self.max_people, self.min_face_width = max(1, int(max_people)), float(min_face_width)
        self.ids = Identities(forget=forget)
        n = self.max_people
        running = vision.RunningMode.VIDEO
        delegate, why = pick_delegate(device)

        def models(people: int):
            """(hands, face, pose) tasks sized for `people` people, and the devices they ended up on."""
            hands, d1 = create_task(vision.GestureRecognizer.create_from_options, vision.GestureRecognizerOptions,
                                    "gesture_recognizer.task", delegate, running_mode=running, num_hands=2 * people,
                                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                                    min_tracking_confidence=0.5)
            # Blendshapes aren't used by any gesture; skipping them saves a model run per face.
            face, d2 = create_task(vision.FaceLandmarker.create_from_options, vision.FaceLandmarkerOptions,
                                   "face_landmarker.task", delegate, running_mode=running, num_faces=people,
                                   output_face_blendshapes=False)
            pose, d3 = None, d2
            if use_pose:
                pose, d3 = create_task(vision.PoseLandmarker.create_from_options, vision.PoseLandmarkerOptions,
                                       "pose_landmarker_lite.task", delegate, running_mode=running, num_poses=people)
            return (hands, face, pose), (d1, d2, d3)

        # MediaPipe only skips its (slow) palm/face/person detectors once it is already tracking as many
        # hands/faces/bodies as it was asked for. Asked for 3 people while 1 is in view, it re-runs them
        # every frame (hands 51 ms vs 26 ms here). So keep a 1-person set for the usual case, and a
        # crowd set that runs when several people are in view, plus a periodic scan for newcomers.
        crowd, devs = models(n)
        self.crowd_set = crowd
        self.solo_set = crowd
        if n > 1:
            self.solo_set, solo_devs = models(1)
            devs += solo_devs
        self.crowd_mode = False
        self.last_multi = -1e9
        self.frame_no = 0
        self.hands, self.face, self.pose = self.solo_set
        self.device = "gpu" if "gpu" in devs else "cpu"
        print(f"[memecam] tracking on {self.device.upper()} ({why})")
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

        scan = not self.crowd_mode and self.frame_no % SCAN_EVERY == 0
        self.frame_no += 1
        self.hands, self.face, pose_task = self.crowd_set if (self.crowd_mode or scan) else self.solo_set
        if pose_task is not None and self.pose_job is None:
            self.pose = pose_task

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

        people = group_people(faces, self.last_poses, hands, w, self.max_people, self.min_face_width,
                              self.ids.recent(ts_ms / 1000))
        self.ids.assign(people, ts_ms / 1000)
        now = ts_ms / 1000
        if len(people) > 1:
            self.crowd_mode, self.last_multi = True, now
        elif self.crowd_mode and now - self.last_multi > CROWD_LINGER:
            self.crowd_mode = False
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
        for t in {id(t): t for t in (*self.solo_set, *self.crowd_set) if t is not None}.values():
            t.close()
