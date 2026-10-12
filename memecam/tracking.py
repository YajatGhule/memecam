"""MediaPipe Tasks wrappers: hands (+ canned gestures), face (+ blendshapes), pose.

All three run in VIDEO mode on the same mirrored frame, concurrently (MediaPipe releases the GIL).
Pose is the slowest model, so by default it runs decoupled: a frame waits for hands + face and
carries the newest finished pose result (usually one frame old) instead of blocking on it.
`pose_async=False` waits for pose every frame (deterministic, for offline clip evaluation).
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
    hands: list[Hand] = field(default_factory=list)
    face: Face | None = None
    pose: np.ndarray | None = None       # (33, 4) pixel x, y, z, visibility
    motion: np.ndarray | None = None     # (MOTION_H, MOTION_W) bool: pixels that changed since last frame
    width: int = 0                       # source frame size, to map `motion` back to pixels
    height: int = 0


MOTION_W, MOTION_H = 160, 90


def motion_gray(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)


class Tracker:
    def __init__(self, use_pose: bool = True, pose_async: bool = True):
        self.pose_async = pose_async
        running = vision.RunningMode.VIDEO
        self.hands = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path("gesture_recognizer.task")),
            running_mode=running, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("face_landmarker.task")),
            running_mode=running, num_faces=1, output_face_blendshapes=True))
        self.pose = None
        if use_pose:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task")),
                running_mode=running, num_poses=1))
        self.pool = ThreadPoolExecutor(3, thread_name_prefix="tracker")
        self.pose_job: tuple[Future, int, int] | None = None
        self.last_pose: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None

    def process(self, rgb: np.ndarray, ts_ms: int) -> Frame:
        h, w = rgb.shape[:2]
        scale = np.array([w, h, w])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = Frame(width=w, height=h)

        hands_job = self.pool.submit(self.hands.recognize_for_video, image, ts_ms)
        face_job = self.pool.submit(self.face.detect_for_video, image, ts_ms)
        if self.pose is not None and self.pose_job is None:
            self.pose_job = (self.pool.submit(self.pose.detect_for_video, image, ts_ms), w, h)

        res = hands_job.result()
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms]) * scale
            label = res.handedness[i][0].category_name if res.handedness else "?"
            g = res.gestures[i][0] if res.gestures and res.gestures[i] else None
            out.hands.append(Hand(pts, label, g.category_name if g else "None", g.score if g else 0.0))

        res = face_job.result()
        if res.face_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]]) * scale
            blend = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
            out.face = Face(pts, blend)

        if self.pose_job is not None and (self.pose_job[0].done() or not self.pose_async):
            job, pw, ph = self.pose_job
            self.pose_job = None
            res = job.result()
            self.last_pose = None
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                self.last_pose = np.array([[p.x * pw, p.y * ph, p.z * pw, p.visibility] for p in lm])
        out.pose = self.last_pose

        # Frame differencing: fast-moving hands blur and drop out of the hand tracker, but they
        # still light up the difference image (used for "talking to a wall").
        gray = motion_gray(rgb)
        if self.prev_gray is not None:
            out.motion = cv2.absdiff(gray, self.prev_gray) > 20
        self.prev_gray = gray
        return out

    def close(self):
        self.pool.shutdown(wait=True)
        for t in (self.hands, self.face, self.pose):
            if t is not None:
                t.close()
