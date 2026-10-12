"""Motion-capture style rendering: glowing markers, skeletons, bracket bounding boxes, HUD."""
from __future__ import annotations

import time

import cv2
import numpy as np
from mediapipe.tasks.python.vision import (FaceLandmarksConnections, HandLandmarksConnections,
                                           PoseLandmarksConnections)

from .tracking import Frame, head_box

FACE_COL = (255, 220, 60)     # BGR cyan-ish
HAND_COL = (90, 255, 90)      # green
POSE_COL = (40, 160, 255)     # orange
TEXT_COL = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Per person id: (face, hand, pose, meme panel accent), BGR. Person 1 keeps the original look.
PERSON_COLORS = [
    (FACE_COL, HAND_COL, POSE_COL, (80, 220, 255)),
    ((255, 110, 240), (210, 140, 255), (200, 60, 255), (255, 110, 240)),     # magenta / pink
    ((90, 240, 255), (60, 200, 255), (70, 90, 255), (90, 240, 255)),         # yellow / red
    ((255, 170, 120), (255, 255, 140), (255, 120, 60), (255, 200, 120)),     # blue / aqua
]


def person_colors(pid: int) -> tuple:
    return PERSON_COLORS[(max(pid, 1) - 1) % len(PERSON_COLORS)]

FACE_EDGES = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]
FACE_DOTS = sorted({i for e in FACE_EDGES for i in e})
HAND_EDGES = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
# Body only: the face and finger points are already covered by the face/hand trackers.
POSE_EDGES = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
              if c.start >= 11 and c.end >= 11 and c.start not in range(17, 23) and c.end not in range(17, 23)]
POSE_LABELS = {11: "L_SHO", 12: "R_SHO", 13: "L_ELB", 14: "R_ELB", 15: "L_WRI", 16: "R_WRI",
               23: "L_HIP", 24: "R_HIP"}
HAND_LABELS = {0: "WRI", 4: "THB", 8: "IDX", 12: "MID", 16: "RNG", 20: "PNK"}
FACE_LABELS = {1: "NOSE", 152: "CHIN", 33: "R_EYE", 263: "L_EYE"}


def _p(pt) -> tuple[int, int]:
    return int(pt[0]), int(pt[1])


def brackets(img, box, color, label=None, pad=12, arm=18, t=2):
    """Corner-bracket bounding box, like tracking HUDs."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, t, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 8, y0), color, -1)
        cv2.putText(img, label, (x0 + 4, y0 - 5), FONT, 0.45, (10, 10, 10), 1, cv2.LINE_AA)


def _tag(img, pt, text, color):
    x, y = _p(pt)
    cv2.line(img, (x, y), (x + 10, y - 10), color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 12, y - 12), FONT, 0.33, color, 1, cv2.LINE_AA)


FACE_EDGE_IDX = np.array(FACE_EDGES)
HAND_EDGE_IDX = np.array(HAND_EDGES)
POSE_EDGE_IDX = np.array(POSE_EDGES)
POSE_DOTS = [i for i in range(11, 33) if i not in range(17, 23)]


def _segments(pts: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """Edges as 2-point int polylines, so all of them go to OpenCV in one polylines() call."""
    return list(pts[edges][:, :, :2].astype(np.int32))


def draw(img: np.ndarray, fr: Frame) -> None:
    """Draw every person's markers in place, one color scheme per person id.
    Glow comes from a blurred copy of the marker layer."""
    people = fr.people if fr.people else [fr]
    layer = np.zeros_like(img)

    for q in people:
        face_col, hand_col, pose_col, _ = person_colors(q.pid)
        if q.pose is not None:
            vis = q.pose[:, 3] > 0.5
            edges = POSE_EDGE_IDX[vis[POSE_EDGE_IDX].all(axis=1)]
            if len(edges):
                cv2.polylines(layer, _segments(q.pose, edges), False, pose_col, 2, cv2.LINE_AA)
            ip = q.pose[:, :2].astype(np.int32).tolist()
            for i in POSE_DOTS:
                if vis[i]:
                    cv2.circle(layer, ip[i], 5, pose_col, -1, cv2.LINE_AA)

        if q.face is not None:
            pts = q.face.pts
            cv2.polylines(layer, _segments(pts, FACE_EDGE_IDX), False, face_col, 1, cv2.LINE_AA)
            ip = pts[:, :2].astype(np.int32).tolist()
            for i in FACE_DOTS:
                cv2.circle(layer, ip[i], 2, face_col, -1, cv2.LINE_AA)
            for i in (1, 152, 33, 263, 10):
                cv2.circle(layer, ip[i], 4, (255, 255, 255), -1, cv2.LINE_AA)

        for hand in q.hands:
            cv2.polylines(layer, _segments(hand.pts, HAND_EDGE_IDX), False, hand_col, 2, cv2.LINE_AA)
            for i, pt in enumerate(hand.pts[:, :2].astype(np.int32).tolist()):
                cv2.circle(layer, pt, 5 if i in HAND_LABELS else 3, hand_col, -1, cv2.LINE_AA)

    # Glow: blur a quarter-size copy, double it there (img + glow + glow == img + 2*glow with saturation),
    # and upscale once, instead of two full-frame adds.
    small = cv2.resize(layer, None, fx=0.25, fy=0.25)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    glow = cv2.resize(cv2.add(small, small), (img.shape[1], img.shape[0]))
    cv2.add(img, glow, dst=img)
    # Gray > 0 as the "something drawn here" mask: layer.any(axis=2) + boolean indexing took ~30 ms/frame.
    cv2.copyTo(layer, cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY), img)

    # Labels and boxes are drawn crisp, without glow.
    for q in people:
        face_col, hand_col, pose_col, _ = person_colors(q.pid)
        who = f"PERSON_{max(q.pid, 1):02d}"
        if q.pose is not None:
            for i, name in POSE_LABELS.items():
                if q.pose[i, 3] > 0.5:
                    _tag(img, q.pose[i], name, pose_col)
        if q.face is not None:
            brackets(img, q.face.bbox, face_col, f"{who} // TRACKED")
            for i, name in FACE_LABELS.items():
                _tag(img, q.face.pts[i], name, face_col)
        elif (box := head_box(q)) is not None:
            brackets(img, box, pose_col, f"{who} // POSE")
        for hand in q.hands:
            side = "L" if hand.label == "Left" else "R"
            brackets(img, hand.bbox, hand_col, f"{side}_HAND {hand.gesture}", pad=10, arm=14)
            for i, name in HAND_LABELS.items():
                _tag(img, hand.pts[i], f"{side}_{name}", hand_col)


def hud(img: np.ndarray, fps: float, active: list[str], recording: bool, show_markers: bool,
        track_fps: float | None = None) -> None:
    h, w = img.shape[:2]
    blink = int(time.time() * 2) % 2 == 0
    if recording and blink:
        cv2.circle(img, (22, 24), 7, (0, 0, 255), -1, cv2.LINE_AA)
    rate = f"MOCAP // {fps:4.1f} FPS" + (f" (tracking {track_fps:4.1f})" if track_fps is not None else "")
    cv2.putText(img, ("REC  " if recording else "LIVE ") + rate, (36, 30), FONT, 0.55,
                TEXT_COL, 1, cv2.LINE_AA)
    if active:
        cv2.putText(img, ">> " + "  ".join(a.upper() for a in active), (36, 56), FONT, 0.55,
                    (80, 220, 255), 1, cv2.LINE_AA)
    hint = f"[d] markers {'on' if show_markers else 'off'}  [i] debug  [r] rec  [c] calibrate  [q] quit"
    cv2.putText(img, hint, (12, h - 12), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
