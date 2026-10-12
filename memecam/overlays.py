"""GIF/PNG playback and compositing onto the webcam frame."""
from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

BOLD_FONTS = ["/usr/share/fonts/noto/NotoSans-Black.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for f in BOLD_FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


class Media:
    """Animated GIF or still image, decoded once into BGRA frames.

    `trim` = (first, last) frame indices to keep. With `loop_from`, the clip plays once and then
    rocks back and forth between that frame and the end instead of restarting from the top."""

    def __init__(self, path: Path, max_side: int = 480, key_black: bool = False, label: str = "",
                 trim: tuple[int, int] | None = None, loop_from: int | None = None):
        self.frames: list[np.ndarray] = []
        self.ends: list[float] = []          # cumulative end time of each frame, seconds
        self.loop_start: float | None = None
        if path.exists():
            im = Image.open(path)
            t = 0.0
            first, last = trim if trim else (0, 1 << 30)
            for k, fr in enumerate(ImageSequence.Iterator(im)):
                if not first <= k <= last:
                    continue
                if loop_from is not None and k == loop_from:
                    self.loop_start = t
                rgba = fr.convert("RGBA")
                rgba.thumbnail((max_side, max_side), Image.LANCZOS)
                a = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
                if key_black:
                    # Footage on pure black (scuba cat): turn the black into transparency.
                    lum = a[:, :, :3].max(axis=2).astype(np.float32)
                    a[:, :, 3] = np.clip((lum - 18) * 8, 0, 255).astype(np.uint8)
                self.frames.append(a)
                t += max(fr.info.get("duration", 80), 20) / 1000
                self.ends.append(t)
        else:
            print(f"[memecam] missing asset {path.name}, using a text card")
            self.frames.append(self._card(label or path.stem))
            self.ends.append(1.0)

    @staticmethod
    def _card(text: str) -> np.ndarray:
        img = Image.new("RGBA", (420, 260), (20, 20, 20, 255))
        ImageDraw.Draw(img).multiline_text((210, 130), text.replace("_", " ").upper(), anchor="mm",
                                           fill="white", font=_font(34), align="center")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

    @property
    def duration(self) -> float:
        return self.ends[-1]

    def at(self, t: float) -> np.ndarray:
        if self.loop_start is not None and t >= self.duration:
            span = self.duration - self.loop_start
            phase = (t - self.duration) % (2 * span)
            t = self.duration - phase if phase < span else self.loop_start + (phase - span)
            t = min(t, self.duration - 1e-6)
        t %= self.duration
        i = int(np.searchsorted(self.ends, t, side="right"))
        return self.frames[min(i, len(self.frames) - 1)]


def blend(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-blend BGRA `src` onto BGR `dst` with top-left at (x, y), clipped to the frame."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    roi = dst[y0:y1, x0:x1]
    roi[:] = (s[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def fit(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)


def panel(dst: np.ndarray, img: np.ndarray, x: int, y: int, title: str, alpha: float, accent=(80, 220, 255)):
    """Meme in a HUD-style frame with a title tab."""
    h, w = img.shape[:2]
    framed = np.zeros((h + 6, w + 6, 4), np.uint8)
    framed[:, :] = (*accent, 255)
    framed[3:-3, 3:-3] = img
    blend(dst, framed, x, y, alpha)
    if title and alpha > 0.5:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(dst, (x, y - th - 10), (x + tw + 10, y), accent, -1)
        cv2.putText(dst, title, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def place_beside(frame_shape, box, w: int, h: int, margin: int = 20) -> tuple[int, int]:
    """Put a w×h overlay next to `box`, on whichever side has more room."""
    H, W = frame_shape[:2]
    x0, y0, x1, y1 = box
    x = x1 + margin if W - x1 > x0 else x0 - margin - w
    y = (y0 + y1) // 2 - h // 2
    return int(np.clip(x, 0, max(0, W - w))), int(np.clip(y, 30, max(30, H - h)))


# Face quad for warping: forehead-left, forehead-right, jaw-right, jaw-left (MediaPipe face mesh ids).
FACE_QUAD = [103, 332, 365, 136]


@functools.cache
def _oval(h: int, w: int) -> np.ndarray:
    """Feathered oval mask, cached per meme size (the big blur is the expensive part)."""
    oval = np.zeros((h, w), np.uint8)
    cv2.ellipse(oval, (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(oval, (0, 0), max(w, h) * 0.04)


def warp_on_face(dst: np.ndarray, img: np.ndarray, face_pts: np.ndarray, alpha: float, grow: float = 1.35):
    """Homography-warp the meme onto the face (Superhero_Mask_Overlay style), feathered oval mask."""
    quad = face_pts[FACE_QUAD, :2].astype(np.float32)
    c = quad.mean(axis=0)
    quad = (quad - c) * grow + c
    # Only warp/blend the quad's bounding box: doing it full-frame cost ~65 ms per frame.
    H, W = dst.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0)).astype(int).clip(0, (W, H))
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int).clip(0, (W, H))
    if x0 >= x1 or y0 >= y1:
        return
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad - np.float32([x0, y0]))
    size = (int(x1 - x0), int(y1 - y0))
    warped = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(_oval(h, w), M, size)
    a = (mask.astype(np.float32) * (warped[:, :, 3] / 255.0 / 255.0) * alpha)[..., None]
    roi = dst[y0:y1, x0:x1]
    np.copyto(roi, (warped[:, :, :3] * a + roi * (1 - a)).astype(np.uint8))


class CinemaEffect:
    """Black-and-white frame + the big 'ABSOLUTE CINEMA' caption."""

    def __init__(self):
        self.cache: dict[int, np.ndarray] = {}

    def _caption(self, width: int) -> np.ndarray:
        if width not in self.cache:
            f1, f2 = _font(width // 22), _font(width // 9)
            img = Image.new("RGBA", (width, width // 5), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.text((width // 2, width // 32), "ABSOLUTE", anchor="mt", fill="white", font=f1,
                   stroke_width=2, stroke_fill="black")
            d.text((width // 2, width // 13), "CINEMA", anchor="mt", fill="white", font=f2,
                   stroke_width=3, stroke_fill="black")
            self.cache[width] = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        return self.cache[width]

    def apply(self, dst: np.ndarray, alpha: float) -> None:
        gray = cv2.cvtColor(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=-25)        # punchier contrast
        cv2.addWeighted(gray, alpha, dst, 1 - alpha, 0, dst=dst)
        cap = self._caption(int(dst.shape[1] * 0.6))
        blend(dst, cap, (dst.shape[1] - cap.shape[1]) // 2, dst.shape[0] - cap.shape[0] - 30, alpha)
