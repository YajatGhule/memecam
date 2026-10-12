"""Meme Cam: webcam + mocap markers + gesture-triggered memes.

Keys: d = toggle markers/bounding boxes, i = gesture debug scores, r = record,
      c = recalibrate head pose, q / Esc = quit
"""
from __future__ import annotations

import argparse
import copy
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import mocap
from .gestures import Detection, Detectors, Trigger
from .overlays import CinemaEffect, Media, blend, fit, panel, place_beside, warp_on_face
from .tracking import Frame, Tracker

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CONFIG = Path(__file__).resolve().parent / "config.toml"


class Meme:
    def __init__(self, name: str, spec: dict, load_media: bool = True):
        self.name = name
        self.title = spec.get("title", "")
        self.placement = spec.get("placement", "beside")
        self.height = spec.get("height", 0.35)
        self.media = (Media(ASSETS / spec["asset"], max_side=520, key_black=spec.get("key_black", False),
                            label=self.title or name, trim=tuple(spec["frames"]) if "frames" in spec else None,
                            loop_from=spec.get("loop_from")) if load_media else None)
        self.trigger = Trigger(hold=spec.get("hold", 0.3), min_show=spec.get("min_show", 2.0))
        self.started: float | None = None
        self.anchor = None

    def update(self, det: Detection, now: float) -> bool:
        was = self.started is not None
        if self.trigger.update(det.raw, now):
            if not was:
                self.started = now
            if det.anchor is not None and (det.raw or self.anchor is None):
                self.anchor = det.anchor
        else:
            self.started = None
            self.anchor = None
        return self.started is not None


def load_memes(cfg: dict, load_media: bool = True) -> dict[str, Meme]:
    return {name: Meme(name, spec, load_media) for name, spec in cfg.get("memes", {}).items()}


def step(tracker: Tracker, detectors: Detectors, memes: dict[str, Meme], frame: np.ndarray, now: float,
         ts: int, force=()) -> tuple[Frame, dict[str, Detection]]:
    """Track one (already mirrored) BGR frame, run the detectors and advance every meme's trigger."""
    fr = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)
    dets = detectors.run(fr, now)
    for name in force:
        dets[name] = Detection(True, 1.0, fr.hands[0].bbox if fr.hands else (fr.face.bbox if fr.face else None))
    for name, meme in memes.items():
        meme.update(dets.get(name, Detection(False, 0.0)), now)
    return fr, dets


class VideoClock:
    """Timestamps for a video file: the container's frame times, so holds/min_show behave like real time.
    Falls back to frame_idx / fps when the backend doesn't report per-frame times."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 30.0
        self.idx = 0
        self.last = -1.0

    def next(self) -> float:
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not t > self.last:
            t = max(self.idx / self.fps, self.last + 1 / self.fps)
        self.idx += 1
        self.last = t
        return t


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoClock]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video {path}")
    return cap, VideoClock(cap)


class Grabber:
    """Reads the camera (or a video file paced to its frame rate) on a background thread and keeps only
    the newest mirrored frame, so JPEG decoding never blocks the display loop and frames never queue up."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool, t0: float, pace: float | None = None,
                 loop: bool = False):
        self.cap, self.mirror, self.t0, self.pace, self.loop = cap, mirror, t0, pace, loop
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.seq = 0
        self.ended = False
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping:
                if self.pace:      # video file: deliver frames like a webcam would
                    time.sleep(max(0.0, self.t0 + (self.seq + 1) * self.pace - time.monotonic()))
                ok, frame = self.cap.read()
                if not ok and self.loop and self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    ok, frame = self.cap.read()
                if not ok:
                    break
                t = time.monotonic() - self.t0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                with self.cond:
                    self.frame, self.t, self.seq = frame, t, self.seq + 1
                    self.cond.notify_all()
        finally:
            with self.cond:
                self.ended = True
                self.cond.notify_all()

    def wait(self, after: int, timeout: float = 1.0) -> tuple[np.ndarray, float, int] | None:
        """Newest frame with seq > `after` (read-only: shared with other threads), None once the source ended."""
        with self.cond:
            while self.seq <= after and not self.ended and not self.stopping:
                self.cond.wait(timeout)
            if self.seq <= after:
                return None
            return self.frame, self.t, self.seq

    def stop(self):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        self.thread.join(timeout=2.0)


class LiveTracker:
    """Runs step() on a background thread, always on the newest grabbed frame, independent of the display.

    Meme trigger state is only ever touched by this thread; the display thread gets a consistent copy
    of every meme (`memes`) published together with the tracking result, so compose() never sees a meme
    switch off halfway through drawing it."""

    def __init__(self, grabber: Grabber, tracker: Tracker, detectors: Detectors, memes: dict[str, Meme],
                 width: int, height: int, force=()):
        self.grabber, self.tracker, self.detectors, self.memes, self.force = grabber, tracker, detectors, memes, force
        self.lock = threading.Lock()
        self.fr = Frame(width=width, height=height)
        self.dets: dict[str, Detection] = {}
        self.shown = {k: copy.copy(m) for k, m in memes.items()}
        self.count = 0
        self.error: BaseException | None = None
        self.want_calibrate = False
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="detect", daemon=True)
        self.thread.start()

    def _run(self):
        seq, last_ts = 0, -1
        try:
            while not self.stopping:
                got = self.grabber.wait(seq, timeout=0.2)
                if got is None:
                    if self.grabber.ended:
                        break
                    continue
                frame, now, seq = got
                if self.want_calibrate:
                    self.want_calibrate = False
                    self.detectors.calibrate()
                ts = max(int(now * 1000), last_ts + 1)
                last_ts = ts
                fr, dets = step(self.tracker, self.detectors, self.memes, frame, now, ts, self.force)
                shown = {k: copy.copy(m) for k, m in self.memes.items()}
                with self.lock:
                    self.fr, self.dets, self.shown = fr, dets, shown
                    self.count += 1
        except BaseException as e:  # noqa: BLE001 - re-raised on the display thread
            self.error = e

    def latest(self) -> tuple[Frame, dict[str, Detection], dict[str, Meme], int]:
        if self.error is not None:
            raise RuntimeError("tracking thread failed") from self.error
        with self.lock:
            return self.fr, self.dets, self.shown, self.count

    def calibrate(self):
        self.want_calibrate = True

    def stop(self):
        self.stopping = True
        self.thread.join(timeout=5.0)


def corner(shape, w, h, where):
    H, W = shape[:2]
    x = 20 if where[1] == "l" else W - w - 20
    y = 80 if where[0] == "t" else H - h - 40
    return x, y


def compose(frame, fr, memes, now, cinema_fx, show_markers):
    out = frame
    active = [m for m in memes.values() if m.started is not None]
    alpha = {m.name: min(1.0, (now - m.started) / 0.15) for m in active}
    H = out.shape[0]

    for m in active:
        if m.placement == "cinema":
            cinema_fx.apply(out, alpha[m.name])
    for m in active:
        if m.placement == "face" and fr.face is not None:
            warp_on_face(out, m.media.at(now - m.started), fr.face.pts, alpha[m.name])

    if show_markers:
        mocap.draw(out, fr)

    for m in active:
        if m.placement == "face" and fr.face is not None:
            continue
        img = fit(m.media.at(now - m.started), int(H * m.height))
        h, w = img.shape[:2]
        if m.placement == "center":
            # Big and frameless, horizontally centered: over the chest (just under the chin), or at
            # the top of the frame when the face sits low, whichever hides less of the face.
            W = out.shape[1]
            low, top = max(30, H - h - 10), 40
            y = int(H * 0.4)
            if fr.face is not None:
                fy0, fy1 = fr.face.bbox[1], fr.face.bbox[3]
                cands = [min(max(fy1 - h // 8, top), low), top]
                y = min(cands, key=lambda c: max(0, min(c + h, fy1) - max(c, fy0)))
            blend(out, img, (W - w) // 2, min(y, low), alpha[m.name])
            continue
        if m.placement in ("tl", "tr", "bl", "br"):
            x, y = corner(out.shape, w, h, m.placement)
        elif m.placement == "cinema":
            x, y = corner(out.shape, w, h, "tl")
        else:
            box = m.anchor or (out.shape[1] // 2 - 1, H // 2 - 1, out.shape[1] // 2, H // 2)
            x, y = place_beside(out.shape, box, w, h)
        panel(out, img, x, y, m.title, alpha[m.name])
    return [m.name for m in active]


def debug_panel(img, dets: dict[str, Detection], memes):
    y = 90
    for name in memes:
        det = dets.get(name)
        score = det.score if det else 0.0
        on = det is not None and det.raw
        col = (80, 255, 80) if on else (180, 180, 180)
        cv2.rectangle(img, (img.shape[1] - 230, y - 11), (img.shape[1] - 230 + int(min(score, 1.0) * 100), y + 1),
                      col, -1)
        cv2.putText(img, f"{name[:16]:16s}", (img.shape[1] - 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                    cv2.LINE_AA)
        y += 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, help="camera index (default from config.toml)")
    ap.add_argument("--vcam", action="store_true", help="also publish output as a v4l2loopback virtual webcam")
    ap.add_argument("--no-pose", action="store_true", help="skip body tracking (faster)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--snapshot", type=Path, help="headless: run --frames frames, save last output here, exit")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--force", action="append", default=[], help="force a meme on (testing), repeatable")
    ap.add_argument("--video", type=Path, help="read frames from this video file instead of the webcam")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the input (already-mirrored video)")
    ap.add_argument("--realtime", action="store_true",
                    help="with --video: play the clip at its frame rate through the live path (background "
                         "tracking), like a webcam; loops the clip")
    ap.add_argument("--bench", type=float, metavar="SECONDS",
                    help="headless: run the live path this long, print display FPS vs tracking FPS, exit")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    cam_cfg = cfg.get("camera", {})
    clock = None
    if args.video:
        cap, clock = open_video(args.video)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        cap = cv2.VideoCapture(args.camera if args.camera is not None else cam_cfg.get("index", 0), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, 30)
        # No CAP_PROP_BUFFERSIZE=1: with a single V4L2 buffer the camera skips every other frame (15 FPS).
        # Latency stays low anyway because the Grabber thread always drains to the newest frame.
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("could not read from the webcam — is another app using it?")
        H, W = frame.shape[:2]
    mirror = cam_cfg.get("mirror", True) and not args.no_mirror

    # Live path: a grabber thread reads frames, a tracking thread runs step() on the newest one, and this
    # loop shows every camera frame with the latest tracking result drawn on it.
    # --video (without --realtime) and --snapshot run synchronously: every frame is tracked, deterministic.
    live = args.bench is not None or (not args.snapshot and (not args.video or args.realtime))
    headless = args.snapshot is not None or args.bench is not None

    print("[memecam] loading models + memes ...")
    tracker = Tracker(use_pose=not args.no_pose, pose_async=True)
    detectors = Detectors(cfg.get("thresholds", {}))
    memes = load_memes(cfg)
    cinema_fx = CinemaEffect()

    vcam = None
    if args.vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="v4l2loopback")
            print(f"[memecam] virtual camera: {vcam.device}")
        except Exception as e:  # noqa: BLE001 - surface any backend error as a hint
            print(f"[memecam] no virtual camera ({e}).\n  sudo pacman -S v4l2loopback-dkms && "
                  "sudo modprobe v4l2loopback exclusive_caps=1 card_label='Meme Cam'")

    show_markers, show_debug = True, False
    writer = None
    fps, last = 0.0, time.perf_counter()
    t0 = time.monotonic()
    last_ts = -1
    n = 0
    out_frame = None
    fr, active = None, []
    dets: dict[str, Detection] = {}
    shown = memes
    det_fps, det_count, det_last = 0.0, 0, time.perf_counter()
    draw_time = 0.0

    grabber = live_tracker = None
    if live:
        grabber = Grabber(cap, mirror, t0, pace=1 / clock.fps if clock else None, loop=args.bench is not None)
        live_tracker = LiveTracker(grabber, tracker, detectors, memes, W, H, args.force)
    seq = 0

    def save_snapshot(img):
        cv2.imwrite(str(args.snapshot), img)
        print(f"[memecam] {n} frames, {fps:.1f} FPS, face={fr.face is not None} hands={len(fr.hands)} "
              f"pose={fr.pose is not None} active={active} -> {args.snapshot}")

    if not headless:
        cv2.namedWindow("Meme Cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Meme Cam", W, H)

    try:
        while True:
            if live:
                got = grabber.wait(seq)
                if got is None:
                    if grabber.ended:
                        break
                    continue
                frame, now, seq = got
                frame = frame.copy()          # the grabbed frame is shared with the tracking thread
                fr, dets, shown, count = live_tracker.latest()
                if count != det_count:
                    t = time.perf_counter()
                    det_fps = 0.9 * det_fps + 0.1 * (count - det_count) / max(t - det_last, 1e-3)
                    det_count, det_last = count, t
            else:
                ok, frame = cap.read()
                if not ok:
                    if args.snapshot and out_frame is not None:     # video ended early: keep the last frame
                        save_snapshot(out_frame)
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                now = clock.next() if clock else time.monotonic() - t0
                ts = max(int(now * 1000), last_ts + 1)
                last_ts = ts
                fr, dets = step(tracker, detectors, memes, frame, now, ts, args.force)

            d0 = time.perf_counter()
            active = compose(frame, fr, shown, now, cinema_fx, show_markers)

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - last, 1e-3)
            last = t
            if show_debug:
                debug_panel(frame, dets, shown)
            mocap.hud(frame, fps, active, writer is not None, show_markers, det_fps if live else None)
            draw_time += time.perf_counter() - d0

            if writer is not None:
                writer.write(frame)
            if vcam is not None:
                vcam.send(frame)

            n += 1
            if args.bench is not None:
                if time.monotonic() - t0 >= args.bench:
                    el = time.monotonic() - t0
                    _, _, _, count = live_tracker.latest()
                    print(f"[memecam] bench {el:.1f}s: display {n / el:.1f} FPS, tracking {count / el:.1f} FPS, "
                          f"compose+hud {draw_time / n * 1000:.1f} ms/frame")
                    break
                continue
            if args.snapshot:
                out_frame = frame
                if n >= args.frames:
                    save_snapshot(frame)
                    break
                continue

            cv2.imshow("Meme Cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty("Meme Cam", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("d"):
                show_markers = not show_markers
            elif key == ord("i"):
                show_debug = not show_debug
            elif key == ord("c"):
                if live_tracker is not None:
                    live_tracker.calibrate()      # detector state belongs to the tracking thread
                else:
                    detectors.calibrate()
            elif key == ord("r"):
                if writer is None:
                    out = Path.home() / "Videos" / f"memecam-{datetime.now():%Y%m%d-%H%M%S}.mp4"
                    out.parent.mkdir(exist_ok=True)
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
                    print(f"[memecam] recording to {out}")
                else:
                    writer.release()
                    writer = None
                    print("[memecam] recording stopped")
    finally:
        # Order matters: stop the tracking thread before closing the models, and the grabber before
        # releasing the capture it reads from.
        if live_tracker is not None:
            live_tracker.stop()
        if grabber is not None:
            grabber.stop()
        cap.release()
        tracker.close()
        if writer is not None:
            writer.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
