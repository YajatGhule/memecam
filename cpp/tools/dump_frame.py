"""Render a reference frame with the Python app and dump its tracking state as JSON for tools/render_check.

Runs the Python synchronous path (--video/--snapshot) for N frames, then writes
  <out>.png   the Python composition (compose + hud) of the last frame
  <out>.json  the tracking result (every person), the shown memes, detections, `now`, fps
so `build/render_check --json <out>.json` can draw the same frame with the C++ renderer.

    uv run python cpp/tools/dump_frame.py ~/Videos/Webcam/Neutral.webm \
        --frames 30 --force nerd --out /tmp/rc/nerd [--v2] [--no-on-face]
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import cv2

from memecam import main as mm
from memecam import mocap
from memecam.overlays import CinemaEffect


def person_json(q) -> dict:
    return {
        "pid": q.pid,
        "face": q.face.pts.tolist() if q.face is not None else None,
        "pose": q.pose.tolist() if q.pose is not None else None,
        "hands": [{"pts": h.pts.tolist(), "label": h.label, "gesture": h.gesture, "score": h.gesture_score}
                  for h in q.hands],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--force", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.png / .json are appended)")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--no-on-face", action="store_true", help="use each meme's own placement")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--debug", action="store_true", help="draw the debug panel too")
    ap.add_argument("--placement", action="append", default=[], metavar="MEME=WHERE",
                    help="override a meme's placement (e.g. talking_to_wall=face), repeatable")
    args = ap.parse_args()

    cfg = tomllib.loads(mm.CONFIG.read_text())
    if args.v2:
        cfg = mm.merge(cfg, tomllib.loads(mm.CONFIG_V2.read_text()))
    placements = dict(p.split("=", 1) for p in args.placement)
    for name, where in placements.items():
        cfg["memes"][name]["placement"] = where
    on_face = bool(cfg.get("display", {}).get("meme_on_face", False)) and not args.no_on_face
    tracker = mm.tracker_from_config(cfg, use_pose=True, pose_async=True)
    crowd = mm.Crowd(cfg)
    cinema = CinemaEffect()
    cap, clock = mm.open_video(args.video)
    last_ts, n = -1, 0
    frame = fr = None
    shown, dets, now = [], {}, 0.0
    while n < args.frames:
        ok, img = cap.read()
        if not ok:
            break
        frame = cv2.flip(img, 1)
        now = clock.next()
        ts = max(int(now * 1000), last_ts + 1)
        last_ts = ts
        fr, dets = mm.step_crowd(tracker, crowd, frame, now, ts, args.force)
        shown = crowd.snapshot()
        n += 1
    raw = frame.copy()
    fps = 29.7
    active = mm.compose(frame, fr, shown, now, cinema, not args.no_markers, on_face)
    if args.debug:
        mm.debug_panel(frame, dets.get(fr.pid, {}), crowd.templates)
    mocap.hud(frame, fps, active, False, not args.no_markers, None, True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out) + ".png", frame)
    cv2.imwrite(str(args.out) + "_raw.png", raw)
    data = {
        "frame_index": n - 1, "now": now, "fps": fps, "on_face": on_face, "v2": args.v2,
        "markers": not args.no_markers, "debug": args.debug, "placements": placements,
        "width": fr.width, "height": fr.height,
        "top": person_json(fr), "people": [person_json(q) for q in fr.people],
        "shown": [{"pid": s.pid, "head": list(s.head) if s.head is not None else None,
                   "memes": [{"name": k, "started": m.started, "anchor": list(m.anchor) if m.anchor else None}
                             for k, m in s.memes.items() if m.started is not None]} for s in shown],
        "dets": {str(pid): {k: {"raw": bool(d.raw), "score": float(d.score)} for k, d in dd.items()}
                 for pid, dd in dets.items()},
        "active": active,
    }
    Path(str(args.out) + ".json").write_text(json.dumps(data))
    tracker.close()
    print(f"[dump] {n} frames, now={now:.3f}, people={[q.pid for q in fr.people]}, active={active} -> {args.out}.*")


if __name__ == "__main__":
    main()
