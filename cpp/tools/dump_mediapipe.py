"""Dump MediaPipe (Python reference) tracking results per frame, for the C++ vision parity check.

Runs memecam.tracking.Tracker (pose_async=False, like tools/eval_clips.py) over a clip, mirrored like the
live camera, and writes the raw per-model results the tracker groups each frame (hands with gesture labels,
face meshes, poses) plus the timestamps, as JSON for `build/vision_parity`.

Run with the Python repo's environment:
  uv run python cpp/tools/dump_mediapipe.py \
      ~/Videos/Webcam/Neutral.webm -o /tmp/parity/Neutral.json [--max-frames N] [--no-mirror]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PY_ROOT))

import memecam.tracking as tracking  # noqa: E402
from memecam.main import open_video  # noqa: E402


class Timed:
    """Proxy around a MediaPipe task that accumulates the wall time of its *_for_video calls."""

    def __init__(self, task, bucket: dict, key: str):
        self._task, self._bucket, self._key = task, bucket, key

    def _wrap(self, name):
        fn = getattr(self._task, name)

        def call(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                b = self._bucket.setdefault(self._key, [0.0, 0])
                b[0] += (time.perf_counter() - t) * 1000
                b[1] += 1
        return call

    def __getattr__(self, name):
        if name.endswith("_for_video"):
            return self._wrap(name)
        return getattr(self._task, name)


def r(v: float, nd: int = 3) -> float:
    return round(float(v), nd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--max-people", type=int, default=1)
    ap.add_argument("--stateless-every", type=int, default=0,
                    help="N>0: only every Nth frame, each through a FRESH tracker (no temporal state): isolates "
                         "detector + landmark accuracy from tracking-ROI divergence")
    args = ap.parse_args()

    tr = tracking.Tracker(use_pose=True, pose_async=False, max_people=args.max_people, device="cpu")
    timing: dict[str, list] = {}
    names = ("hands", "face", "pose")
    for label, s in (("solo", tr.solo_set), ("crowd", tr.crowd_set)):
        wrapped = tuple(None if t is None else Timed(t, timing, f"{n}_{label}") for n, t in zip(names, s))
        if s is tr.solo_set:
            tr.solo_set = wrapped
        if s is tr.crowd_set:
            tr.crowd_set = wrapped
    tr.hands, tr.face, tr.pose = tr.solo_set

    captured = {}
    real_group = tracking.group_people

    def spy(faces, poses, hands, *a, **kw):
        captured["faces"], captured["poses"], captured["hands"] = faces, poses, hands
        return real_group(faces, poses, hands, *a, **kw)
    tracking.group_people = spy

    cap, clock = open_video(Path(args.clip).expanduser())
    idx = -1
    frames = []
    last_ts = -1
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and len(frames) >= args.max_frames):
            break
        idx += 1
        if args.stateless_every:
            if idx % args.stateless_every:
                clock.next()
                continue
            tr.close()
            tr = tracking.Tracker(use_pose=True, pose_async=False, max_people=args.max_people, device="cpu")
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        now = clock.next()
        ts = max(int(now * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr = tr.process(rgb, ts)
        crowd = tr.hands is tr.crowd_set[0] and tr.crowd_set is not tr.solo_set
        frames.append({
            "ts": ts,
            "idx": idx,
            "pixsum": int(rgb.sum(dtype="uint64")) if len(frames) % 30 == 0 else -1,   # decode check
            "crowd": crowd,
            "people": len(fr.people),
            "hands": [{"pts": [[r(x, 2), r(y, 2), r(z, 2)] for x, y, z in h.pts], "label": h.label,
                       "gesture": h.gesture, "score": r(h.gesture_score)} for h in captured["hands"]],
            "faces": [[[r(x, 2), r(y, 2), r(z, 2)] for x, y, z in f.pts] for f in captured["faces"]],
            "poses": [[[r(x, 2), r(y, 2), r(z, 2), r(v)] for x, y, z, v in p] for p in captured["poses"]],
        })
    elapsed = time.perf_counter() - t0
    tr.close()
    out = {
        "clip": str(Path(args.clip).expanduser()),
        "mirror": not args.no_mirror,
        "stateless": bool(args.stateless_every),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": len(frames) / max(elapsed, 1e-6),
        "timing_ms": {k: v[0] / max(v[1], 1) for k, v in timing.items()},
        "timing_calls": {k: v[1] for k, v in timing.items()},
        "frames": frames,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(frames)} frames, {out['fps']:.1f} fps, per call ms: "
          + ", ".join(f"{k} {v:.1f}" for k, v in out["timing_ms"].items()))


if __name__ == "__main__":
    main()
