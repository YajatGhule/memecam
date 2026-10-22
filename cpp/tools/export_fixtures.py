"""Export the cached Python tracking replays as logic-parity fixtures for the C++ port.

Run with the Python repo's environment (it unpickles memecam.tracking.Frame objects and replays them
through memecam.main.Crowd):

    uv run python cpp/tools/export_fixtures.py [pkl ...]

Inputs: frames.pkl / frames2.pkl / frames3.pkl (dict filename -> (expected meme, [(now, Frame)])), made by
running memecam.tracking.Tracker (pose_async=False) over the clips in ~/Videos/Webcam. frames.pkl predates
multi-person support (top-level Frame only): each frame is wrapped as person 1, like the old single-person
replay.

Outputs (tests/fixtures/, gitignored — regenerate with this script):
  replays.bin      the tracking results, float64 exact (binary format below)
  expected.json    per clip and config (v1 = config.toml, v2 = + config_v2.toml): the memes the Python Crowd
                   fires (pid, meme, frame index, time) and every frame's detections per person
                   (raw flags + scores), for tests/test_parity.cpp

replays.bin (little endian):
  "MCFX" u32 version=1, u32 n_clips
  clip:   str source_pkl, str file, str expected, u32 n_frames, frame*
  frame:  f64 now, i32 width, i32 height, i32 pid,
          u8 has_motion [u32 n_runs, u16 runs*]   (run lengths over the 90x160 mask, alternating 0 / 1, from 0)
          u8 n_renamed (i32 old, i32 new)*, u8 n_people, person*
  person: i32 pid, u8 has_face [u16 n, (u16 index, f64 x, y, z)*]   (only the points the logic reads, plus the
          bbox extremes; every other point = point 1, which keeps the bbox exact)
          u8 has_pose [u32 uid, u8 is_new [f64 x, y, z, visibility * 33]]   (uid = Python object identity: the same
          uid on two frames is the same pose result)
          u8 n_hands, hand*
  hand:   f64 (x, y, z) * 21, str label, str gesture, f64 gesture_score
  str:    u16 length + utf-8 bytes
"""
from __future__ import annotations

import json
import pickle
import struct
import sys
import tomllib
from pathlib import Path

import numpy as np

PY_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PY_REPO))

from memecam.main import CONFIG, CONFIG_V2, Crowd, merge  # noqa: E402
from memecam.tracking import Frame  # noqa: E402

JOB = Path("/home/loopero/.claude/jobs/b6ecaa70/tmp")
DEFAULT_PKLS = [JOB / "frames.pkl", JOB / "frames2.pkl", JOB / "frames3.pkl"]
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FACE_KEEP = sorted({1, 10, 13, 14, 152, 234, 454,
                    172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397})
DETS = sorted(["middle_finger", "gojo", "godfather", "looksmaxxing", "thumbs_up", "nerd", "higuruma", "scuba_cat",
               "dafoe", "absolute_cinema", "mahoraga", "sixty_seven", "talking_to_wall"])


class Writer:
    def __init__(self):
        self.parts: list[bytes] = []
        self.pose_ids: dict[int, int] = {}

    def put(self, fmt: str, *v):
        self.parts.append(struct.pack("<" + fmt, *v))

    def str(self, s: str):
        b = s.encode()
        self.put("H", len(b))
        self.parts.append(b)

    def motion(self, m):
        if m is None:
            self.put("B", 0)
            return
        flat = np.asarray(m, bool).ravel()
        assert flat.size == 90 * 160
        runs, cur, n = [], False, 0
        for v in flat:
            if bool(v) == cur:
                n += 1
            else:
                runs.append(n)
                cur, n = bool(v), 1
        runs.append(n)
        self.put("BI", 1, len(runs))
        self.put(f"{len(runs)}H", *runs)

    def face(self, f):
        if f is None:
            self.put("B", 0)
            return
        pts = np.asarray(f.pts, float)
        keep = set(FACE_KEEP)
        keep |= {int(pts[:, 0].argmin()), int(pts[:, 0].argmax()), int(pts[:, 1].argmin()), int(pts[:, 1].argmax())}
        self.put("BH", 1, len(keep))
        for i in sorted(keep):
            self.put("Hddd", i, *map(float, pts[i, :3]))

    def pose(self, p):
        if p is None:
            self.put("B", 0)
            return
        new = id(p) not in self.pose_ids
        uid = self.pose_ids.setdefault(id(p), len(self.pose_ids))
        self.put("BIB", 1, uid, int(new))
        if new:
            arr = np.asarray(p, float)
            assert arr.shape == (33, 4)
            self.put("132d", *map(float, arr.ravel()))

    def person(self, q: Frame, pid: int):
        self.put("i", pid)
        self.face(q.face)
        self.pose(q.pose)
        self.put("B", len(q.hands))
        for h in q.hands:
            pts = np.asarray(h.pts, float)
            assert pts.shape == (21, 3)
            self.put("63d", *map(float, pts.ravel()))
            self.str(h.label)
            self.str(h.gesture)
            self.put("d", float(h.gesture_score))


def wrap(fr: Frame) -> Frame:
    """A pre-multi-person Frame as a one-person top-level frame (person 1)."""
    if hasattr(fr, "people"):
        return fr
    return Frame(hands=fr.hands, face=fr.face, pose=fr.pose, motion=fr.motion, width=fr.width, height=fr.height, pid=1)


def replay(cfg: dict, frames) -> dict:
    """Same bookkeeping as the job's replay2.py: a meme fires when it switches on for a person."""
    crowd = Crowd(cfg, load_media=False)
    fired, per_frame = [], []
    for k, (now, fr) in enumerate(frames):
        was = {(p.pid, m) for p in crowd.people.values() for m, s in p.memes.items() if s.started is not None}
        dets = crowd.update(fr, now)
        for p in crowd.people.values():
            for m, s in p.memes.items():
                if s.started is not None and (p.pid, m) not in was:
                    fired.append([p.pid, m, k, now])
        row = {}
        for pid, dd in dets.items():
            bits = sum(1 << i for i, name in enumerate(DETS) if name in dd and dd[name].raw)
            row[str(pid)] = [bits, [round(float(dd[n].score), 9) if n in dd else None for n in DETS]]
        per_frame.append(row)
    return {"fired": fired, "dets": per_frame}


def main():
    pkls = [Path(a) for a in sys.argv[1:]] or DEFAULT_PKLS
    cfgs = {"v1": tomllib.loads(CONFIG.read_text())}
    cfgs["v2"] = merge(cfgs["v1"], tomllib.loads(CONFIG_V2.read_text()))
    w = Writer()
    clips, expected = [], {"dets_order": DETS, "clips": []}
    for pkl in pkls:
        data = pickle.load(open(pkl, "rb"))
        for fn, (name, frames) in data.items():
            frames = [(now, wrap(fr)) for now, fr in frames]
            clips.append((pkl.name, fn, name, frames))
    w.put("4sII", b"MCFX", 1, len(clips))
    for src, fn, name, frames in clips:
        w.str(src)
        w.str(fn)
        w.str(name)
        w.put("I", len(frames))
        for now, fr in frames:
            people = fr.people if fr.people else ([fr] if fr.pid else [])
            for q in people:
                assert q.width == fr.width and q.height == fr.height
                assert q.motion is fr.motion or (q.motion is None and fr.motion is None) or q is fr
            renamed = getattr(fr, "renamed", {}) or {}
            w.put("diii", float(now), int(fr.width), int(fr.height), int(fr.pid))
            w.motion(fr.motion)
            w.put("B", len(renamed))
            for old, new in renamed.items():
                w.put("ii", old, new)
            w.put("B", len(people))
            for q in people:
                w.person(q, q.pid)
        entry = {"source": src, "file": fn, "expected": name, "frames": len(frames)}
        for key, cfg in cfgs.items():
            entry[key] = replay(cfg, frames)
            got = sorted({m for _, m, _, _ in entry[key]["fired"]})
            ok = (name in ("neutral", "peace") and not got) or name in got
            print(f"{'PASS' if ok else 'FAIL'} {key} {src:12s} {fn:28s} want={name:16s} fired={got}")
        expected["clips"].append(entry)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replays.bin").write_bytes(b"".join(w.parts))
    (OUT / "expected.json").write_text(json.dumps(expected, separators=(",", ":")))
    n = sum(len(c[3]) for c in clips)
    print(f"{len(clips)} clips, {n} frames, {len(w.pose_ids)} poses -> {OUT}")


if __name__ == "__main__":
    main()
