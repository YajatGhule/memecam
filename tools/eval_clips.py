"""Run recorded clips through the real tracker → Detectors → Trigger pipeline and check what fires.

Usage:
  uv run python tools/eval_clips.py                          # default: ~/Videos/Webcam (+ clips/) via DEFAULT_MAP
  uv run python tools/eval_clips.py ~/Videos/Webcam clips/   # folders: files named by DEFAULT_MAP or <meme>.<ext>
  uv run python tools/eval_clips.py peace="~/Videos/Webcam/Peace sign.webm" neutral=foo.mp4
  uv run python tools/eval_clips.py --only peace,nerd --dump /tmp/memecam-scores   # per-frame CSVs
  (scuba_cat / higuruma scores in the CSV are covering_hands / 2)

A clip named after a meme passes when that meme fires; `neutral` passes when nothing fires. Other memes
firing are listed separately ("also fired") since clips often contain incidental movement.
Clips are mirrored like the live camera (config.toml [camera] mirror); use --no-mirror for clips that
were recorded already mirrored.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memecam.gestures import Detectors  # noqa: E402
from memecam.main import CONFIG, load_memes, open_video, step  # noqa: E402
from memecam.tracking import Tracker  # noqa: E402

EXTS = {".webm", ".mp4", ".mkv", ".mov", ".avi"}
NEUTRAL = "neutral"
WEBCAM_DIR = Path.home() / "Videos" / "Webcam"
# Cheese names its recordings freely; map those file names to the meme they demonstrate.
DEFAULT_MAP = {
    "Absolute cinema.webm": "absolute_cinema",
    "Gojo nah idve win.webm": "gojo",
    "Nah idve win .webm": "gojo",
    "Higuruma giving up.webm": "higuruma",
    "Maharoga.webm": "mahoraga",
    "Middle finger.webm": "middle_finger",
    "NErd.webm": "nerd",
    "Neutral.webm": "neutral",
    "Peace sign.webm": "peace",
    "Scubaaaa.webm": "scuba_cat",
    "William dafoe looking up.webm": "dafoe",
    "Yapping to a wall.webm": "talking_to_wall",
}


@dataclass
class Result:
    name: str
    path: Path
    frames: int = 0
    duration: float = 0.0
    proc_fps: float = 0.0
    fired: dict[str, list[tuple[float, float]]] = field(default_factory=dict)   # meme -> [(start, end)]
    max_score: dict[str, float] = field(default_factory=dict)
    raw_frames: dict[str, int] = field(default_factory=dict)

    def verdict(self, memes) -> tuple[str, list[str]]:
        others = [m for m in self.fired if m != self.name]
        if self.name == NEUTRAL:
            return ("PASS" if not self.fired else "FAIL"), others
        if self.name not in memes:
            return "-", others
        return ("PASS" if self.name in self.fired else "FAIL"), others


def label_for(path: Path, memes) -> str:
    if path.name in DEFAULT_MAP:
        return DEFAULT_MAP[path.name]
    stem = path.stem.strip().lower().replace(" ", "_")
    return stem if stem in memes or stem == NEUTRAL else f"?{path.stem}"


def collect(items: list[str], memes) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    if not items:
        items = [str(d) for d in (WEBCAM_DIR, ROOT / "clips") if d.is_dir()]
    for item in items:
        if "=" in item and not Path(item).exists():
            name, path = item.split("=", 1)
            clips.append((name.strip(), Path(path).expanduser()))
            continue
        p = Path(item).expanduser()
        if p.is_dir():
            clips += [(label_for(f, memes), f) for f in sorted(p.iterdir()) if f.suffix.lower() in EXTS]
        elif p.exists():
            clips.append((label_for(p, memes), p))
        else:
            raise SystemExit(f"no such clip or folder: {item}")
    return clips


def run_clip(name: str, path: Path, cfg: dict, mirror: bool, use_pose: bool, dump: Path | None) -> Result:
    memes = load_memes(cfg, load_media=False)
    detectors = Detectors(cfg.get("thresholds", {}))
    tracker = Tracker(use_pose=use_pose, pose_async=False)     # deterministic: wait for pose every frame
    cap, clock = open_video(path)
    res = Result(name, path)
    writer = rows = None
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        rows = (dump / f"{name}__{path.stem.strip().replace(' ', '_')}.csv").open("w", newline="")
        writer = csv.writer(rows)
        writer.writerow(["t", "face", "hands", "pose", "gestures",
                         *[f"{m}_{k}" for m in memes for k in ("raw", "score", "on")]])
    last_ts, t0 = -1, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = clock.next()
            ts = max(int(now * 1000), last_ts + 1)
            last_ts = ts
            was = {m: meme.started is not None for m, meme in memes.items()}
            fr, dets = step(tracker, detectors, memes, frame, now, ts)
            res.frames += 1
            res.duration = now
            for m, meme in memes.items():
                det = dets.get(m)
                if det is not None:
                    res.max_score[m] = max(res.max_score.get(m, 0.0), det.score)
                    res.raw_frames[m] = res.raw_frames.get(m, 0) + bool(det.raw)
                on = meme.started is not None
                if on and not was[m]:
                    res.fired.setdefault(m, []).append((now, now))
                if on:
                    start, _ = res.fired[m][-1]
                    res.fired[m][-1] = (start, now)
            if writer is not None:
                row = [f"{now:.3f}", int(fr.face is not None), len(fr.hands), int(fr.pose is not None),
                       "|".join(h.gesture for h in fr.hands)]
                for m, meme in memes.items():
                    det = dets.get(m)
                    row += [int(bool(det and det.raw)), f"{det.score:.3f}" if det else "", int(meme.started is not None)]
                writer.writerow(row)
    finally:
        cap.release()
        tracker.close()
        if rows is not None:
            rows.close()
    res.proc_fps = res.frames / max(time.perf_counter() - t0, 1e-6)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", help="clip files, folders, or name=path pairs")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--no-mirror", action="store_true", help="clips are already mirrored")
    ap.add_argument("--no-pose", action="store_true", help="skip body pose (faster, but hand-over-face memes need it)")
    ap.add_argument("--only", help="comma-separated clip labels to run")
    ap.add_argument("--dump", type=Path, metavar="DIR", help="write per-frame score CSVs here")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    meme_names = list(cfg.get("memes", {}))
    clips = collect(args.clips, meme_names)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        clips = [c for c in clips if c[0] in keep]
    if not clips:
        raise SystemExit("no clips found")
    mirror = cfg.get("camera", {}).get("mirror", True) and not args.no_mirror

    results = []
    for name, path in clips:
        r = run_clip(name, path, cfg, mirror, not args.no_pose, args.dump)
        results.append(r)
        verdict, others = r.verdict(meme_names)
        print(f"\n== {name}  [{path.name}]  {r.duration:.1f}s, {r.frames} frames, {r.proc_fps:.0f} fps processed")
        fired = ", ".join(f"{m} @" + " ".join(f"{a:.1f}-{b:.1f}s" for a, b in spans) for m, spans in r.fired.items())
        print(f"   fired: {fired or '-'}")
        scores = sorted(r.max_score.items(), key=lambda kv: -kv[1])
        print("   max score (raw% of frames): " + "  ".join(
            f"{m} {s:.2f} ({100 * r.raw_frames.get(m, 0) / max(r.frames, 1):.0f}%)" for m, s in scores))
        print(f"   {verdict}" + (f"   also fired: {', '.join(others)}" if others else ""))

    print("\n== summary")
    for r in results:
        verdict, others = r.verdict(meme_names)
        print(f"   {verdict:4s}  {r.name:16s} {r.path.name:34s} also fired: {', '.join(others) or '-'}")
    if args.dump:
        print(f"\nper-frame CSVs in {args.dump}")


if __name__ == "__main__":
    main()
