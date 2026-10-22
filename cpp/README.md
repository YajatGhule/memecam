# Meme Cam (C++)

> **Alpha.** Part of the [Meme Cam](../README.md) repo (`cpp/`); the Python app is in `memecam/`.

A C++20 port of Meme Cam: your webcam with motion-capture style markers (glowing face / hand /
body landmarks, corner-bracket boxes, HUD) that pops up a meme while you make the matching gesture
(thumbs up, nerd finger, Absolute Cinema, 67, ...). The Python app stays the reference implementation; this
fork matches its gestures and rendering, but is built for **one person**: it tracks whoever is closest to the
camera (two hands, one face, one body), which keeps every model in its fast tracking mode.

Two executables are built:

- `memecam`   - the original 11 gestures (`config/config.toml`)
- `memecamv2` - the same plus the v2 gestures (Godfather, looksmaxxing; `config/config_v2.toml` is layered on top)

## Install

Windows zip: see the [main README](../README.md#install). Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/YajatGhule/memecam/main/cpp/install.sh | bash
```

It installs build tools and OpenCV (pacman / apt / dnf, asks for sudo), clones the repo to
`~/.local/share/memecam-cpp`, downloads LiteRT and the models, builds, and adds `memecam-cpp` /
`memecamv2-cpp` launchers plus app-menu entries. The first build compiles LiteRT: ~20-40 minutes, ~2 GB.
Tested on Arch; Debian / Ubuntu need 24.04+ (CMake >= 3.28). Re-run to update.

## Build (from `cpp/`)

Requirements: CMake >= 3.28, a C++20 compiler (GCC 16 tested), OpenCV 5 (with highgui, videoio, and
optionally freetype for the bold captions), unzip. Everything else is fetched by CMake (toml++, CLI11,
nlohmann/json, Catch2) or built from source (LiteRT).

```sh
tools/build_litert.sh                 # clone the LiteRT (TensorFlow Lite C++) sources into third_party/tensorflow
tools/extract_models.sh [DIR]         # extract the .tflite models from MediaPipe's .task bundles into models/
                                      # (downloads the .task files when DIR doesn't have them)
cmake -B build -G Ninja               # or the default generator
cmake --build build -j6
ctest --test-dir build                # logic + parity tests (need no LiteRT)
```

LiteRT is compiled as part of the main build (first build takes a while, cached afterwards). Without it you
can still build the rendering, logic and tests: `cmake -B build-core -DMEMECAM_WITH_VISION=OFF`.

`tools/extract_models.sh` reads `gesture_recognizer.task`, `face_landmarker.task` and
`pose_landmarker_lite.task` from `../models` (or the directory you pass; downloaded from
Google's MediaPipe model storage when missing) and unpacks the palm / hand-landmark / gesture / face / pose
`.tflite` files into `models/`.

## Run

```sh
build/memecam                       # webcam, window "Meme Cam"
build/memecamv2
build/memecam --video clip.webm --snapshot out.png --frames 30 --force nerd   # headless render test
build/memecam --video clip.webm --realtime        # play a clip through the live path
build/memecam --bench 10            # headless: display FPS vs tracking FPS
build/memecam --video clip.webm --events          # print every meme switching on/off in a clip
```

Flags: `--camera N`, `--no-pose` (faster), `--config FILE`, `--video FILE`, `--realtime`, `--no-mirror`,
`--snapshot FILE --frames N`, `--force MEME` (repeatable), `--bench SECONDS`, `--events`, `--verbose` (show LiteRT /
OpenCV / Qt logs).

Keys: `d` markers and boxes, `f` meme overlays, `m` mute sounds, `i` gesture debug scores, `r` record to
`~/Videos/memecam-*.mp4`, `c` recalibrate head pose, `q` / `Esc` quit.

Settings live in `config/config.toml`: camera, tracking device, `[person]` (minimum face width, so people in
the background are ignored; dropout bridging time), `[display] meme_on_face` (put every meme over your face), the gesture
thresholds, and one `[memes.*]` table per meme (asset, title, placement, height, hold, trim `frames`,
`loop_from`, ...).

## How it works

- `src/vision/` - MediaPipe-equivalent hand, face and pose pipelines (detector -> landmark model -> ROI
  tracking) on LiteRT, using the same model files; behind the `Tracker` interface (`vision/tracker.h`).
  The detectors only run when tracking is lost; every 15 (face) / 30 (body) frames the detector also checks
  for a clearly bigger face/body than the tracked one and switches to it, so it never sticks to someone in
  the background.
- Hands go beyond MediaPipe's graph: a visible body-pose wrist with no hand seeds a hand crop (as MediaPipe
  Holistic does), tracked crops move with the hand's velocity, the output points are one-euro smoothed, and
  Left / Right comes from the body pose. On the recorded clips hands are found at 95.4 % of visible wrists
  (MediaPipe: 92.0 %). `MEMECAM_HAND_MP=1` turns all of it off (plain MediaPipe), `MEMECAM_HAND_OFF=seeds,
  predict,smooth,handed` single parts.
- `src/tracking.*` - attaches hands and body to the face (and drops background faces), motion map.
- `src/gestures.*` - all gesture detectors and the `Trigger` state machine (1:1 with `gestures.py`). A meme
  shows only while its gesture is active (hold 0.3 s to start, 0.3 s linger after).
- `src/crowd.*` - the tracked `Person`'s detectors and meme triggers (kept through short tracking dropouts).
- `src/overlays.*` - GIF/PNG `Media` (trim, `loop_from` rocking loop), alpha blend, fit/panel caches,
  `over_face`, `warp_on_face`, the black-and-white Cinema effect.
- `src/mocap.*` - glowing markers, brackets, labels, HUD.
- `src/compose.*` - draws the active memes and markers onto a frame.
- `src/app.cpp` - a grabber thread (newest frame only), a tracking thread, and the display loop that shows
  every camera frame with the latest tracking result; recording and the key handling.

Tracking check against the Python app: `tools/dump_mediapipe.py` (run with `uv run` from the repo root)
dumps MediaPipe's landmarks for a clip; `build/vision_parity dump.json...` runs the C++ tracker on the same
frames. On 15 recorded clips (~3000 frames): face mesh within 0.03 % of the face width (median), hands
within 0.5 % of the hand size (median, 99 % handedness / gesture-label agreement), body within ~4 % of the
shoulder width (MediaPipe's smoothing amplifies tiny differences on the off-screen legs). End to end,
`--events` on each of the 20 recorded clips fires exactly the clip's meme and nothing else (v1 and v2).

Rendering check against the Python app: `tools/dump_frame.py` (run from the repo root) writes a rendered
frame plus its tracking state; `build/render_check <prefix>` redraws it with the C++ renderer and writes
`<prefix>_cmp.png` (python | c++ | diff x4) and the pixel difference. All memes match to within rounding
(only the Cinema caption differs by a few pixels from the font rasteriser).

## Meme sounds

See the [main README](../README.md#meme-sounds); `../tools/fetch_sounds.sh` downloads them into `../assets/sounds`.

## Meme assets

The GIFs and images in `assets/` are copyrighted by their respective owners (films, anime, memes). They are
included for personal use only; do not redistribute them or publish them in a public repository without
checking the rights.
