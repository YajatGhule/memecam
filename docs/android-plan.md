# Meme Cam for Android (APK) — plan

## Goal
A sideloadable Android APK with the same features as the laptop app: front-camera mocap markers,
multi-person tracking, and every gesture → meme (Higuruma, Dafoe, Gojo, nerd, talking to a wall,
scuba cat, absolute cinema, Mahoraga, thumbs up dad, middle finger → ThinkPad, 67). Personal use /
friends; no Play Store release (the memes are copyrighted).

## Target phone: iQOO Z5
- Snapdragon 778G (8 cores, up to 2.4 GHz), Adreno 642L GPU, 6/8/12 GB RAM, 120 Hz screen,
  16 MP front camera; ships with Android 11 (Funtouch OS 12), updatable to Android 13.
- Plenty for this app: MediaPipe's web GPU path on a 778G should run all three models at roughly
  **20–30 tracking FPS** at 640×480 — faster than the ThinkPad on battery. Camera preview stays at 30 FPS.
- Build target: minSdk 24, targetSdk 35 (covers Android 11–13+); arm64-v8a only.
- Funtouch OS specifics:
  - Developer options: Settings → About phone → Software version → tap 7 times.
    Then Developer options → **USB debugging** on, and **Install via USB** on (vivo/iQOO block
    `adb install` without it; it may ask you to sign in to a vivo account once).
  - Sideloading a copied APK: allow "Install unknown apps" for the file manager.
  - Battery: add Meme Cam to "High background power consumption"/disable battery optimisation only if
    recording gets cut off; not needed for normal foreground use.
  - Funtouch can show its own "this app is unverified / scan" prompt on install — tap install anyway.

## Approach
TypeScript web app + **MediaPipe Tasks Vision (web)** + **Capacitor** to package it as an APK.
- Same models as the Python app (gesture_recognizer, face_landmarker, pose_landmarker_lite), running on
  the phone GPU (WebGL) — expected 20–30 tracking FPS on a mid-range phone.
- One codebase also runs in a phone browser (quick testing, and the iOS route later).
- Alternatives rejected: native Kotlin (≈2× the work, no gain for this app), Flutter (weak MediaPipe
  support), Python on Android (MediaPipe/OpenCV don't run there).

## What gets ported (from `memecam/`)
| Python | Web/TS | Notes |
|---|---|---|
| `tracking.py` (Tracker, group_people, Identities) | `tracking.ts` | MediaPipe JS returns the same landmarks; port grouping/ids 1:1 |
| `gestures.py` (Detectors, Trigger) | `gestures.ts` | pure math on landmarks → straight port; numpy → small vector helpers |
| motion map (frame differencing) | `motion.ts` | 160×90 canvas, `getImageData` diff |
| `mocap.py` | `mocap.ts` | Canvas 2D: glow via `shadowBlur`, brackets, labels |
| `overlays.py` (GIF playback, blend, cinema) | `overlays.ts` | decode GIFs once with `gifuct-js` into ImageBitmaps; cinema = `filter: grayscale` |
| `config.toml` | `config.json` | same keys |
| `tests/` (172 tests) | `vitest` | port the synthetic-landmark tests; they are the safety net for the port |
| clip tuning (`tools/eval_clips.py`) | keep in Python | thresholds stay tuned on the laptop; a parity test checks TS == Python |

## Phases
1. **Toolchain (½ day)**
   - Install: `npm` (Arch `npm` package), JDK 21 (`jdk21-openjdk`; Android Gradle needs 17/21, system has 26),
     Android SDK command-line tools + platform 35 + build-tools (≈6–8 GB).
   - ⚠ Disk: only ~13 GB free on /home — clear space first.
   - Phone: enable Developer options → USB debugging + Install via USB (see Target phone);
     plug in with a data USB-C cable, accept the RSA prompt, check with `adb devices`.
2. **Web app skeleton (1 day)**
   - Vite + TypeScript; front camera via `getUserMedia({facingMode: "user"})`, mirrored, portrait.
   - Load the three MediaPipe tasks (models bundled in the app, not downloaded at runtime).
   - Draw markers on a canvas over the video. Milestone: dots on face/hands/body on the phone browser.
3. **Port the logic (2–3 days)**
   - Port `gestures.py`, grouping/identities, Trigger, motion, then the 172 tests to vitest.
   - **Parity check:** dump landmarks from the cached clips (`frames*.pkl`) to JSON, run both Python and TS
     detectors on them, require identical fired memes per clip.
4. **Overlays & UX (1–2 days)**
   - Memes on the triggering person's face (current default), Absolute Cinema B&W + caption, 67 overlay.
   - Phone UI: markers toggle (replaces `d`), debug toggle, front/back camera switch,
     **record + share** button (MediaRecorder on the canvas → share sheet).
5. **Package the APK (½ day)**
   - `npx cap add android`; camera permission in AndroidManifest; lock to portrait; keep screen on.
   - Debug APK: `cd android && ./gradlew assembleDebug` → `app/build/outputs/apk/debug/app-debug.apk`.
   - Install: `adb install -r app-debug.apk` (or copy the file to the phone).
   - Optional signed release APK (keystore) so friends can install updates over the old one.
6. **Test & tune on the phone (1 day)**
   - Real-phone check of every gesture on the Z5; FPS overlay on screen; heat/battery over 10 minutes.
   - Use the Z5's own front camera to record a few clips per gesture (the app's record button) and add
     them to the parity/tuning set.
   - Phone front cameras frame you closer than the laptop — re-check `min_face_width` and distance-based
     thresholds; record a few phone clips if something misfires.

Rough total: **~6–8 working days.**

## Layout
```
meme-cam/
  android-app/          # new
    src/ tracking.ts gestures.ts motion.ts mocap.ts overlays.ts main.ts
    public/assets/      # memes (copied from ../assets) + MediaPipe .task models
    config.json
    tests/              # vitest ports of tests/*.py + parity fixtures
    capacitor.config.ts
    android/            # generated by Capacitor
  memecam/              # existing Python app stays the reference implementation
```

## Risks
- **Performance:** not a concern on the 778G; still keep 640×480 input and an option to run pose every
  other frame, and watch heat during long recordings (the Z5 throttles after ~10 min of heavy GPU use).
- **Behaviour drift between Python and TS:** mitigated by the parity check on the recorded clips.
- **Front-camera differences** (field of view, auto-exposure): may need small threshold changes.
- **GIF memory:** decode at reduced size (as the Python app does, max 520 px).

## Needed from you
- The iQOO Z5 with USB debugging + Install via USB on, and its Android version
  (Settings → About phone) so the build targets it exactly.
- Free ~10 GB disk.
- OK to install: npm, JDK 21, Android SDK.
