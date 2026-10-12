# Meme Cam

Webcam with motion-capture style tracking markers that throws memes at you when you pull the right pose.

Built on MediaPipe Tasks (hands + gestures, face mesh + blendshapes, body pose) and OpenCV, following
the approach of projects like [SAT0RU](https://github.com/reinesana/SAT0RU) (finger-state hand signs) and
[Superhero_Mask_Overlay](https://github.com/kranok-dev/Superhero_Mask_Overlay) (homography face overlays).

## Run

```sh
cd ~/Projects/meme-cam
uv run memecam              # window
uv run memecam --vcam       # also publish as a virtual webcam (Discord / Meet / OBS)
uv run memecam --no-pose    # skip body tracking if FPS is low
```

Virtual webcam needs v4l2loopback once:
`sudo pacman -S v4l2loopback-dkms && sudo modprobe v4l2loopback exclusive_caps=1 card_label="Meme Cam"`

## Keys

| key | action |
|---|---|
| `d` | toggle tracking markers + bounding boxes |
| `q` / `Esc` | quit |
| `i` | gesture debug scores |
| `r` | start/stop recording to `~/Videos/memecam-*.mp4` |
| `c` | recalibrate neutral head pose |

## Gestures

| do this | get this |
|---|---|
| both hands over your face | Higuruma giving up |
| tilt your head back, look up | Willem Dafoe looking up |
| cross your index and middle finger (Gojo's Infinite Void sign) | Gojo "Nah, I'd win" |
| index finger pointing up beside your face ☝ | nerd "umm, actually..." |
| wave your hands around while talking | man talking to a wall, warped over your face |
| one hand over your mouth/nose (holding your breath), other hand swimming | scuba cat |
| both open palms up beside your head | ABSOLUTE CINEMA (frame goes black & white) |
| two fists knuckle to knuckle in front of your chest | Megumi summoning Mahoraga |
| peace sign ✌ (fingers spread) | piece of cake |
| middle finger 🖕 | a ThinkPad flips you off back |
| open palms up at chest height, bouncing up and down in turns like weighing scales ("six seven") | a huge **67** |

The detectors were tuned on real recordings. To re-check after changing thresholds:
`uv run python tools/eval_clips.py` (reads the clips in `~/Videos/Webcam`, see the name map in the script),
and `uv run pytest -q tests` for the synthetic-landmark tests.

Thresholds, placement, sizes and which file each meme uses live in `memecam/config.toml`.
Swap a meme by dropping a file into `assets/` and pointing the config at it.

`tools/make_thinkpad.py`, `tools/make_nerd.py` and `tools/make_67.py` regenerate the composed assets. For the nerd photo,
save it as `assets/nerd_source.jpg` and rerun `uv run python tools/make_nerd.py`.
