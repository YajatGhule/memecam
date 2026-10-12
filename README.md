# Meme Cam

Webcam with motion-capture style tracking markers that throws memes at you when you pull the right pose.

Built on MediaPipe Tasks (hands + gestures, face mesh + blendshapes, body pose) and OpenCV, following
the approach of projects like [SAT0RU](https://github.com/reinesana/SAT0RU) (finger-state hand signs) and
[Superhero_Mask_Overlay](https://github.com/kranok-dev/Superhero_Mask_Overlay) (homography face overlays).

## Run

```shcd ~/Projects/meme-cam
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
| roll your eyes up | Higuruma looking up |
| tilt your head back, look up | Willem Dafoe looking up |
| one-sided cocky smirk | Gojo "Nah, I'd win" |
| index finger pointing up ☝ | nerd "umm, actually..." |
| wave your hands around while talking | man talking to a wall, warped over your face |
| open your mouth wide | scuba cat |
| both open palms up beside your head | ABSOLUTE CINEMA (frame goes black & white) |
| fist against your other hand in front of you | Megumi summoning Mahoraga |
| peace sign ✌ | piece of cake |
| middle finger 🖕 | a ThinkPad flips you off back |

Thresholds, placement, sizes and which file each meme uses live in `memecam/config.toml`.
Swap a meme by dropping a file into `assets/` and pointing the config at it.

`tools/make_thinkpad.py` and `tools/make_nerd.py` regenerate the two composed assets. For the nerd photo,
save it as `assets/nerd_source.jpg` and rerun `uv run python tools/make_nerd.py`.
