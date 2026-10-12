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
uv run memecam --verbose    # show MediaPipe/OpenCV/Qt log output (hidden by default)
```

Tracking runs on a dedicated NVIDIA/AMD GPU automatically when the machine has one (hybrid laptops are
routed to the dedicated GPU), otherwise on the CPU. Override with `[tracking] device = "gpu"` or `"cpu"`
in `memecam/config.toml`; if the GPU fails to start it falls back to the CPU.

Virtual webcam needs v4l2loopback once:
`sudo pacman -S v4l2loopback-dkms && sudo modprobe v4l2loopback exclusive_caps=1 card_label="Meme Cam"`

## Keys

| key | action |
|---|---|
| `d` | toggle tracking markers + bounding boxes |
| `f` | toggle the meme overlays (detection keeps running) |
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
| talk while gesturing: hands flying around, or both open palms held up at chest height | man talking to a wall |
| one hand over your mouth/nose (holding your breath), other hand swimming | scuba cat |
| both open palms up beside your head | ABSOLUTE CINEMA (frame goes black & white) |
| two fists knuckle to knuckle in front of your chest | Megumi summoning Mahoraga |
| thumbs up 👍 | thumbs up dad |
| middle finger 🖕 | a ThinkPad flips you off back |
| open palms up at chest height, bouncing up and down in turns like weighing scales ("six seven") | a huge **67** |

By default every meme is drawn on top of the face of whoever triggered it, following the face as it moves
(when your hands hide the face, the body pose's head position is used). Absolute cinema still turns the
frame black & white with its caption. Set `[display] meme_on_face = false` in `memecam/config.toml` to use
each meme's own `placement` instead (corner panel, beside your hand, warped onto the face, ...).

## Several people

Up to `[people] max_people` people (default 3) are tracked at once. Each person gets a persistent id
(`PERSON_01`, `PERSON_02`, ... with their own marker colors) and their own gesture detectors and meme
triggers, so everyone's memes fire independently and show up on (or next to) that person.

- Faces narrower than `[people] min_face_width` × the frame width (default 0.08) are ignored, so the
  classmates in the back of the room don't count; of the rest, the largest `max_people` are kept.
- Each face takes the body pose whose nose is nearest; a pose without a face (hands over it) is still a person.
  Each hand goes to the person whose pose wrist is nearest, else to the nearest face within ~3.5 face widths;
  other hands are ignored. With just one person in view, every hand is theirs (same as before).
- A person's markers, box and memes are drawn only while they are detected in the current frame; the moment
  they leave, everything of theirs disappears. Their state is kept (frozen, nothing runs for them) for just
  `[people] forget` seconds (0.4 s) so a brief tracking dropout keeps the same id, then it is dropped.
- Ids follow the head from frame to frame. One person in view is `PERSON_01`; a `PERSON_02` appears only when a
  second person is in view. Newcomers take the lowest free id, and when only one person is left they become
  `PERSON_01` again. With several corner panels of the same kind, they sit side by side.
- More people cost more tracking time (face and hand models run per person); lower `max_people` if FPS drops.

The detectors were tuned on real recordings. To re-check after changing thresholds:
`uv run python tools/eval_clips.py` (reads the clips in `~/Videos/Webcam`, see the name map in the script),
and `uv run pytest -q tests` for the synthetic-landmark tests. (The clip evaluation looks at the largest person only.)

Thresholds, placement, sizes and which file each meme uses live in `memecam/config.toml`.
Swap a meme by dropping a file into `assets/` and pointing the config at it.

`tools/make_thinkpad.py`, `tools/make_nerd.py` and `tools/make_67.py` regenerate the composed assets. For the nerd photo,
save it as `assets/nerd_source.jpg` and rerun `uv run python tools/make_nerd.py`.
