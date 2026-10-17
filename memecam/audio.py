"""Meme sounds: each meme's sound plays while the meme is on screen.

A sound starts from the beginning when its meme switches on and fades out quickly when the meme switches
off (or when the clip ends). Several memes can sound at once; they are mixed in one output stream.
Sounds are 16-bit PCM WAV files: `sound = "..."` in a meme's config table (relative to assets/), else
`assets/sounds/<meme>.wav` when it exists (tools/fetch_sounds.sh downloads a default set).
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

RATE = 44100
FADE = 0.08          # seconds of fade-out when a meme switches off


def load_wav(path: Path) -> np.ndarray | None:
    """Mono float32 samples at RATE, or None if the file can't be read."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (OSError, EOFError, wave.Error) as e:
        print(f"[memecam] can't read sound {path.name}: {e}")
        return None
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    else:
        print(f"[memecam] unsupported sample width in {path.name}")
        return None
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE and len(x) > 1:          # linear resample
        n = int(len(x) * RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def sound_paths(cfg: dict, assets: Path) -> dict[str, Path]:
    out = {}
    for name, spec in cfg.get("memes", {}).items():
        p = assets / spec["sound"] if spec.get("sound") else assets / "sounds" / f"{name}.wav"
        if p.exists():
            out[name] = p
    return out


class Sounds:
    """Mixer for the meme sounds. `update(active)` once per displayed frame with the active meme names."""

    def __init__(self, cfg: dict, assets: Path):
        acfg = cfg.get("audio", {})
        self.volume = float(acfg.get("volume", 0.8))
        self.muted = False
        self.clips = {}
        self._voices: dict[str, list] = {}   # name -> [samples, position, gain step (0 = playing)]
        self._lock = threading.Lock()
        self._stream = None
        if not acfg.get("enabled", True):
            return
        for name, p in sound_paths(cfg, assets).items():
            x = load_wav(p)
            if x is not None and len(x):
                self.clips[name] = x
        if not self.clips:
            return
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32", callback=self._mix)
            self._stream.start()
        except Exception as e:  # noqa: BLE001 - no audio device / PortAudio: run silently
            print(f"[memecam] no audio output ({e}); memes stay silent")
            self._stream = None

    @property
    def ok(self) -> bool:
        return self._stream is not None

    def update(self, active) -> None:
        if self._stream is None:
            return
        names = {a.split("#")[0] for a in active} if not self.muted else set()
        with self._lock:
            for n in names:
                if n in self.clips and n not in self._voices:
                    self._voices[n] = [self.clips[n], 0, 0.0, 1.0]
            for n, v in self._voices.items():
                if n not in names and v[2] == 0.0:
                    v[2] = 1.0 / (FADE * RATE)       # start fading out

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def _mix(self, out, frames, _time, _status):
        buf = np.zeros(frames, np.float32)
        with self._lock:
            done = []
            for n, v in self._voices.items():
                x, pos, step, gain = v
                chunk = x[pos:pos + frames]
                k = len(chunk)
                if step:
                    g = gain - step * np.arange(1, k + 1, dtype=np.float32)
                    np.clip(g, 0, 1, out=g)
                    buf[:k] += chunk * g
                    v[3] = float(g[-1]) if k else 0.0
                else:
                    buf[:k] += chunk
                v[1] = pos + k
                if v[1] >= len(x) or v[3] <= 0.0:
                    done.append(n)
            for n in done:
                # a finished clip stays "playing" (silent) until its meme switches off, so it doesn't restart
                if self._voices[n][2] == 0.0:
                    self._voices[n][1] = len(self._voices[n][0])
                else:
                    del self._voices[n]
        out[:, 0] = np.clip(buf * self.volume, -1, 1)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
