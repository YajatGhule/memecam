# Meme Cam on Windows

> Alpha. Easiest: the no-install zips on the [Releases](https://github.com/YajatGhule/memecam/releases) page. The options below are for the Python version.

## Quick install

In PowerShell (no admin rights needed; installs uv + Python 3.12, the app and Start-menu shortcuts):

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/YajatGhule/memecam/main/install.ps1 | iex"
```

Then start **Meme Cam** from the Start menu, or run `memecam` in a new terminal. Re-run it to update.

## Manual install

1. Install [Git for Windows](https://git-scm.com/download/win) and [uv](https://docs.astral.sh/uv/getting-started/installation/). With `winget`, uv can also be installed from PowerShell:

   ```powershell
   winget install --id astral-sh.uv -e
   ```

2. Open a new PowerShell window, download the project, and install its dependencies:

   ```powershell
   git clone https://github.com/YajatGhule/memecam.git
   cd memecam
   uv python install 3.12
   uv sync
   ```

3. Start Meme Cam:

   ```powershell
   uv run memecam
   ```

The first run downloads the MediaPipe models automatically, so it needs an internet connection. Allow camera access for desktop apps in Windows Settings under **Privacy & security > Camera**.

## Use

The default camera is index `0`. If that is not the webcam you want, try another index:

```powershell
uv run memecam --camera 1
```

Press `q` or `Esc` to close the window. Press `d` to toggle tracking markers, `i` to show gesture scores, `r` to start or stop a recording, and `c` to recalibrate your neutral head pose. Add `--no-pose` if tracking is running slowly:

```powershell
uv run memecam --no-pose
```

`--vcam` currently uses the Linux-only `v4l2loopback` backend and is not available on Windows. Run the app normally to use its camera window.

## Camera troubleshooting

- Close apps that may already be using the webcam, then try `--camera 1` or another camera index.
- Check that Windows camera access and desktop-app camera access are enabled in Settings.
- If installation picks an unexpected Python version, keep using `uv python install 3.12` followed by `uv sync` from the project folder.