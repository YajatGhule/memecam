#!/usr/bin/env bash
# Meme Cam installer for Linux and macOS (Windows: install.ps1).
#
#   curl -fsSL https://raw.githubusercontent.com/YajatGhule/memecam/main/install.sh | bash
#
# Installs uv if needed (it brings its own Python 3.12, nothing system-wide), clones the repo, installs the
# dependencies, downloads the MediaPipe models and puts `memecam` / `memecamv2` launchers in ~/.local/bin
# (plus app-menu entries on Linux). Re-run it to update.
#
# Environment overrides: MEMECAM_DIR (install dir), MEMECAM_BIN_DIR (launcher dir), MEMECAM_REF (branch / tag).
set -euo pipefail

REPO="${MEMECAM_REPO:-https://github.com/YajatGhule/memecam.git}"
DIR="${MEMECAM_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/memecam}"
BIN="${MEMECAM_BIN_DIR:-$HOME/.local/bin}"
REF="${MEMECAM_REF:-main}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root"
command -v git >/dev/null || die "git is missing: install it first (e.g. sudo pacman -S git / sudo apt install git / xcode-select --install)"
command -v curl >/dev/null || die "curl is missing"

# ---- 1. uv (Python + package manager) ----------------------------------------------------------------
UV="$(command -v uv || true)"
[ -n "$UV" ] || { [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"; } || true
if [ -z "$UV" ]; then
    say "installing uv (https://docs.astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || die "uv install failed"
fi

# ---- 2. source ---------------------------------------------------------------------------------------
if [ -d "$DIR/.git" ]; then
    say "updating $DIR"
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" checkout -q -B "$REF" FETCH_HEAD
else
    say "cloning into $DIR"
    mkdir -p "$(dirname "$DIR")"
    git clone --depth 1 --branch "$REF" "$REPO" "$DIR"
fi

# ---- 3. dependencies + models ------------------------------------------------------------------------
say "installing Python 3.12 + dependencies (MediaPipe, OpenCV, ...)"
"$UV" sync --project "$DIR" --no-dev --python 3.12
say "downloading the MediaPipe models"
"$UV" run --project "$DIR" --no-dev python -c \
    "from memecam.tracking import MODELS, model_path; [model_path(m) for m in MODELS]"

# ---- 4. launchers ------------------------------------------------------------------------------------
mkdir -p "$BIN"
for exe in memecam memecamv2; do
    cat > "$BIN/$exe" <<EOF
#!/bin/sh
exec "$UV" run --project "$DIR" --no-dev --quiet $exe "\$@"
EOF
    chmod +x "$BIN/$exe"
done

if [ "$(uname -s)" = Linux ]; then
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    for v in "memecam:Meme Cam" "memecamv2:Meme Cam v2"; do
        exe="${v%%:*}" name="${v#*:}"
        cat > "$apps/$exe.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$name
Comment=Webcam with mocap markers and gesture-triggered memes
Exec=$BIN/$exe
Icon=camera-web
Terminal=false
Categories=AudioVideo;Video;
EOF
    done
fi

say "done"
echo "  run:     memecam          (or memecamv2 for the Godfather / looksmaxxing memes)"
echo "  keys:    d markers, f memes, i debug, r record, c recalibrate, q quit"
echo "  update:  re-run this installer"
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on your PATH: add  export PATH=\"$BIN:\$PATH\"  to your shell rc";; esac
if [ "$(uname -s)" = Linux ] && [ -e /dev/video0 ] && [ ! -r /dev/video0 ]; then
    warn "no access to /dev/video0: sudo usermod -aG video $USER, then log in again"
fi
