#!/usr/bin/env bash
# Meme Cam (C++) installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/YajatGhule/memecam/main/cpp/install.sh | bash
#
# Installs the build tools + OpenCV with your package manager (pacman / apt / dnf; asks for sudo), clones the
# repo, downloads LiteRT and the MediaPipe models, builds everything and puts `memecam-cpp` and
# `memecamv2-cpp` launchers in ~/.local/bin. Re-run it to update. The first build compiles LiteRT
# (TensorFlow Lite) from source: ~20-40 minutes and ~2 GB of disk; later updates take a minute.
#
# Environment overrides: MEMECAM_CPP_DIR (install dir), MEMECAM_BIN_DIR (launcher dir), JOBS (build jobs),
# MEMECAM_SKIP_DEPS=1 (don't touch system packages), MEMECAM_REF (branch / tag to install).
set -euo pipefail

REPO="${MEMECAM_REPO:-https://github.com/YajatGhule/memecam.git}"
DIR="${MEMECAM_CPP_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/memecam-cpp}"
BIN="${MEMECAM_BIN_DIR:-$HOME/.local/bin}"
REF="${MEMECAM_REF:-main}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "the C++ version is Linux-only (webcam via V4L2). On macOS / Windows use the Python version (see the repo README)"
[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root (it asks for sudo when it needs it)"

# ---- 1. system packages ------------------------------------------------------------------------------
install_deps() {
    local sudo=""
    [ "$(id -u)" -eq 0 ] || sudo="sudo"
    if command -v pacman >/dev/null; then
        say "installing build tools + OpenCV (pacman)"
        $sudo pacman -S --needed --noconfirm base-devel cmake ninja git curl unzip opencv qt6-base hdf5 ffmpeg yt-dlp \
            ttf-dejavu
    elif command -v apt-get >/dev/null; then
        say "installing build tools + OpenCV (apt)"
        $sudo apt-get update
        $sudo apt-get install -y build-essential cmake ninja-build git curl unzip pkg-config libopencv-dev \
            fonts-dejavu-core
        # freetype module for the bold meme captions (optional; plain font otherwise)
        $sudo apt-get install -y libopencv-contrib-dev 2>/dev/null || true
        $sudo apt-get install -y ffmpeg yt-dlp 2>/dev/null || true   # meme sounds (optional)
    elif command -v dnf >/dev/null; then
        say "installing build tools + OpenCV (dnf)"
        $sudo dnf install -y gcc-c++ make cmake ninja-build git curl unzip opencv-devel dejavu-sans-fonts
        $sudo dnf install -y ffmpeg-free yt-dlp 2>/dev/null || true   # meme sounds (optional)
    else
        warn "unknown package manager: install a C++20 compiler, CMake >= 3.28, Ninja, git, curl, unzip and"
        warn "OpenCV (with highgui + videoio) yourself, then re-run with MEMECAM_SKIP_DEPS=1"
        exit 1
    fi
}
[ "${MEMECAM_SKIP_DEPS:-0}" = 1 ] || install_deps

for tool in git curl unzip cmake c++; do
    command -v "$tool" >/dev/null || die "'$tool' is missing"
done
cmake_ver="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
if [ "$(printf '%s\n3.28\n' "$cmake_ver" | sort -V | head -1)" != 3.28 ]; then
    die "CMake $cmake_ver is too old (need >= 3.28). A newer one: 'pip install --user cmake' or https://cmake.org/download/"
fi
generator=()
command -v ninja >/dev/null && generator=(-G Ninja)

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
ROOT="$DIR"; cd "$DIR/cpp"

say "fetching LiteRT sources (first time only: a few hundred MB)"
tools/build_litert.sh
say "fetching MediaPipe models"
[ -f models/hand_landmarks_detector.tflite ] || tools/extract_models.sh --download

# ---- meme sounds (copyrighted clips, so not in the repo: downloaded with yt-dlp + ffmpeg) ----
if [ -z "$(ls "$ROOT/assets/sounds/"*.wav 2>/dev/null)" ]; then
    if command -v yt-dlp >/dev/null && command -v ffmpeg >/dev/null; then
        say "downloading the meme sounds"
        "$ROOT/tools/fetch_sounds.sh" >/dev/null || warn "some meme sounds failed to download (re-run tools/fetch_sounds.sh)"
    else
        warn "no yt-dlp / ffmpeg: memes stay silent. Install both, then run $ROOT/tools/fetch_sounds.sh"
    fi
fi

# ---- 3. build ----------------------------------------------------------------------------------------
say "building with $JOBS jobs (the first build compiles LiteRT: grab a coffee)"
if [ -f build/CMakeCache.txt ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
else
    cmake -S . -B build "${generator[@]}" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j"$JOBS" --target memecam memecamv2

# ---- 4. launchers ------------------------------------------------------------------------------------
mkdir -p "$BIN"
for exe in memecam memecamv2; do
    cat > "$BIN/$exe-cpp" <<EOF
#!/bin/sh
exec "$DIR/cpp/build/$exe" "\$@"
EOF
    chmod +x "$BIN/$exe-cpp"
done

apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$apps"
for v in "memecam:Meme Cam (C++)" "memecamv2:Meme Cam v2 (C++)"; do
    exe="${v%%:*}" name="${v#*:}"
    cat > "$apps/$exe-cpp.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$name
Comment=Webcam with mocap markers and gesture-triggered memes
Exec=$DIR/cpp/build/$exe
Icon=camera-web
Terminal=false
Categories=AudioVideo;Video;
EOF
done

say "done"
echo "  run:     memecam-cpp      (or memecamv2-cpp for the Godfather / looksmaxxing memes)"
echo "  keys:    d markers, f memes, i debug, n next camera, r record, c recalibrate, q quit"
echo "  update:  re-run this installer"
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on your PATH: add  export PATH=\"$BIN:\$PATH\"  to your shell rc";; esac
id -nG 2>/dev/null | grep -qw video || [ -r /dev/video0 ] || warn "no access to /dev/video0: sudo usermod -aG video $USER, then log in again"
