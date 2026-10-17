#!/usr/bin/env bash
# Download the meme sounds into assets/sounds/<meme>.wav (mono 44.1 kHz, loudness-normalised, <= 10 s).
# Needs yt-dlp + ffmpeg. Personal use only: the clips are copyrighted, keep them out of public repos.
# Swap any sound by dropping your own <meme>.wav into assets/sounds/ (or set `sound = "..."` in config.toml).
#   tools/fetch_sounds.sh [meme ...]      (default: all)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/assets/sounds"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
# meme -> "youtube id  start_seconds  max_seconds"
declare -A SRC=(
  [higuruma]="Y0XLbA6VGLk 0 10"          # Hiromi Higuruma meme template
  [dafoe]="b2SF5ilurOM 0 10"             # Willem Dafoe looking at the sky template
  [gojo]="eOPNyBNg92g 0 5"               # "Nah, I'd win"
  [nerd]="tr2wF1zA2mg 0 6"               # nerd emoji sound
  [talking_to_wall]="azCIFCNkmuE 0 10"   # guy talking to a wall, with music
  [scuba_cat]="1jvqMJ379rc 0 10"         # scuba cat dancing
  [absolute_cinema]="4ICFdtuy6QY 0 10"   # absolute cinema viral sound
  [mahoraga]="EPbsnTpjIo4 0 10"          # "with this treasure I summon"
  [thumbs_up]="shZWSqVM8bE 0 5"          # "great job"
  [middle_finger]="kpwNjdEPz7E 0 3"      # bruh
  [sixty_seven]="fsSt3NHYLu4 0 5"        # 6-7
  [godfather]="JMY7WZJmcJc 0 10"         # cat plays the Godfather waltz on trumpet
  [looksmaxxing]="GKZug0kDuL8 0 10"      # rock mewing
)
names=("$@"); [ ${#names[@]} -gt 0 ] || names=("${!SRC[@]}")
for m in "${names[@]}"; do
  read -r id start dur <<<"${SRC[$m]:?unknown meme $m}"
  echo "== $m ($id)"
  if ! yt-dlp -q --no-warnings --no-playlist -f bestaudio -o "$TMP/$m.%(ext)s" "https://www.youtube.com/watch?v=$id"; then
    echo "!! $m: download failed, skipped" >&2; continue
  fi
  ffmpeg -loglevel error -y -ss "$start" -t "$dur" -i "$TMP/$m".* -ac 1 -ar 44100 -sample_fmt s16 \
    -af "loudnorm=I=-16:TP=-1.5,afade=t=out:st=$(awk "BEGIN{print $dur - 0.3}"):d=0.3" "$OUT/$m.wav"
done
ls -la "$OUT"
