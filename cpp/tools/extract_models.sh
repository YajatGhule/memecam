#!/usr/bin/env bash
# Pull the .tflite models out of MediaPipe's .task bundles (zip files, the hand one nested) into models/.
# The .task files come from DIR (default: the Python repo's models/, if present) or are downloaded from
# Google's MediaPipe model storage (the same files the Python app uses).
#   tools/extract_models.sh [DIR | --download]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/../models}"
OUT="$ROOT/models"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
BASE="https://storage.googleapis.com/mediapipe-models"
declare -A URL=(
  [gesture_recognizer]="$BASE/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
  [face_landmarker]="$BASE/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
  [pose_landmarker_lite]="$BASE/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
mkdir -p "$OUT"
for t in gesture_recognizer face_landmarker pose_landmarker_lite; do
  task="$SRC/$t.task"
  if [ "$SRC" = --download ] || [ ! -f "$task" ]; then
    task="$TMP/$t.task"
    echo "downloading $t.task ..."
    curl -fsSL --retry 3 -o "$task" "${URL[$t]}"
  fi
  mkdir -p "$TMP/$t" && unzip -qo "$task" -d "$TMP/$t"
done
for inner in "$TMP"/gesture_recognizer/*.task; do unzip -qo "$inner" -d "$TMP/gesture_recognizer"; done
find "$TMP" -name '*.tflite' -exec cp {} "$OUT/" \;
ls "$OUT"
