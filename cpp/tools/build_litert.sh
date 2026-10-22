#!/usr/bin/env bash
# Fetch LiteRT (TensorFlow Lite C++ with XNNPACK) sources into third_party/tensorflow.
# The real build is part of the main CMake project (add_subdirectory in CMakeLists.txt):
#   cmake -B build && cmake --build build
# `tools/build_litert.sh --check` additionally builds LiteRT standalone (a compiler-compatibility check).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_TAG="${TF_TAG:-v2.21.0}"
SRC="$ROOT/third_party/tensorflow"
BUILD="$ROOT/third_party/litert-build"
[ -d "$SRC" ] || git clone --depth 1 --branch "$TF_TAG" https://github.com/tensorflow/tensorflow.git "$SRC"
# MSVC: `friend class Tensor;` in Buffer doesn't see the later `class [[nodiscard]] Tensor` (C2248);
# forward-declare it. Harmless elsewhere; idempotent.
MB="$SRC/tensorflow/lite/core/model_building.h"
grep -q "^class Tensor;  // memecam" "$MB" ||
  sed -i 's/^class \[\[nodiscard\]\] Buffer {/class Tensor;  \/\/ memecam: forward declaration for MSVC\nclass [[nodiscard]] Buffer {/' "$MB"
[ "${1:-}" = "--check" ] || { echo "LiteRT sources in $SRC"; exit 0; }
# TENSORFLOW_SOURCE_DIR is required: without it tensorflow/lite/CMakeLists.txt downloads another TF release.
cmake -S "$SRC/tensorflow/lite" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DTENSORFLOW_SOURCE_DIR="$SRC" \
  -DTFLITE_ENABLE_XNNPACK=ON -DTFLITE_ENABLE_GPU=OFF -DTFLITE_ENABLE_INSTALL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-Wno-error -w -include cstdint" -DCMAKE_C_FLAGS="-Wno-error -w"
cmake --build "$BUILD" -j"$(nproc)" --target tensorflow-lite
echo "LiteRT built in $BUILD"
