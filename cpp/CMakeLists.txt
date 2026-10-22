cmake_minimum_required(VERSION 3.28)
project(memecam_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # M_PI, no min/max macros from <windows.h>, UTF-8 sources, the C++ __cplusplus value, parallel compiles.
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
  add_compile_options(/utf-8 /Zc:__cplusplus /MP)
endif()

if(ANDROID)   # the APK: OpenCV Android SDK (core imgproc imgcodecs videoio, no window), no desktop app / tests / tools
  # Not find_package(OpenCV): its config imports a target called libprotobuf, which clashes with LiteRT's own.
  # The SDK's single shared library (libopencv_java4.so, all modules) and its headers are all that is needed.
  get_filename_component(_ocv_native "${OpenCV_DIR}/.." ABSOLUTE)
  set(OpenCV_INCLUDE_DIRS ${_ocv_native}/jni/include)
  set(OpenCV_LIBS ${_ocv_native}/libs/${ANDROID_ABI}/libopencv_java4.so)
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
endif()
find_package(Threads REQUIRED)
include(cmake/deps.cmake)

# Core: tracking logic, gestures, crowd, config, rendering — everything except model inference.
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/src/(app|main)[^/]*\\.cpp$")
add_library(memecam_core STATIC ${CORE_SOURCES})
target_include_directories(memecam_core PUBLIC src)
target_include_directories(memecam_core SYSTEM PRIVATE ${miniaudio_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(memecam_core PRIVATE ${CMAKE_DL_LIBS} m)
endif()
target_link_libraries(memecam_core PUBLIC ${OpenCV_LIBS} Threads::Threads tomlplusplus::tomlplusplus)
target_compile_definitions(memecam_core PUBLIC MEMECAM_ROOT="${CMAKE_SOURCE_DIR}")
# Rendering: bold TTF captions through OpenCV's freetype module when available (else OpenCV's built-in font).
if(TARGET opencv_geometry)   # OpenCV 5: getPerspectiveTransform etc. moved here from imgproc
  target_link_libraries(memecam_core PUBLIC opencv_geometry)
endif()
if(TARGET opencv_freetype)
  target_link_libraries(memecam_core PUBLIC opencv_freetype)
  target_compile_definitions(memecam_core PUBLIC MEMECAM_HAVE_FREETYPE=1)
endif()
# Dev tool: C++ rendering of a frame dumped by tools/dump_frame.py, compared with the Python rendering.
if(NOT ANDROID)
  add_executable(render_check tools/render_check.cpp)
  target_link_libraries(render_check PRIVATE memecam_core nlohmann_json::nlohmann_json)
endif()

# Vision: MediaPipe-equivalent pipelines on LiteRT (TensorFlow Lite C++).
option(MEMECAM_WITH_VISION "Build the LiteRT vision pipelines and the app" ON)
set(TF_SRC ${CMAKE_SOURCE_DIR}/third_party/tensorflow)
if(MEMECAM_WITH_VISION)
  if(NOT EXISTS ${TF_SRC}/tensorflow/lite/CMakeLists.txt)
    message(FATAL_ERROR "LiteRT source missing: run tools/build_litert.sh (or -DMEMECAM_WITH_VISION=OFF)")
  endif()
  set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  # Without this, tensorflow/lite/CMakeLists.txt downloads a *different* TF release (v2.19) and mixes it
  # with this tree's sources (missing-header errors in the XNNPACK delegate).
  set(TENSORFLOW_SOURCE_DIR ${TF_SRC} CACHE PATH "" FORCE)
  # CMake 4 dropped compatibility with cmake_minimum_required(< 3.5) used by some LiteRT deps.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "")
  # GCC 16: some LiteRT/abseil/XNNPACK headers use uint*_t without <cstdint>. Only LiteRT gets these flags.
  set(_memecam_cxx_flags "${CMAKE_CXX_FLAGS}")
  set(_memecam_c_flags "${CMAKE_C_FLAGS}")
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstdint -Wno-error -w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -w")
  endif()
  add_subdirectory(${TF_SRC}/tensorflow/lite ${CMAKE_BINARY_DIR}/litert EXCLUDE_FROM_ALL SYSTEM)
  set(CMAKE_CXX_FLAGS "${_memecam_cxx_flags}")
  set(CMAKE_C_FLAGS "${_memecam_c_flags}")

  file(GLOB VISION_SOURCES CONFIGURE_DEPENDS src/vision/*.cpp)
  list(FILTER VISION_SOURCES EXCLUDE REGEX ".*_main\\.cpp$")
  if(VISION_SOURCES)
    add_library(memecam_vision STATIC ${VISION_SOURCES})
    target_include_directories(memecam_vision PUBLIC src)
    target_link_libraries(memecam_vision PUBLIC memecam_core tensorflow-lite)
    if(NOT ANDROID)
      # Parity/benchmark tool: runs the Tracker on a clip against tools/dump_mediapipe.py output.
      add_executable(vision_parity src/vision/parity_main.cpp)
      target_link_libraries(vision_parity PRIVATE memecam_vision nlohmann_json::nlohmann_json)
    endif()
  endif()

  if(ANDROID)   # JNI library of the Android app (android/)
    add_library(memecam_android SHARED android/engine.cpp)
    target_link_libraries(memecam_android PRIVATE memecam_vision jnigraphics log android)
    return()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/src/app.cpp AND VISION_SOURCES)
    foreach(exe memecam memecamv2)
      add_executable(${exe} src/app.cpp)
      target_link_libraries(${exe} PRIVATE memecam_vision CLI11::CLI11)
    endforeach()
    target_compile_definitions(memecamv2 PRIVATE MEMECAM_V2=1)
  endif()
endif()

# Tests (no LiteRT needed for the logic tests).
enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
if(TEST_SOURCES)
  add_executable(memecam_tests ${TEST_SOURCES})
  target_link_libraries(memecam_tests PRIVATE memecam_core Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
  include(Catch)
  catch_discover_tests(memecam_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
