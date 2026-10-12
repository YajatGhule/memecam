"""Meme Cam: gesture-triggered meme webcam with mocap-style tracking markers."""
import os

# Quieter native libraries; must be set before mediapipe / OpenCV / Qt start up. Override from the shell.
os.environ.setdefault("GLOG_minloglevel", "2")          # MediaPipe C++ INFO/WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # TensorFlow Lite
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;*.warning=false;*.info=false")
if os.path.isdir("/usr/share/fonts/TTF"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/TTF")  # OpenCV's bundled Qt ships no fonts
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")     # OpenCV's Qt has no Wayland plugin; use XWayland
