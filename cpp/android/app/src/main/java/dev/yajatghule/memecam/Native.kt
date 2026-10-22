package dev.yajatghule.memecam

import android.graphics.Bitmap
import java.nio.ByteBuffer

/** JNI bridge to the C++ engine (cpp/android/engine.cpp). */
object Native {
    init {
        System.loadLibrary("c++_shared")
        System.loadLibrary("opencv_java4")
        System.loadLibrary("memecam_android")
    }

    /** Loads config, models and memes from [root]; null on success, else an error message. */
    @JvmStatic external fun init(root: String, v2: Boolean): String?

    /** One RGBA camera frame in, the finished upright frame drawn into [out]. Returns the active memes. */
    @JvmStatic external fun process(
        rgba: ByteBuffer, width: Int, height: Int, rowStride: Int, rotation: Int, mirror: Boolean, out: Bitmap,
    ): String?

    @JvmStatic external fun setOptions(markers: Boolean, memes: Boolean, mute: Boolean)
    @JvmStatic external fun calibrate()
    @JvmStatic external fun release()
}
