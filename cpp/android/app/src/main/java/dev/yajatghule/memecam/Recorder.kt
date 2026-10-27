package dev.yajatghule.memecam

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.view.Surface

/**
 * Records the rendered frames (markers + memes) to an MP4 in Movies/MemeCam, with the microphone when allowed.
 * Frames are drawn onto MediaRecorder's input surface as they are shown.
 */
class Recorder(private val context: Context, width: Int, height: Int, withMic: Boolean) {
    val uri: Uri
    private val fd: ParcelFileDescriptor
    private val recorder: MediaRecorder
    private val surface: Surface
    @Volatile private var stopped = false

    init {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "memecam-${System.currentTimeMillis()}.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/MemeCam")
            put(MediaStore.Video.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: error("can't create the video file")
        fd = resolver.openFileDescriptor(uri, "w") ?: error("can't open the video file")
        recorder = (if (Build.VERSION.SDK_INT >= 31) MediaRecorder(context) else @Suppress("DEPRECATION") MediaRecorder()).apply {
            if (withMic) setAudioSource(MediaRecorder.AudioSource.MIC)
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(fd.fileDescriptor)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(width and 1.inv(), height and 1.inv())
            setVideoFrameRate(30)
            setVideoEncodingBitRate(6_000_000)
            if (withMic) {
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setAudioSamplingRate(44100)
                setAudioEncodingBitRate(128_000)
            }
            prepare()
        }
        surface = recorder.surface
        recorder.start()
    }

    /** Called with every shown frame (any thread). */
    @Synchronized fun frame(bm: Bitmap) {
        if (stopped) return
        try {
            val canvas = surface.lockHardwareCanvas()
            canvas.drawBitmap(bm, null, android.graphics.Rect(0, 0, canvas.width, canvas.height), null)
            surface.unlockCanvasAndPost(canvas)
        } catch (_: Exception) {
        }
    }

    /** Finishes the file; true when a playable video was saved. */
    @Synchronized fun stop(): Boolean {
        if (stopped) return false
        stopped = true
        val ok = try { recorder.stop(); true } catch (_: RuntimeException) { false }   // throws if no frame arrived
        recorder.release()
        surface.release()
        fd.close()
        val resolver = context.contentResolver
        if (ok) resolver.update(uri, ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }, null, null)
        else resolver.delete(uri, null, null)
        return ok
    }
}
