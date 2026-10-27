package dev.yajatghule.memecam

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val analysis = Executors.newSingleThreadExecutor()
    private val prefs by lazy { getSharedPreferences("memecam", MODE_PRIVATE) }
    private lateinit var image: ImageView
    private lateinit var status: TextView
    private lateinit var buttons: List<Button>
    private val bitmaps = arrayOfNulls<Bitmap>(2)
    private var flip = 0
    @Volatile private var lastShown: Bitmap? = null
    @Volatile private var recorder: Recorder? = null
    private lateinit var recButton: Button
    @Volatile private var ready = false
    private var front = true
    private var markers = true
    private var memes = true
    private var mute = false
    private var debug = false
    private var v2 = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        front = prefs.getBoolean("front", true)
        v2 = prefs.getBoolean("v2", false)
        mute = prefs.getBoolean("mute", false)
        buildUi()
        status.text = "Loading models and memes ..."
        Thread { loadEngine() }.start()
    }

    // ---- UI (built in code: one screen, no layout files) ----
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        root.addView(image, FrameLayout.LayoutParams(-1, -1))
        status = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 15f; gravity = Gravity.CENTER; setPadding(dp(24), dp(24), dp(24), dp(24))
        }
        root.addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.CENTER))

        fun button(label: String, click: (Button) -> Unit) = Button(this).apply {
            text = label; textSize = 12f; isAllCaps = false; setTextColor(Color.WHITE)
            minWidth = 0; minimumWidth = 0; setPadding(dp(10), 0, dp(10), 0)
            background = GradientDrawable().apply { cornerRadius = dp(18).toFloat(); setColor(0x99202830.toInt()) }
            setOnClickListener { click(this) }
        }
        fun toggle(label: String, initial: Boolean, set: (Boolean) -> Unit): Button {
            var on = initial
            return button("") { b -> on = !on; set(on); b.text = "$label ${if (on) "on" else "off"}"; apply() }
                .also { it.text = "$label ${if (initial) "on" else "off"}" }
        }
        val b = listOf(
            toggle("Markers", markers) { markers = it },
            toggle("Memes", memes) { memes = it },
            toggle("Sound", !mute) { mute = !it; prefs.edit().putBoolean("mute", mute).apply() },
            toggle("v2 memes", v2) { v -> v2 = v; prefs.edit().putBoolean("v2", v).apply(); reload() },
            toggle("Debug", debug) { debug = it },
            button("Flip camera") { front = !front; prefs.edit().putBoolean("front", front).apply(); bindCamera() },
            button("Recalibrate") { Native.calibrate(); Toast.makeText(this, "Look straight ahead", Toast.LENGTH_SHORT).show() },
            button("Photo") { savePhoto() },
            button("Record") { toggleRecording() }.also { recButton = it },
        )
        buttons = b
        val bar = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(8), dp(8), dp(8), dp(12)) }
        for (chunk in b.chunked(4)) {
            val row = LinearLayout(this).apply { gravity = Gravity.CENTER }
            for (x in chunk) row.addView(x, LinearLayout.LayoutParams(0, dp(40), 1f).apply { setMargins(dp(3), dp(3), dp(3), dp(3)) })
            bar.addView(row, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(bar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
        buttons.forEach { it.visibility = View.INVISIBLE }
    }

    private fun apply() { Native.setOptions(markers, memes, mute); Native.setDebug(debug) }

    // ---- engine ----
    /** Copies config/, assets/ and models/ out of the APK on first start (and after an update). */
    private fun unpack(): File {
        val dst = File(filesDir, "memecam")
        val stamp = File(filesDir, "memecam.version")
        val version = packageManager.getPackageInfo(packageName, 0).lastUpdateTime.toString()
        if (stamp.exists() && stamp.readText() == version && dst.exists()) return dst
        dst.deleteRecursively()
        fun copy(path: String) {
            val kids = assets.list(path)
            if (kids.isNullOrEmpty()) {
                File(dst.parentFile, path).apply { parentFile?.mkdirs() }.outputStream().use { o -> assets.open(path).use { it.copyTo(o) } }
            } else kids.forEach { copy("$path/$it") }
        }
        copy("memecam")
        stamp.writeText(version)
        return dst
    }

    private fun loadEngine() {
        ready = false
        val err = try { Native.init(unpack().absolutePath, v2) } catch (e: Throwable) { e.toString() }
        runOnUiThread {
            if (err != null) { status.text = "Could not start:\n$err"; return@runOnUiThread }
            status.text = ""
            buttons.forEach { it.visibility = View.VISIBLE }
            apply()
            ready = true
            requestCamera()
        }
    }

    private fun reload() {
        ready = false
        status.text = "Loading ..."
        Thread { loadEngine() }.start()
    }

    // ---- camera ----
    private fun requestCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) bindCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 2) { toggleRecording(); return }
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) bindCamera()
        else status.text = "Meme Cam needs the camera.\nAllow it in Settings > Apps > Meme Cam."
    }

    private fun bindCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val use = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder()
                    .setResolutionStrategy(ResolutionStrategy(android.util.Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            val mirror = front
            use.setAnalyzer(analysis) { proxy ->
                try {
                    if (ready) {
                        val rot = proxy.imageInfo.rotationDegrees
                        val w = if (rot % 180 == 0) proxy.width else proxy.height
                        val h = if (rot % 180 == 0) proxy.height else proxy.width
                        val i = flip; flip = 1 - flip
                        var bm = bitmaps[i]
                        if (bm == null || bm.width != w || bm.height != h) { bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888); bitmaps[i] = bm }
                        val p = proxy.planes[0]
                        Native.process(p.buffer, proxy.width, proxy.height, p.rowStride, rot, mirror, bm)
                        lastShown = bm
                        recorder?.frame(bm)
                        image.post { image.setImageBitmap(bm) }
                    }
                } finally { proxy.close() }
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA, use)
        }, ContextCompat.getMainExecutor(this))
    }

    // ---- recording ----
    private fun toggleRecording() {
        recorder?.let { r ->
            recorder = null
            recButton.text = "Record"
            Thread {
                val ok = r.stop()
                runOnUiThread { Toast.makeText(this, if (ok) "Saved to Movies/MemeCam" else "Recording failed", Toast.LENGTH_SHORT).show() }
            }.start()
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED && !prefs.getBoolean("asked_mic", false)) {
            prefs.edit().putBoolean("asked_mic", true).apply()
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 2)   // the answer starts the recording
            return
        }
        val bm = lastShown ?: return
        val mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        try {
            recorder = Recorder(this, bm.width, bm.height, mic)
            recButton.text = "Stop ●"
        } catch (e: Exception) {
            Toast.makeText(this, "Can't record: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // ---- photo ----
    private fun savePhoto() {
        val bm = lastShown?.copy(Bitmap.Config.ARGB_8888, false) ?: return
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "memecam-${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MemeCam")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) { Toast.makeText(this, "Could not save", Toast.LENGTH_SHORT).show(); return }
        contentResolver.openOutputStream(uri)?.use { bm.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        Toast.makeText(this, "Saved to Pictures/MemeCam", Toast.LENGTH_SHORT).show()
    }

    override fun onPause() {
        super.onPause()
        if (recorder != null) toggleRecording()   // don't leave a recording running in the background
    }

    override fun onDestroy() {
        ready = false
        analysis.shutdown()
        Native.release()
        super.onDestroy()
    }
}
