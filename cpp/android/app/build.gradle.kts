import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// local.properties (not in git): sdk.dir=..., opencv.dir=<OpenCV-android-sdk>/sdk/native, tflite.host.tools=<dir with a host flatc>
val local = Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.reader()?.use { load(it) } }
val opencvNative = local.getProperty("opencv.dir") ?: error("set opencv.dir in cpp/android/local.properties (OpenCV Android SDK, sdk/native)")
val hostTools = local.getProperty("tflite.host.tools") ?: error("set tflite.host.tools in cpp/android/local.properties (directory containing a host flatc)")
val repoRoot = rootProject.projectDir.resolve("../..").canonicalFile   // the git checkout
val cppRoot = rootProject.projectDir.resolve("..").canonicalFile        // cpp/

android {
    namespace = "dev.yajatghule.memecam"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.yajatghule.memecam"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-alpha"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared", "-DOpenCV_DIR=$opencvNative/jni", "-DTFLITE_HOST_TOOLS_DIR=$hostTools",
                    "-DCMAKE_BUILD_TYPE=Release",
                )
                targets += "memecam_android"   // not every LiteRT target (benchmark tools etc.)
            }
        }
    }

    externalNativeBuild { cmake { path = file("../../CMakeLists.txt"); version = "3.31.6" } }
    ndkVersion = "27.2.12479018"

    sourceSets["main"].jniLibs.srcDir("$opencvNative/libs")   // libopencv_java4.so
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("memecam-assets"))

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")   // sideload build: installs without extra setup
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("tflite", "gif", "wav", "toml", "png") }
    packaging { jniLibs { useLegacyPackaging = false } }
}

// The app unpacks config/, assets/ (memes, plus sounds/ when present locally) and models/ on first start.
val bundle = tasks.register<Sync>("bundleMemecamAssets") {
    into(layout.buildDirectory.dir("memecam-assets/memecam"))
    from(cppRoot.resolve("config")) { into("config") }
    from(repoRoot.resolve("assets")) { into("assets") }
    from(cppRoot.resolve("models")) { into("models"); include("*.tflite") }
}
tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(bundle) }

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
}
