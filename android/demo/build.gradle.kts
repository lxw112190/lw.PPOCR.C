plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.lxw112190.ppocr.demo"
    compileSdk = 35
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "com.lxw112190.ppocr.demo"
        minSdk = 21
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-preview"
        ndk { abiFilters += "arm64-v8a" }
    }
    buildTypes {
        release {
            // Keep JNI entry points and packet types through the release shrinker.
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
}

dependencies {
    implementation(project(":lw-ppocr-android"))
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}
