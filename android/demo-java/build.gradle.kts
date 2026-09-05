plugins {
    id("com.android.application")
}

android {
    namespace = "com.lxw112190.ppocr.javademo"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.lxw112190.ppocr.javademo"
        minSdk = 21
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-preview"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
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
    implementation("androidx.activity:activity:1.10.1")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
}
