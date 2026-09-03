plugins {
    id("com.android.library")
    kotlin("android")
}

android {
    namespace = "com.lxw112190.ppocr"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        minSdk = 21
        targetSdk = 35
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c11", "-fvisibility=hidden")
                arguments += listOf(
                    "-DLW_BUILD_HTTP_DEMO=OFF",
                    "-DLW_BUILD_CSHARP_DEMOS=OFF",
                    "-DLW_RUNTIME_ONLY=ON",
                    "-DBUILD_TESTING=OFF"
                )
            }
        }
        consumerProguardFiles("consumer-rules.pro")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(rootProject.file("generated-model-assets"))
        }
    }

    buildFeatures {
        buildConfig = false
    }
}

dependencies {
    implementation("androidx.annotation:annotation:1.9.1")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
