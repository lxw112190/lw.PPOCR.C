# lw.PPOCR Android ARM64 Preview

这是一个实验性 Android Native SDK，当前只支持 arm64-v8a 和 minSdk 21。
本机不要求安装 Android Studio、Android SDK、NDK 或模拟器；GitHub Actions 会负责构建。

SDK 模块为 lw-ppocr-android，Demo 模块为 demo。generated-model-assets 是 CI 生成的模型
assets，不提交到 Git。

API 基本用法：

    val engine = LwPpocrEngine.create(context, OcrOptions(useCls = false))
    try {
        val result = engine.recognize(bitmap.copy(Bitmap.Config.ARGB_8888, false))
    } finally {
        engine.close()
    }

应用不需要网络或存储权限。模型首次使用时复制到应用的 noBackupFilesDir，
后续复用已安装的模型文件。

CI 验证 Gradle/NDK 编译、AAR/APK 内容、AArch64 ELF、依赖白名单、模型和权限。
真机 OCR、厂商 ROM 图片选择器、长时间内存和温度表现需要在 ARM64 手机上单独验证。
