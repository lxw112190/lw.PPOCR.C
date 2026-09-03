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

Demo 使用流程：选择图片后会先显示本地预览，不会自动开始 OCR。确认方向分类
CLS 和阅读顺序后点击“开始识别”；识别框可在预览区显示或隐藏，点击结果行
会高亮对应检测框。结果区支持复制、系统分享，以及通过系统文件选择器保存
UTF-8 TXT 或 schema_version=1 的 JSON。Demo 只使用系统 Photo Picker 和
CreateDocument，不申请相机或存储权限。

应用不需要网络或存储权限。模型首次使用时会根据随 AAR 提供的
`manifest.json` 内容身份复制到 `noBackupFilesDir/lw-ppocr/models/` 下的
`ppocrv6-tiny-<asset_set_id>` 目录，并在复制过程中校验每个文件的大小和
SHA-256。后续启动只检查缓存身份、manifest 和文件大小，不会重复计算整套
模型的 SHA-256；当 AAR 中模型或字典变化时会自动安装到新的目录，成功创建
引擎后再清理该缓存根目录内的旧目录。

CI 验证 Gradle/NDK 编译、AAR/APK 内容、AArch64 ELF、依赖白名单、模型和权限。
真机 OCR、厂商 ROM 图片选择器、长时间内存和温度表现需要在 ARM64 手机上单独验证。
