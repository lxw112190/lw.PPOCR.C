# lw.PPOCR Android ARM64 Preview

这是一个实验性 Android Native SDK，当前只支持 arm64-v8a 和 minSdk 21。
本机不要求安装 Android Studio、Android SDK、NDK 或模拟器；GitHub Actions 会负责构建。

SDK 模块为 lw-ppocr-android，Demo 模块为 demo。generated-model-assets 是 CI 生成的模型
assets，不提交到 Git。

API 基本用法：

    val engine = LwPpocrEngine.create(context, OcrOptions(useCls = false))
    try {
        engine.setReadingOrder(ReadingOrder.HORIZONTAL_LTR)
        val result = engine.recognize(bitmap.copy(Bitmap.Config.ARGB_8888, false))
    } finally {
        engine.close()
    }

Demo 使用流程：选择图片后会先显示本地预览，不会自动开始 OCR。确认方向分类
CLS 和阅读顺序后点击“开始识别”；识别框可在预览区显示或隐藏，点击结果行
会高亮对应检测框。结果区支持复制、系统分享，以及通过系统文件选择器保存
UTF-8 TXT 或 schema_version=1 的 JSON。Demo 只使用系统 Photo Picker 和
CreateDocument，不申请相机或存储权限。

支持的阅读顺序为 `HORIZONTAL_LTR`（标准横排）、`VERTICAL_RTL`（古籍竖排，
右到左）和 `VERTICAL_LTR`（竖排，左到右）。阅读顺序可以在同一个 Engine
上运行时修改；CLS 设置改变时 Demo 会在下一次识别前重新创建 Engine。

图片通过后台加载器读取，先检查尺寸，再按约 1200 万像素上限采样，并应用
EXIF 方向后交给 OCR。预览使用 `FIT_CENTER`，检测框使用同一 ImageView
矩阵换算，因此显示框和识别坐标保持一致。换图或 Activity 销毁时会释放旧
Bitmap。

## CI 产物

Android workflow 会构建并上传以下文件：

- `lw-ppocr-android-release.aar`：`arm64-v8a`、`minSdk 21` 的 SDK AAR；
- `demo-preview.apk`：开启 R8 的优化 Preview Demo，已用本次 CI 临时密钥签名；
- `SHA256SUMS.txt`：AAR 和 APK 的 SHA-256。

`demo-debug.apk` 只在 CI 中参与构建验证，不作为下载产物。Preview APK 的
签名密钥不会持久化，因此不同 CI 运行生成的 APK 不能直接覆盖安装；更新
测试版时请先卸载旧 Preview，或使用同一签名重新构建。该 APK 不是 Play
商店生产签名版本。

AAR 可以作为本地依赖接入 Android 项目；Demo APK 可直接安装到支持
`arm64-v8a` 的设备。下载后应先按 `SHA256SUMS.txt` 校验文件，再进行安装
或集成。

应用不需要网络或存储权限。模型首次使用时会根据随 AAR 提供的
`manifest.json` 内容身份复制到 `noBackupFilesDir/lw-ppocr/models/` 下的
`ppocrv6-tiny-<asset_set_id>` 目录，并在复制过程中校验每个文件的大小和
SHA-256。后续启动只检查缓存身份、manifest 和文件大小，不会重复计算整套
模型的 SHA-256；当 AAR 中模型或字典变化时会自动安装到新的目录，成功创建
引擎后再清理该缓存根目录内的旧目录。

CI 验证 Gradle/NDK 编译、AAR/APK 内容、AArch64 ELF、依赖白名单、模型和权限。
同时验证 Release AAR、R8/签名 Preview APK、模型 manifest 和 SHA-256。真机
OCR、厂商 ROM 图片选择器、长时间内存和温度表现仍应在 ARM64 手机上单独验证。

本 Preview 明确不包含 Camera/CameraX、实时视频、PDF、批量 OCR、历史记录、
GPU/NNAPI 或网络上传功能；图片来源仅限系统 Photo Picker/文件选择器。
