# Java/JVM JNI OCR 示例

这是一个非常小的桌面 Java 消费者示例，面向已经安装的
`lw.PPOCR.C` 开发包。支持 Java 8+、Windows x64 和 Linux x64，图片解码
直接使用标准库 `ImageIO`，JNI 只调用现有 `lw_ocr_*` C ABI，不修改 Runtime、
LWM 格式或公共 C 头文件。

本示例不是 Android SDK、Maven 构件，也不会自动下载或加载 native 文件。
请把 `lw_ppocr_java` 和匹配的 `lw_ppocr_c` 动态库放在同一个目录，并通过
`-Djava.library.path` 指定该目录。

## 使用安装开发包构建

假设开发包前缀为 `/opt/lw.PPOCR.C`：

```bash
cmake -S examples/java-jni -B build-java-jni -G Ninja \
  -DCMAKE_PREFIX_PATH=/opt/lw.PPOCR.C \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-java-jni

javac -encoding UTF-8 -d build-java-classes \
  examples/java-jni/java/NativeOcr.java \
  examples/java-jni/java/OcrDemo.java

java -Djava.library.path=build-java-jni \
  -cp build-java-classes OcrDemo \
  /opt/lw.PPOCR.C/models \
  /opt/lw.PPOCR.C/models/sample.jpg
```

Windows 将 `CMAKE_PREFIX_PATH` 改为 `C:/lw.PPOCR.C` 即可。CMake 构建完成后
会自动把匹配的核心动态库和 Windows 运行时 DLL 复制到 JNI 输出目录。干净的
PowerShell 中还应同时设置 `PATH` 和 JVM library path，因为 Windows 加载器会
通过 `PATH` 查找 JNI 依赖：

```powershell
$native = (Resolve-Path .\build-java-jni\Release).Path
$env:Path = "$native;$env:Path"
java "-Djava.library.path=$native" -cp .\build-java-classes OcrDemo `
  C:\lw.PPOCR.C\models C:\lw.PPOCR.C\models\sample.jpg
```

Linux 会设置 `$ORIGIN` RPATH。

## 下载 CI 验证的 bundle

`Desktop Java JNI OCR` GitHub Actions workflow 支持手动运行，也会在影响
本示例的代码变更时运行。它会生成两个临时 Actions artifact：
`lw-ppocr-java-jni-windows-x64` 和 `lw-ppocr-java-jni-linux-x64`。其中包含
native 依赖闭包、Java 源码、模型、许可证、构建信息以及
`SHA256SUMS.txt`。这些是 CI 下载包，不是 tagged Release 资产，也不代表
稳定 ABI 承诺。

解压后在 bundle 根目录编译并运行示例：

```powershell
javac -encoding UTF-8 -d classes java\NativeOcr.java java\OcrDemo.java
$env:Path = "$(Resolve-Path .\native);$env:Path"
java "-Djava.library.path=$(Resolve-Path .\native)" -cp classes OcrDemo `
  "$(Resolve-Path .\models)" "$(Resolve-Path .\models\sample.jpg)"
```

Linux 下：

```bash
javac -encoding UTF-8 -d classes java/NativeOcr.java java/OcrDemo.java
java -Djava.library.path="$PWD/native" -cp classes OcrDemo \
  "$PWD/models" "$PWD/models/sample.jpg"
```

## Java API

```java
try (NativeOcr ocr = new NativeOcr("models", false, 0)) {
    String[] lines = ocr.recognizeFile("models/sample.jpg");
    ocr.setReadingOrder(NativeOcr.READING_ORDER_HORIZONTAL_LTR);
}
```

`workerCount = 0` 使用 Runtime 默认值。`recognizeFile` 只返回按阅读顺序排列
的文字，坐标和置信度留给后续完整绑定。一个 `NativeOcr` 实例内部串行化调用；
`close()` 可重复调用，关闭后识别会抛出 `IllegalStateException`。

JNI 会把 Java UTF-16 路径转换成标准 UTF-8，并手动把 OCR 的 UTF-8 转成 Java
字符串，不使用 `NewStringUTF`，因此中文路径和扩展 Unicode 不会被误当成
Modified UTF-8。

本示例暂不包含 Swing/JavaFX UI、Android、Maven 发布、native 自动下载、PDF、
批量 OCR 或完整坐标/分数对象模型。
