# Java/JVM JNI OCR example

This is a deliberately small desktop Java consumer of the installed
`lw.PPOCR.C` package. It supports Java 8+, Windows x64 and Linux x64, and
uses `ImageIO` for JPEG/PNG/BMP decoding. The JNI layer calls the existing
`lw_ocr_*` C ABI; it does not change the runtime, LWM format, or public C
header.

This example is not an Android SDK, Maven artifact, or automatic native-loader
library. Put `lw_ppocr_java` and the matching `lw_ppocr_c` shared library in
the same native directory and pass that directory through
`-Djava.library.path`.

## Build against an installed package

From a development package containing `lib/cmake/lw.PPOCR.C`:

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

On Windows, use `-DCMAKE_PREFIX_PATH=C:/lw.PPOCR.C` and replace the path
separators as needed. The CMake post-build step copies the matching core
shared library and Windows runtime DLLs beside the JNI library. In a clean
PowerShell session, put that directory on both the JVM library path and
`PATH`, because the Windows loader resolves dependent DLLs through `PATH`:

```powershell
$native = (Resolve-Path .\build-java-jni\Release).Path
$env:Path = "$native;$env:Path"
java "-Djava.library.path=$native" -cp .\build-java-classes OcrDemo `
  C:\lw.PPOCR.C\models C:\lw.PPOCR.C\models\sample.jpg
```

Linux embeds an `$ORIGIN` RPATH.

## Download a CI-tested bundle

The `Desktop Java JNI OCR` GitHub Actions workflow can be dispatched manually
or runs for changes affecting this example. It publishes two temporary Actions
artifacts: `lw-ppocr-java-jni-windows-x64` and
`lw-ppocr-java-jni-linux-x64`. They contain the native dependency closure,
Java sources, models, licenses, build provenance, and `SHA256SUMS.txt`.
These artifacts are CI downloads, not tagged Release assets or a stable ABI
promise.

After extracting a bundle, compile and run the included example from its root:

```powershell
javac -encoding UTF-8 -d classes java\NativeOcr.java java\OcrDemo.java
$env:Path = "$(Resolve-Path .\native);$env:Path"
java "-Djava.library.path=$(Resolve-Path .\native)" -cp classes OcrDemo `
  "$(Resolve-Path .\models)" "$(Resolve-Path .\models\sample.jpg)"
```

On Linux:

```bash
javac -encoding UTF-8 -d classes java/NativeOcr.java java/OcrDemo.java
java -Djava.library.path="$PWD/native" -cp classes OcrDemo \
  "$PWD/models" "$PWD/models/sample.jpg"
```

## API

```java
try (NativeOcr ocr = new NativeOcr("models", false, 0)) {
    String[] lines = ocr.recognizeFile("models/sample.jpg");
    ocr.setReadingOrder(NativeOcr.READING_ORDER_HORIZONTAL_LTR);
}
```

`workerCount = 0` selects the runtime default. `recognizeFile` returns ordered
text only; coordinates and scores are intentionally left to a future binding.
One `NativeOcr` instance serializes its operations. Calling `close()` more than
once is safe, while recognition after close throws `IllegalStateException`.

The JNI bridge converts Java UTF-16 paths to standard UTF-8 and decodes OCR
UTF-8 output without `NewStringUTF`, so non-ASCII paths and supplementary
Unicode are not silently treated as Modified UTF-8.

The example intentionally does not provide Swing/JavaFX UI, Android support,
Maven publishing, native auto-download, PDF, batch OCR, or a complete result
object model.
