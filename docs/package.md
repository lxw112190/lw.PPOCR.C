# Development package

The `0.1.x` archives are experimental development packages, not ABI-frozen 1.0
releases. Windows x64 and x86 archives are separate and must not be mixed.

## Contents

```text
bin/                         PPM demo, benchmark, shared runtime, Windows CRT
include/lw_infer.h           public C header
lib/                         static library and shared-library import library
lib/cmake/lw.PPOCR.C/        CMake package configuration
examples/                    standalone REC/CLS/DET/full-OCR CMake consumers
models/rec.lwm               converted PP-OCRv6 tiny REC model
models/cls.lwm               converted PP-OCRv6 tiny CLS model
models/det.lwm               converted DET model for the public detector
models/ppocr_keys.txt        UTF-8 recognition dictionary
models/sample-crop.ppm       dependency-free demo input
models/sample.ppm            full-image DET demo input
docs/                        API and implementation documentation
LICENSE
README.md
THIRD-PARTY-NOTICES.md
dependencies.lock.json
sbom.cdx.json
```

## Run the packaged demo

From an extracted Windows archive:

```powershell
.\bin\lw-recognize-ppm.exe `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample-crop.ppm
```

Expected text:

```text
text=纯臻营养护发素
```

The demo intentionally supports only binary P6 PPM. This keeps it pure C and
dependency-free while demonstrating the complete public recognizer API. The
core library accepts decoded BGR8 pixels; production applications may use their
own JPEG/PNG decoder.

Run the direction-classification demo against the same decoded crop:

```powershell
.\bin\lw-classify-ppm.exe `
  .\models\cls.lwm `
  .\models\sample-crop.ppm
```

It reports label `0`/`1`, orientation `0`/`180`, Softmax score, and resized
content width. It reports orientation but does not rotate the image.

Run the text-detection Demo against the full-image fixture:

```powershell
.\bin\lw-detect-ppm.exe `
  .\models\det.lwm `
  .\models\sample.ppm
```

It prints clockwise quadrilateral coordinates in the original image coordinate
system and one score per detected region. The package smoke test executes this
installed binary and requires a non-empty result.

Run the complete OCR Demo against the same full-image fixture:

```powershell
.\bin\lw-ocr-ppm.exe `
  .\models\det.lwm `
  .\models\cls.lwm `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample.ppm
```

It runs DET, pure-C perspective crop, CLS direction correction, and REC, then
prints UTF-8 text, all stage scores, applied rotation, and one original-image
quadrilateral per line. The staged-package smoke test builds the installed
consumer and requires the expected sample title from this installed binary.

## Run the scalar benchmark

The packaged benchmark reuses one recognizer and emits machine-readable JSON:

```powershell
.\bin\lw-rec-benchmark.exe `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample-crop.ppm `
  3 20
```

The last two values select warm-up and measured iterations. Latency and RSS are
environment-dependent measurements, not release-wide performance guarantees.

## Consume with CMake

Point `CMAKE_PREFIX_PATH` at the extracted package, then use either target:

```cmake
find_package(lw.PPOCR.C CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE lw.PPOCR.C::shared)
# or: lw.PPOCR.C::static
```

Consumers of `lw.PPOCR.C::shared` receive the required DLL import definition
through the imported target. Copy the DLL beside the application on Windows.

## Build a local archive

After configuring and building a Release tree:

```powershell
cpack --config .\build\CPackConfig.cmake -C Release -G ZIP
```

CPack writes the archive and its `.sha256` checksum under `build/packages`.
The package test installs to a clean staging directory, builds consumers using
the installed CMake package, and runs the installed Demos against installed
models, dictionary, and fixtures before packaging.
