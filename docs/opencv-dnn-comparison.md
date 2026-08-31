# OpenCV DNN full-OCR comparison

This repository includes a reproducible, correctness-gated comparison against
the official [`lw.PPOCR.OpenCVDNN`](https://github.com/lxw112190/lw.PPOCR.OpenCVDNN)
HTTP service. The script launches both packaged services and measures the same
single-image full-OCR workload. The OpenCV project is a comparison input, not a
build or runtime dependency of the pure-C library.

## Comparison contract

The comparison is deliberately narrower than a general product benchmark:

- both services receive the exact same uncompressed 24-bit BMP bytes over
  HTTP keep-alive on loopback;
- the source image and the DET, CLS, REC, and dictionary SHA-256 values must
  match the checked-in full-OCR Golden contract on both sides; pure-C runtime
  LWM and dictionary hashes are also recorded;
- DET uses a 960-pixel limit, threshold 0.3, box threshold 0.6, unclip ratio
  1.6, no dilation, and direction classification with threshold 0.9;
- the pure-C service uses four line workers and a 960-pixel maximum adaptive
  REC width; the OpenCV package uses its published one-engine, four HTTP
  worker, REC batch-size-eight configuration;
- both processes may be pinned to the same logical processors;
- warm-up is excluded, measured rounds alternate which engine runs first, and
  both client-observed and server-reported latency are retained;
- a performance result is rejected unless line count and NFKC/no-whitespace
  text match and every corresponding box boundary is within 5 pixels.

Quadrilateral start-point ordering is not part of the gate. The comparison uses
each box's left, top, right, and bottom boundaries, matching the semantics of
the existing Golden corpus.

The script does not print recognized text. It records text checksums and
mismatch indexes, plus executable/model/input hashes, configuration, latency,
throughput, and observed working set in a versioned JSON report.

The LWM converter itself only accepts the three source ONNX hashes named by the
Golden contract. The benchmark verifies those source files independently and
records the resulting LWM hashes, without treating the evolving LWM 0.1 bytes
as a frozen public-format contract.

## Windows example

Build the pure-C x64 HTTP Demo and extract or build an x64
`lw.PPOCR.OpenCVDNN` package. From this repository root:

```powershell
python .\tools\compare_opencv_dnn.py `
  --c-server .\build-ninja-c\bin\lw.PPOCR.C.HttpServer.exe `
  --c-models .\build-ninja-c\models `
  --c-www .\build-ninja-c\www `
  --opencv-package ..\lw.PPOCR.OpenCVDNN\dist\lw.PPOCR.OpenCVDNN-v1.0.0-windows-x64 `
  --image .\build-ninja-c\models\sample.ppm `
  --source-image .\models\ppocrv6-tiny\sample.jpg `
  --c-source-models .\models\ppocrv6-tiny `
  --contract .\tests\fixtures\ocr-golden-corpus.json `
  --c-workers 4 --c-rec-max-width 960 `
  --cpu-affinity 0-7 `
  --warmup 10 --rounds 5 --iterations-per-round 20 `
  --output .\build-ninja-c\perf-comparison\opencv-dnn.json
```

`--cpu-affinity` is optional. Choose valid logical processor indexes for the
host and keep the same mask when comparing commits. The report is written
under the ignored build directory by design; publish it only with the machine,
OS, package version, commit, power-mode, and command used to obtain it.

## Local repeat measurements

On 2026-08-31, Windows 10 x64 with 16 logical processors, processors 0-7
pinned, the bundled 500 x 500 sample, OpenCV DNN package 1.0.0, 10 warm-ups,
and 100 measured sequential requests per engine produced three independent
runs with the final binaries:

| Run | pure-C mean | OpenCV mean | pure-C mean-latency reduction |
|---|---:|---:|---:|
| 1 | 188.453 ms | 184.588 ms | -2.094% |
| 2 | 195.923 ms | 199.267 ms | +1.678% |
| 3 | 187.517 ms | 187.407 ms | -0.058% |

The mean of the three run means was 190.631 ms for pure C and 190.421 ms for
OpenCV DNN. The result crosses the zero-speedup boundary, so this sample does
not establish a stable full-OCR speed advantage for either engine. The maximum
sampled working-set ranges were 130.19-131.68 MiB for pure C and
149.04-149.32 MiB for OpenCV DNN.

All 16 texts matched exactly in every run. All 16 box boundaries passed the
5-pixel gate; the maximum absolute boundary delta was 0.732 pixel.

## Interpreting the result

The sequential-request throughput is `1000 / mean client latency`; it is not
a concurrent saturation test. Server timing excludes response JSON
serialization in both services, while client timing includes HTTP transfer and
JSON parsing. Working set is sampled after warm-up and after each measured
round, so it is an observed value rather than an operating-system peak counter.

A result from one image and one host answers only whether these two builds are
faster under this contract. It does not establish a universal advantage across
images, CPUs, concurrency levels, or OpenCV builds. Extend the Golden corpus
and repeat the same correctness gate before making a broader claim. Run at
least three independent trials when the measured difference is small.
