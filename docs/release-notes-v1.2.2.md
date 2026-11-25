# lw.PPOCR.C v1.2.2

This patch accelerates native ARM64 full OCR. 本版重点提升 ARM64 原生 OCR 速度。

- NEON adds spatial tiling for packed 1×1, regular 2×2, and stride-2 3×3
  convolution. Workers reuse packed weights and operate on disjoint output
  regions.
- ARM64 REC distributes the available thread budget across active line workers.
  Large convolutions, exact GELU, and the packed CTC projection can use
  intra-op parallelism. The depthwise 5×5 path also uses NEON.
- DB contour processing uses less scratch space, and the source tree groups
  build modules and DET postprocessing by responsibility.

## BM1684 full OCR

The table compares `88c2291` and `c097a5b`. The baseline already includes the
ARM64 thread-budget and 1×1 tiling improvements. Both source builds used
GCC 9.4 Release on the same eight-core Cortex-A53 system with official ONNX
models, CLS enabled, and adaptive REC width capped at 960. Each process
warmed up each input once before timing the BGR-to-result call. Peak RSS
covers the entire benchmark process.

| Model | Line workers | Inputs | Pairs | OCR total before → after | Peak RSS before → after |
|---|---:|---:|---:|---:|---:|
| Tiny | 1 | 4 | 3 | 6.53 → 6.30 s | 116.2 → 114.9 MiB |
| Tiny | 4 | 4 | 2 | 4.44 → 3.92 s | 186.8 → 186.7 MiB |
| Tiny | 8 (default) | 4 | 2 | 4.15 → 3.58 s | 158.0 → 163.1 MiB |
| Small | 4 | 2 | 2 | 6.68 → 5.60 s | 259.3 → 267.0 MiB |
| Small | 8 (default) | 2 | 2 | 6.71 → 5.45 s | 302.4 → 306.2 MiB |
| Medium | 4 | 1 | 1 | 26.77 → 23.19 s | 476.0 → 477.3 MiB |

Tiny used the bundled 500×500 sample, a header crop, a detail crop, and a
padded wide image. Small used the sample and header crop. Medium used the
original sample. Each pair used fresh processes, and run order alternated.
The table reports the median summed OCR latency per input set and median
process peak RSS. Tiny with one line worker varied from 2.9% slower to 4.8%
faster across pairs. Text, box, and score checksums matched in all cases.

中文：默认 8 行 workers 时，Tiny 和 Small 总耗时分别下降 13.7% 和 18.8%。
Medium 在 4 行 workers 下的原图单轮对照下降 13.3%。这些数字来自同一台 BM1684
的源码构建和同源样例，适用于复现该次对照。

## Compatibility and downloads / 兼容性与下载

The stable contract covers Tiny with C ABI v1 and WASM Host ABI v1.
Small and Medium remain Preview. LWM v0.1 is an internal Preview format.
Modern Tiny/Small browser assets use compiled SIMD128, Pointwise 2×16, and
lazy REC fallback. Medium browser and Node use canonical execution, and
Legacy uses scalar. The global `LW_WASM_COMPILED_REC` default is OFF.

The manually dispatched customer Linux workflow builds the standalone
Linux ARM64 Preview archive on Ubuntu 22.04. Check the target system's
glibc against the archive's `BUILD-ENVIRONMENT.txt` before deployment.
Android ARM64 and desktop Java/JNI Preview assets are part of the
standard tagged Release inventory. [Package guide](package.md#build-customer-linux-architecture-packages)

BM1684 source builds passed convolution reference tests, ASan/UBSan on the
new spatial kernel, shared-library compilation, and the REC-thread-budget
regression. The Windows x64 source build passed ten targeted regressions.
The tagged Release workflow validates its own artifacts before publishing.
[Release checklist](release-readiness-v1.2.2.md)
