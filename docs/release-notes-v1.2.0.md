# lw.PPOCR.C v1.2.0

Direct official ONNX and faster Tiny/Small browser OCR / 官方 ONNX 直接加载与浏览器提速版。

## Highlights / 主要变化

- A bounded pure-C ONNX importer loads the reviewed official PP-OCRv6
  Tiny/Small/Medium DET/REC and shared CLS models through the existing API.
  Legacy LWM remains compatible. No protobuf/ONNX Runtime deployment dependency
  is added. Official models and dictionaries are not re-exported or replaced.
- Tiny/Small modern single-file HTML and Web SDK adopt compiled SIMD128,
  Pointwise 2×16, packed-constant sharing, deferred materialization and lazy
  REC fallback. Adaptive 192/320/480/640/960 widths and maximum REC width 960
  remain unchanged.
- Exact SIMD128 Erf/GELU contracts and browser UI/lifecycle/fallback/PDF
  checks cover the optimized distribution. C ABI v1 and WASM Host ABI v1
  preserve symbols, layouts, ownership and error contracts.
- Release SBOM includes the seven official ONNX assets and Small/Medium's
  shared dictionary, matched to the checked-in model catalog SHA-256 values.

## Distribution profiles / 发行配置

| Asset | Model | Execution |
|---|---|---|
| Modern HTML / Web SDK | Tiny, Small | Compiled SIMD128 + 2×16 + lazy REC fallback |
| Modern HTML / Web SDK | Medium | Canonical SIMD128, desktop-first Preview |
| Node WASM package | Tiny | Canonical SIMD128 |
| Legacy HTML / SDK | Tiny | Scalar, no PDF |
| Native packages / model packs | Tiny, Small, Medium | Existing native policies |

The global `LW_WASM_COMPILED_REC=OFF` default is unchanged. Release CI opts
in by model, tests the exact final files, then stages one flat artifact.
Medium and Node are not taken from the optimized Tiny/Small build. A separate
CI audit records configuration, sizes and SHA-256; it is not a release asset.
Fine widths (`LW_WASM_FINE_REC_WIDTHS`) and FFN tiling
(`LW_EXPERIMENTAL_REC_FFN_TILING`) remain OFF. No relaxed SIMD, fast-math,
pthread requirement, CTC rewrite or model coverage expansion is introduced.

## Evidence and limits / 验证与性能口径

See [browser A/B methodology and historical measurements](wasm-browser-benchmark.md).
These engineering comparisons support the selected Tiny/Small profile; they
are not a benchmark of tagged v1.2.0 versus v1.1.0. Do not multiply speedups
from separate experiments or claim universal phone performance. Timings and
heap/RSS remain informational on hosted runners; text, fallback, lifecycle
and artifact identity are gates.

The importer supports the documented batch-one FP32 PP-OCR subset, not
arbitrary ONNX graphs. External weights, custom domains and unsupported
structures fail explicitly. Import temporarily retains source and normalized
IR; WASM memory does not shrink merely because buffers are freed. See
[ONNX limits and provenance](onnx-runtime.md).

Tiny is the sole stable model. Small and Medium remain Preview, as do Android,
Desktop Java/JNI and optional architecture packages. Medium is not promoted
for mobile. LWM v0.1 stays internal Preview; model minimum-runtime metadata
remains independent of product version.

## Upgrade / 升级

Replace binaries, SDKs and assets as a matching release set and verify SHA-256.
v1.1.0 binaries do not gain ONNX import by swapping filenames. High-level ABI
v1 consumers retain their contract; rebuild and smoke-test extracted packages
when upgrading. Low-level model/session APIs remain experimental. Android
demo versionCode advances from 3 to 4 and retains its Preview label.

The workflow still verifies 18 primary assets and emits provenance attestations.
Create a new annotated `v1.2.0` tag; signing is optional. Never move published
tags. Follow the [v1.2 checklist](release-readiness-v1.2.md) before tagging.
