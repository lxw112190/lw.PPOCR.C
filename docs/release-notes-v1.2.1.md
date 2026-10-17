# lw.PPOCR.C v1.2.1

C# large-image support and portable WinForms Demo / C# 大图支持与 WinForms 便携包。

## Changes / 主要变化

- C# `NativeOcr` adds a seven-argument constructor with `maxImagePixels`.
  It synchronizes managed decode and native DET source limits. Existing
  five/six-argument constructors keep the 40,000,000-pixel default.
- WinForms adds **原图上限(百万像素)** and a BGR-only memory estimate.
  Set **100** before selecting a large image to allow up to 100 million
  source pixels. Preview, repeated benchmarks and normal OCR use the same
  policy; changing it rebuilds the engine on the next initialization/run.
- Dimension errors show width, height, total pixels and selected limit.
  The independent 16,000,000-pixel text-crop limit is unchanged. No automatic
  downscale, tiling or silent crop skipping is introduced.
- A new Release asset, `lw.PPOCR.C-1.2.1-winforms-windows-x64-net48.zip`,
  contains the EXE, static-CRT native DLL, official Tiny/Small/Medium ONNX
  models, dictionaries, sample, launchers, C# source project and checksums.
  CI tests the exact extracted archive before it can be published.

```csharp
using (var ocr = new NativeOcr(detPath, clsPath, recPath, dictionaryPath,
    true, 4u, 100000000UL))
{
    var result = ocr.RecognizeFile(imagePath);
}
```

## Compatibility and limits / 兼容性与限制

C ABI v1 and WASM Host ABI v1, model/dictionary hashes, OCR goldens, REC
maximum width 960 and runtime arithmetic are unchanged. Tiny remains the sole
stable model; Small and Medium remain Preview. The current browser release
profile is unchanged: Tiny/Small compiled SIMD128 + Pointwise 2×16 + lazy
fallback; Medium/Node canonical; Legacy scalar. Global
`LW_WASM_COMPILED_REC=OFF`, fine widths OFF and FFN tiling OFF remain unchanged.
LWM v0.1 stays internal Preview.

The portable Demo needs Windows 10/11 x64 and .NET Framework 4.8 or newer
4.x. It does not require a development SDK, Python, OpenCV, ONNX Runtime or
separate VC++ Redistributable. The original repository project still targets
.NET 3.5; that compatibility target does not prove new Windows 7 validation.

The managed wrapper accepts 1..200 million pixels on x64; x86 stays capped
at 40 million. Zero is not unlimited. These are protection limits, not a
promise that RAM or GDI+ can decode all images below the cap. One 100 MP BGR
buffer alone needs about 300 MB; preview/decoded images, models and workspace
are additional. Restrict concurrent large-image requests. This does not fix
unrelated GDI+ image-save errors or change Web/HTTP/Android pixel policies.

中文：默认4000万不变，按需调高不是取消内存保护。单框仍限1600万；这轮没有修改核心
推理或改变识别结果，也未扩大稳定模型范围。新版便携ZIP内置三模型，Small/Medium仍为Preview。

## Verification and upgrade / 验证与升级

The Windows contract generates a temporary 8000×5200 (41.6 MP) image with
real text: default policy rejects it; a 50 MP policy completes full OCR with
16 detections and the expected first line. Bounds, constructor compatibility,
preview recovery, busy state and engine rebuilding are covered. Generated
large test images are not committed or included in releases.

Replace the EXE, DLL and configuration as a matching package set. C# users
must rebuild against the new wrapper to use the additional constructor; merely
replacing an old DLL does not expose new managed parameters. Verify the ZIP
sidecar and internal `SHA256SUMS.txt`, then extract the entire directory before
launching. Use the per-model launchers or select the corresponding model folder.

Publication waits for all eight reusable workflows and strict verification
of 19 primary release assets, with SHA-256/provenance retained. Stable tags
must be annotated; cryptographic tag signing remains optional. Never move
`v1.2.0` or reuse it for this update. See [the release checklist](release-readiness-v1.2.1.md).
