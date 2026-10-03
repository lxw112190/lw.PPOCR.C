# v1.2.0 release readiness / 发布准备清单

Status: local preparation; do not tag until the committed candidate passes CI.
范围不变：Tiny 稳定，Small/Medium 与 Android/Java/可选架构仍为 Preview。

## Prepared in source / 源码收口

- CMake/product metadata, C# assembly, Android demos, model-pack tools and
  SBOM share the 1.2.0 base; Android versionCode is 4.
- ONNX import preserves high-level C/WASM ABI v1 and LWM loading. Reviewed
  official model hashes and golden contracts are unchanged.
- Tiny/Small modern Web use compiled SIMD128 + Pointwise 2×16 + lazy fallback.
  Medium/Node remain canonical and Legacy scalar. Global compiled REC, fine
  widths and experimental FFN defaults are not promoted.
- CI tests final Tiny SDK/HTML including fallback/PDF and Small SDK/HTML
  text/lifecycle. Canonical/scalar A/B checks remain. Stage selection rejects
  wrong configuration, missing/stale archives and bad hashes.

The previously green `ed175a6cbc722d6149cbee656632b3c4ca9cfbad` baseline does
not prove that this preparation commit or the new tag is green.

## Local verification, 2026-10-03 / 本地验证

- 40 version, release inventory, model-pack, profile selection and compiled
  diagnostic unit tests passed; stable metadata check returned no blockers.
- 10 targeted CTest checks passed: ONNX comparison/import/deployment, source
  newline, C/WASM ABI and release metadata. Existing native drivers were reused;
  exact-candidate native builds still require CI.
- Fresh Emscripten 4.0.15 release builds produced compiled Tiny/Small and
  canonical Medium/Node. Actual SDK versions were verified as 1.2.0.
- Pointwise: 792 bitwise cases; Erf/GELU: 65,536 vector blocks plus alias/tail
  cases passed. Tiny/Small/Medium SDK and HTML matched their unchanged goldens.
- Tiny browser fallback and offline two-page PDF OCR passed with zero network
  requests. PDF checks now distinguish successful compiled status messages
  from actual errors using the existing strict diagnostics matcher.
- Real artifact staging preserved selected SHA-256 values. The selected Node
  ZIP was extracted and its lifecycle/full-text smoke passed on Node 22.
- YAML parsing and `git diff --check` passed. Generic skill preflight's OpenCV
  version rejection is N/A; its generated-cache warning was checked with
  `git check-ignore` and those caches remain ignored.

These are local checks, not Android/device validation or exact-tag CI evidence.

## Before tagging / 打 tag 前

- [ ] Commit/push changes and record the exact candidate SHA.
- [ ] Native Windows/Linux, sanitizers, backend correctness, Android,
  Java/JNI, Tiny/Small/Medium validation and browser WASM pass that SHA.
- [ ] Inspect `web-release-profile-<SHA>`: compiled Tiny/Small, canonical
  Medium/Node. If main CI omitted Medium, run the full-variant profile:

  ```bash
  gh workflow run wasm-html.yml --ref main -f build_web_variants=true -f release_web_profile=true
  ```

  Verify head SHA equals the candidate. Do not combine `compiled_rec=true`
  with `release_web_profile=true`; the latter needs a canonical primary build.
- [ ] Download exact candidate Tiny/Small HTMLs; test offline on desktop and
  intended phones: select/clear/reselect, paste, PDF, export, CLS and reading
  order. Previous phone evidence does not replace this artifact test.
- [ ] Run metadata/profile checks and inspect the final diff.
- [ ] Confirm `v1.2.0` is absent locally/remotely, then tag the tested SHA.

```bash
python -m unittest tests.test_versioning tests.test_release_assets tests.test_runtime_model_pack tests.test_web_release_profile -v
python tools/check_release_readiness.py --mode stable --version 1.2.0
git diff --check
git rev-parse HEAD
git tag -a v1.2.0 -m "lw.PPOCR.C v1.2.0: direct ONNX and faster Tiny/Small browser OCR"
git show --no-patch v1.2.0
git push origin v1.2.0
```

## Exact tag / tag 流水线

All seven reusable jobs must pass the exact tag; publication waits for them.
Verify the strict 18-asset inventory, archive/Android checksums and provenance,
then smoke-test extracted packages. Signing remains optional. Audit/benchmark
JSONs stay CI-only, outside release-assets. Never move an existing tag.
See [package checks](package.md) and [release notes](release-notes-v1.2.0.md).

OpenCV-specific generic release-preflight checks are not applicable to this
dependency-free C runtime; runtime ABI, model formats, deployment and actual
archive verification are the project-specific gates.
