# v1.2.1 release readiness / 发布准备清单

Status: source preparation. v1.2.0 remains the latest published release until
the new tag's build/publish workflow succeeds.

## Scope / 范围

- C# configurable large-image policy and WinForms controls from `8eec673`.
- Dedicated Windows x64 .NET 4.8 portable Demo with Tiny/Small/Medium, static
  CRT, source, licenses and SHA-256. Original .NET 3.5 project is retained.
- Product metadata uses 1.2.1; Android Preview demos advance to versionCode 5.
- Native/WASM ABI v1, core arithmetic, models, dictionaries, goldens and browser
  backend selection remain unchanged. Tiny stable; Small/Medium Preview.
- Release waits for eight reusable workflows. Strict primary asset inventory
  increases from 18 to 19 solely to add the verified WinForms ZIP.

## Gates / 操作前检查

- [ ] Commit and push these release-preparation changes; record the exact SHA.
- [ ] All required workflows pass that same SHA, including the new Windows
  WinForms portable Demo workflow. The earlier green `8eec673` is not evidence
  that this new version/package configuration has passed.
- [ ] Inspect/download the exact WinForms artifact; verify hash and test the
  customer's original image with the intended cap on their x64 machine.
  A successful synthetic large image is not a universal GDI+/memory guarantee.
- [ ] Check all 19 planned asset names and checksums. Final Release inventory
  stays strict; no PDB, report, helper EXE or generated test image is published.
- [ ] Confirm `v1.2.1` does not exist locally or remotely. Create a new annotated
  tag on the reviewed, CI-green commit; do not move an existing tag.

```powershell
python -m unittest tests.test_versioning tests.test_release_assets tests.test_runtime_model_pack tests.test_web_release_profile -v
python tools/check_release_readiness.py --mode stable --version 1.2.1
git diff --check
git rev-parse HEAD
git tag --list v1.2.1
git ls-remote --tags origin refs/tags/v1.2.1
git tag -a v1.2.1 -m "lw.PPOCR.C v1.2.1: C# large-image support and portable WinForms Demo"
git show --no-patch v1.2.1
git push origin v1.2.1
```

The tag starts a full rebuild. The publish job validates all exact-tag assets
before exposing the draft. Tag signing is optional, not a new blocker.

## Local checks and limitations / 本地验证口径

During release preparation: 43 version/asset/profile tests, six targeted CTest
checks (including the 41.6 MP C# contract), .NET 3.5 reference compilation and
the freshly built v1.2.1 exact ZIP smoke passed locally. All three packaged
models matched their existing full-text checksums at 1/4 workers; 36 internal
file hashes and the ZIP checksum/CRC passed. This is not a claim that the new
workflow or exact tag has already passed remote CI.

Run version/asset contracts, the real C# large-image integration and the exact
WinForms ZIP smoke before handing off commands. The ZIP smoke verifies each
internal hash, constructs the packaged Form, and checks complete OCR text for
all three models at 1/4 workers with a minimal PATH. It does not substitute
for exact-tag CI, phone tests or new customer hardware claims.

Generic skill preflight's OpenCV-version requirement is N/A for the pure-C
core; web/service assets are N/A for the WinForms-only ZIP. Relevant gates are
native dependency closure, managed/runtime versions, model hashes, actual
archive execution and strict release inventory. Ignored build caches are not
release inputs.
