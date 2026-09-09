# PP-OCRv6 Runtime Model Packs

`v0.2.0-preview.1` introduces a model-pack boundary without creating a separate Runtime binary for every model. A pack contains the converted LWM assets required by one PP-OCRv6 variant and a manifest that identifies the exact asset set.

The pack layout is self-contained and namespaced by variant:

```text
ppocrv6-small/
  manifest.json
  SHA256SUMS
  det.lwm
  cls.lwm
  rec.lwm
  ppocr_keys.txt
```

The manifest and checksum paths inside the pack are relative to that variant directory. Extracting Tiny, Small and Medium packs into one `models` directory therefore creates `models/ppocrv6-tiny`, `models/ppocrv6-small` and `models/ppocrv6-medium` without overwriting assets.

`asset_set_id` is derived from the variant, runtime revision and all four asset hashes. Applications can use it as a cache-directory revision, so a model upgrade cannot silently reuse an older set of files.

The packager consumes the exact converted directory that was tested by CI:

```powershell
python tools/prepare_ppocrv6_runtime_variant.py `
  --variant small `
  --build-dir build `
  --output-dir build/runtime-models/ppocrv6-small

python tools/package_ppocrv6_runtime.py `
  --input-dir build/runtime-models/ppocrv6-small `
  --variant small `
  --output dist/lw-ppocr-model-ppocrv6-small-0.2.0-preview.1.zip

python tools/validate_runtime_model_pack.py `
  dist/lw-ppocr-model-ppocrv6-small-0.2.0-preview.1.zip
```

The input directory must contain `det.lwm`, `cls.lwm`, `rec.lwm` and `ppocr_keys.txt`. Use `tools/prepare_ppocrv6_runtime_variant.py` to produce the canonical directory from a native build before packaging. The same command works for `tiny`, `small` and `medium`. The ZIP is deterministic and stored without compression so its checksum is stable across CI runners.

Manifest version fields are part of the pack contract:

- `model_revision` must use `MAJOR.MINOR.PATCH` with an optional prerelease suffix. A leading `v` is accepted by the packager and normalized away.
- `runtime_status` is derived from `model_revision`: revisions containing `-` are `preview`; stable revisions are `production`.
- `minimum_runtime_version` is an independent compatibility floor and defaults to `0.2.0`.
- `lwm_format_version` uses the `major.minor` form.
- `recommended.rec.adaptive_width` and `recommended.rec.max_width` describe the tested REC policy.

For a production revision, pass an explicit minimum runtime when needed:

```powershell
python tools/prepare_ppocrv6_runtime_variant.py `
  --variant tiny `
  --build-dir build `
  --output-dir build/runtime-models/ppocrv6-tiny

python tools/package_ppocrv6_runtime.py `
  --input-dir build/runtime-models/ppocrv6-tiny `
  --variant tiny `
  --runtime-version 0.2.0 `
  --minimum-runtime-version 0.2.0 `
  --output dist/lw-ppocr-model-ppocrv6-tiny-0.2.0.zip
```

Tiny remains the default model in the C/HTTP, Web, Android, Desktop Java and Node/WASM packages. Small and Medium are opt-in preview assets and do not replace those defaults. On a tagged release, the workflow publishes `lw.PPOCR.C-<version>-ppocrv6-tiny-runtime.zip`, `lw.PPOCR.C-<version>-ppocrv6-small-runtime.zip` and `lw.PPOCR.C-<version>-ppocrv6-medium-runtime.zip` only after the Windows and Linux validation packs have identical SHA-256 values.

The same tagged release also publishes separately named Small and Medium browser SDK/HTML files. Those browser artifacts pass variant-specific full-text golden OCR and engine lifecycle tests before publication. They are self-contained model distributions; do not combine an SDK file from one variant with the filename or expectations of another. PDF regression and the default mobile-browser recommendation remain Tiny-only.
