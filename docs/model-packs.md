# PP-OCRv6 Runtime Model Packs

`v0.2.0-preview.1` introduces a model-pack boundary without creating a separate Runtime binary for every model. A pack contains the converted LWM assets required by one PP-OCRv6 variant and a manifest that identifies the exact asset set.

The pack layout is self-contained:

```text
manifest.json
SHA256SUMS
det.lwm
cls.lwm
rec.lwm
ppocr_keys.txt
```

`asset_set_id` is derived from the variant, runtime revision and all four asset hashes. Applications can use it as a cache-directory revision, so a model upgrade cannot silently reuse an older set of files.

The packager consumes the exact converted directory that was tested by CI:

```powershell
python tools/package_ppocrv6_runtime.py `
  --input-dir build-model-pack/models `
  --variant small `
  --output dist/lw-ppocr-model-ppocrv6-small-0.2.0-preview.1.zip

python tools/validate_runtime_model_pack.py `
  dist/lw-ppocr-model-ppocrv6-small-0.2.0-preview.1.zip
```

The input directory must contain `det.lwm`, `cls.lwm`, `rec.lwm` and `ppocr_keys.txt`. The same command works for `tiny`, `small` and `medium`. The ZIP is deterministic and stored without compression so its checksum is stable across CI runners.

The first preview keeps Tiny as the default model in existing C, HTTP, Web, Android and Java packages. Small and Medium packs are opt-in native model assets until their production conversion and platform-specific validation gates are complete. On a tagged release, the Release workflow publishes `lw.PPOCR.C-<version>-ppocrv6-small-runtime.zip` and `lw.PPOCR.C-<version>-ppocrv6-medium-runtime.zip` only after the Windows and Linux validation packs have identical SHA-256 values.