# Direct official ONNX loading

Current source builds load the repository's official PP-OCRv6 Tiny, Small and
Medium ONNX models through the existing C ABI. LWM v0.1 loading remains
compatible. This feature does **not** imply that older release binaries accept
ONNX, nor does it turn this project into a general-purpose ONNX runtime.

## Assets and provenance

The seven distinct ONNX files were compared byte-for-byte with
lw.PPOCR.Vulkan's official model catalog: all hashes already match. No model
binary was replaced or re-exported. The authoritative hashes and dictionaries
are in [models/ppocrv6-models.json](../models/ppocrv6-models.json).

- Tiny: `models/ppocrv6-tiny/{det,rec,cls}.onnx`, `ppocr_keys.txt`.
- Small/Medium: their own `det.onnx` and `rec.onnx`, the same Tiny CLS, and
  `models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt`.
- The assets remain under their existing PaddleOCR Apache-2.0 license.

## Runtime path

```text
official .onnx -> bounded pure-C protobuf reader -> checked internal IR
legacy .lwm ------------------------------------> same IR
                                                  -> same execution backends
```

Import runs once when a model handle is loaded. No protobuf library, ONNX
Runtime, Python, NumPy, or Vulkan library is required at deployment. Import
does not execute a model to infer shapes; it uses the existing shape resolver
at representative sizes, then actual session shapes are resolved normally.
Existing canonical and compiled x64/WASM execution paths are reused.

The importer targets the reviewed PP-OCR graph subset: one batch-one FP32
NCHW RGB input, one numeric output, standard opsets 7/11/14, supported numeric
operators and bounded constant shape-control patterns. Raw BN is preserved;
it is not a byte-for-byte reproduction of the development converter's folding.
Reshape may have at most one varying output axis. Nearest Resize must use
asymmetric coordinates/floor rounding and constant scales. Arbitrary dynamic
control flow, custom domains, external tensor files, sparse tensors, local
functions and unsupported operators/attributes are rejected.

Budgets: 256 MiB per ONNX file and total constant payload, 4,096 nodes,
16,384 values, rank/control vectors up to eight, 16 attributes per node and
8 MiB of tensor names. File budget is checked before allocating its full
buffer. A smaller `lw_model_options.max_file_size` remains effective.
Partial imports are cleaned up and never published as model handles.

Import temporarily retains source bytes and normalized bytes; it is not a
zero-copy or mmap loader. Long-lived applications should reuse model/OCR
handles. On WASM, freed import allocations do not shrink linear memory.

`lw_model_get_info()` and `lwm-inspect` report the **normalized internal IR**:
format 0.1, node/tensor counts, size and checksum. The checksum is not the
source ONNX SHA-256. C ABI and WASM Host ABI layouts/exports are unchanged.

## C / command-line usage

Pass ONNX paths to `lw_model_load`, `lw_recognizer_create`,
`lw_classifier_create`, `lw_detector_create` or `lw_ocr_create`. Format is
detected by content, not the file extension.

```c
lw_model *model = NULL;
lw_error error;
lw_error_init(&error);
lw_status status = lw_model_load("models/ppocrv6-tiny/rec.onnx",
                                NULL, &model, &error);
/* Check status, create/reuse sessions, then release model. */
lw_model_free(model);
```

```powershell
.\build\Release\lw-ocr-ppm.exe models/ppocrv6-tiny/det.onnx models/ppocrv6-tiny/cls.onnx models/ppocrv6-tiny/rec.onnx models/ppocrv6-tiny/ppocr_keys.txt build/models/sample.ppm 960
```

A native runtime-only build has no conversion dependency:

```bash
cmake -S . -B build/runtime-onnx -DLW_RUNTIME_ONLY=ON -DBUILD_TESTING=OFF
cmake --build build/runtime-onnx --config Release
```

The ordinary development build still generates LWM fixtures for compatibility
and reference tests. Existing versioned LWM runtime model packs and their
schema remain unchanged.

## Deployment adapters

- HTTP, WinForms and Desktop Java model-directory constructors prefer
  `det.onnx`, `cls.onnx`, `rec.onnx` when present, otherwise use the
  corresponding LWM. A present but invalid ONNX fails explicitly; no silent
  fallback to a different model is performed.
- Native install packages include official Tiny ONNX alongside legacy LWM.
- Browser SDK/HTML and Node packages default to official ONNX via
  `LW_DEPLOY_ONNX_MODELS=ON`. Set it to `OFF` for old LWM packaging.
  This switch does **not** enable `LW_WASM_COMPILED_REC`.
- SDK packagers accept a consistent ONNX or LWM set. Worker and main-thread FS
  names both follow the selected format; public JS APIs remain unchanged.
- Node's manifest `runtime.modelFormat` and asset paths describe its actual
  files. The adapter still accepts legacy LWM assets.
- Android CI packages Tiny official ONNX directly. The existing content-based
  model cache accepts either ONNX or legacy LWM manifests; their file sets and
  asset-set IDs prevent cross-format cache reuse.

To stage a self-contained official Small model directory without conversion:

```bash
python tools/prepare_ppocrv6_runtime_variant.py --variant small \
  --model-format onnx --build-dir build --output-dir build/official-small
```

## Verification

`native_onnx_import` is registered with the ordinary native CTest suite:

```bash
ctest --test-dir build -C Release -R native_onnx_import --output-on-failure
```

It loads all seven assets; compares REC probabilities with ONNX Runtime at
widths 17/192/960 for all three variants; checks three 16-line full-text golden
contracts; and tests truncated protobuf, malformed varints, invalid arity,
external data, wrong output shape/type, unsupported operators and budgets.
ONNX/ORT are test-only dependencies. Numerical comparison allows bounded
FP32 differences from ORT BN/FMA optimizations, with an absolute probability
error cap of 1e-3; OCR full text remains an exact SHA-256 gate.

Browser SDK/HTML and Node lifecycle tests exercise the packaged ONNX assets
through the same existing golden contracts. Android AAR/APK and other platform
builds are verified by their CI workflows, not claimed from a local x64 test.

Native x64 end-to-end padding experiments and their reproducible three-model
1/4-worker A/B are documented in [finer REC widths](native-fine-rec-widths.md).
That option remains off by default and does not weaken production golden tests.
