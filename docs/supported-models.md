# Supported models

## Production-supported

| Family | Variant | DET | CLS | REC | Full OCR |
|---|---|---|---|---|---|
| PP-OCRv6 | Tiny | ✅ | ✅ | ✅ | ✅ |

## Preview runtime model packs

The `v0.2.0-preview.1` model-pack format packages the exact converted LWM
assets tested by CI. It is shared by Tiny, Small and Medium and does not
create a separate Runtime binary per model. Each pack contains `manifest.json`,
`SHA256SUMS`, `det.lwm`, `cls.lwm`, `rec.lwm` and its dictionary; the manifest
includes an `asset_set_id` for cache invalidation.

Small and Medium remain preview/opt-in until their production conversion,
quality, resource and platform gates are complete. Existing default C, HTTP,
Web, Android and Java packages continue to use Tiny.

See [`docs/model-packs.md`](model-packs.md) for the packaging and validation
commands.
## Experimental model support

| Family | Variant | ONNX analysis | Experimental LWM | Full OCR | Release |
|---|---|---|---|---|---|
| PP-OCRv6 | Small | ✅ | ✅ fixed + dynamic prototypes | ✅ experimental | model archive only |
| PP-OCRv6 | Medium | ✅ | ✅ fixed + dynamic analysis prototypes | ✅ analysis gate | model archive only |

Experimental means that ONNX checker, shape inference, operator inventory,
fixed-width and narrow dynamic LWM conversion, graph-output comparison, and
(for Small) a dependency-free full-OCR pipeline have been validated when the
external model assets are supplied. Medium currently has analysis-only
fixed-shape/fixed-width and narrow dynamic conversion checkpoints, plus one
complete sample-image composition check and a scheduled Windows/Linux analysis
gate at REC width 960; it has no production package. This is not a
production support claim and does not add either variant to the default model
package, C ABI, Android, WASM, or the default runtime release. The dedicated
model archive contains the checked ONNX inputs for reproducible analysis only.

The authoritative asset layout and sharing rules are in
[`models/ppocrv6-models.json`](../models/ppocrv6-models.json). Small and
Medium use the same `PP-OCRv6_small_rec_dict.txt`, and all three variants use
the same Tiny CLS asset. The release archive is named
`lw.PPOCR.C-<version>-ppocrv6-models.zip` and has a matching `.sha256` file.

See the [PP-OCRv6 Small analysis snapshot](ppocrv6-small-analysis.md) for the
current graph-size and operator Go/No-Go findings.
The generated [PP-OCRv6 Medium analysis report](ppocrv6-medium-analysis.md)
records its DET/REC graph, current operator surface, and fixed-width REC
conversion checkpoint. Its versioned CI policy is
[`ci/ppocrv6-medium-validation.json`](../ci/ppocrv6-medium-validation.json).

The exact Tiny REC, fixed-batch CLS, and DET models are exposed through the
production public C APIs. Small REC and DET are currently exposed only through
experimental conversion and validation tools; the Small full-OCR experiment
uses the exact shared Tiny CLS asset. Its SHA-256 is part of the Small
validation contract. Encoded
image-file decoding stays outside the core API.

### Small and Medium model assets

These hashes identify the checked model assets. Small and Medium are still
analysis-only, but the ONNX files are included in the source tree and in the
dedicated model archive.

| Asset | Role | SHA-256 |
|---|---|---|
| PP-OCRv6 Small DET | detector | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |
| PP-OCRv6 Small REC | recognizer | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |
| PP-OCRv6 Small REC dictionary | CTC dictionary | `118d0f0714ad2a37668c23d6541f2c3feb65b8214041265b567f7fd5b3365d8e` |
| PP-OCRv6 Tiny CLS (shared) | direction classifier | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| PP-OCRv6 Medium DET | detector | `eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1` |
| PP-OCRv6 Medium REC | recognizer | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |

The Small dictionary contains 18,708 entries and the REC graph exposes 18,710
classes, matching the current decoder convention (blank plus trailing space).
The Small profile deliberately reuses the exact Tiny CLS asset above; no
separate Small CLS package is required.

### Tiny asset hashes

The following exact conversion inputs are analysis-verified. All three are
converter-, loader-, workspace-planner-, full-graph-output-, and public-pipeline
verified. Their composed full-OCR path has a real-image Golden test.

| Model | Role | Runtime priority | SHA-256 |
|---|---|---|---|
| PP-OCRv6 tiny REC | Recognition | v0.1 primary | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| PP-OCRv6 tiny CLS | Direction classification | v0.1 fixed batch | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| PP-OCRv6 tiny DET | Text detection quadrilaterals | v0.1 public pipeline | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |

“Analysis-verified” means ONNX validation, shape inference, operator inventory,
initializer inventory, dynamic-shape reporting, and representative FLOP
analysis pass. For REC and CLS, the private executor additionally produces the
complete output tensor and compares it with the original ONNX model. REC also
passes the pure-C preprocessing-to-text golden corpus with the production
dictionary; CLS preprocessing and its public result are compared with
independent NumPy and ONNX Runtime references.

DET is tested at dynamic `[1,3,32,32]` and `[1,3,32,64]` graph inputs. Its
converted output shape remains `[1,1,H,W]`; every graph output value is compared
with ONNX Runtime. Separate reference tests cover preprocessing, synthetic
postprocessing geometry/capacity, and the public real-image box pipeline.

The composed full-OCR gates verify the sample's 16 reading-order text lines,
DET/CLS/REC metadata, the detected 180-degree correction, exact capacity-query
semantics, unchanged output buffers on capacity failure, CLS-disabled creation,
and crop-pixel resource rejection. A separate seven-case versioned corpus adds
deterministic scale, aspect-ratio, 90-degree rotation, and blank-image coverage,
including tolerant original-image box coordinates. OpenCV is used only by the
crop test oracle; it is not linked into or shipped with the runtime.

The deterministic REC conversion currently produces a 4,455,632-byte LWM v0.1
file with SHA-256
`f5d8250797d0de82fc781efa988bf5bcfc1f7598c59a7668a7bd4b5ba84ff289`.
This hash is experimental and will change when the format or workspace plan
changes.

The deterministic fixed-batch CLS conversion produces a 1,017,568-byte LWM
v0.1 file with SHA-256
`cd453e08523d4c9677ea6b0d234277f7bba7c4371554f626d9dba0dd96042c84`.
This hash is likewise experimental before the format is frozen.

The deterministic DET conversion produces a 1,770,448-byte LWM v0.1 file with
SHA-256
`ba9164d371ac7003f90710c3106a344aeb906df2b0f1e7617fcf4608fa8cd66c`.
It is also experimental and will change with converter or format changes.
