# Supported models

## Release support matrix

The project uses one pure-C Runtime and one LWM format for all three PP-OCRv6
profiles. Model support and default integration support are separate claims:

| Variant | Native full OCR | Browser SDK / HTML | Runtime model pack | Release status |
|---|---|---|---|---|
| Tiny | ✅ Windows/Linux golden | ✅ default artifact | ✅ | default |
| Small | ✅ Windows/Linux golden | ✅ opt-in artifact | ✅ | preview |
| Medium | ✅ Windows/Linux golden | ✅ opt-in artifact | ✅ | preview |

Tiny remains bundled by default in the C/HTTP, Android, Desktop Java, Node/WASM,
and browser packages. Small and Medium do not silently replace those defaults.
They are distributed as separate runtime model packs and as explicitly named
browser SDK/HTML artifacts.

All three profiles reuse the exact Tiny CLS model. Small and Medium share
`PP-OCRv6_small_rec_dict.txt`; Tiny uses its own dictionary. The authoritative
asset paths and hashes are in
[`models/ppocrv6-models.json`](../models/ppocrv6-models.json).

## Choosing a variant

| Use case | Starting point | Reason |
|---|---|---|
| Phone browser, embedded WebView, ordinary screenshots | Tiny | Lowest initialization, latency, and memory cost |
| Desktop browser or native application with a measured quality need | Small | Opt-in quality/resource trade-off; validate on the real corpus |
| Desktop/server evaluation with a large memory budget | Medium | Highest resource cost; not a mobile default |

A larger graph is not an automatic quality guarantee for every document. Choose
with the project's corpus tools or a customer-owned golden set, and compare
exact-line rate/CER as well as latency and peak working set. The versioned
Windows x64 960 baseline currently records Small at roughly 4.20x/4.58x Tiny
latency and Medium at roughly 21.95x/25.28x for one/four workers. See
[`docs/performance-baseline.md`](performance-baseline.md); those measurements are
an engineering reference, not a portable performance promise.

For browser deployments, prefer Tiny on phones. Small may be appropriate after
testing the exact target device. Medium's single-file HTML embeds its large
models and should be treated as a desktop-first preview; browser process limits,
initialization time, and memory pressure vary by device.

## Preview model-pack and browser gates

`v0.2.0-preview.1` introduces namespaced runtime model packs containing
`manifest.json`, `SHA256SUMS`, `det.lwm`, `cls.lwm`, `rec.lwm`, and the matching
dictionary. The manifest includes an `asset_set_id` for cache invalidation. On a
tagged release, each Tiny/Small/Medium pack is published only after Windows and
Linux produce byte-identical ZIP files.

The tagged-release browser workflow additionally builds Small and Medium SDK
and standalone HTML artifacts. Each SDK performs real OCR twice on one engine,
destroys it, creates a new engine, and repeats OCR. Both the SDK and standalone
HTML must match the variant-specific 16-line full-text SHA-256 contract in
`ci/web-ppocrv6-*.json`. PDF regression remains a Tiny-only gate. Medium browser
size and timing values are informational CI output, not hard performance limits.

These checks make Small and Medium usable preview variants; they do not freeze
the C ABI or LWM format and do not constitute broad physical-device support.
See [`docs/model-packs.md`](model-packs.md) for packaging details and the Small
and Medium analysis snapshots for graph/converter evidence.

### Small and Medium model assets

These hashes identify the checked model assets used by the opt-in preview
variants. The ONNX inputs are included in the source tree and in the dedicated
model archive.

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
