# Supported models

The exact REC and fixed-batch CLS models are exposed through experimental
public C APIs. REC is verified through preprocessing, its full graph, and UTF-8
CTC decoding. CLS is verified through preprocessing, its full graph, and the
0/180-degree result. Input/output ownership is defined; encoded image-file
decoding stays outside the core API.

The following exact conversion inputs are analysis-verified. All three are
converter-, loader-, workspace-planner-, and full-graph-output-verified. REC
and CLS additionally have public pipeline verification; DET currently stops at
the private probability-map graph boundary.

| Model | Role | Runtime priority | SHA-256 |
|---|---|---|---|
| PP-OCRv6 tiny REC | Recognition | v0.1 primary | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| PP-OCRv6 tiny CLS | Direction classification | v0.1 fixed batch | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| PP-OCRv6 tiny DET | Text detection probability map | v0.1 private graph | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |

“Analysis-verified” means ONNX validation, shape inference, operator inventory,
initializer inventory, dynamic-shape reporting, and representative FLOP
analysis pass. For REC and CLS, the private executor additionally produces the
complete output tensor and compares it with the original ONNX model. REC also
passes the pure-C preprocessing-to-text golden corpus with the production
dictionary; CLS preprocessing and its public result are compared with
independent NumPy and ONNX Runtime references.

DET is tested at dynamic `[1,3,32,32]` and `[1,3,32,64]` inputs. Its converted
output shape remains `[1,1,H,W]`; every output value is compared with ONNX
Runtime. This is not yet a claim for preprocessing or detection boxes.

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
