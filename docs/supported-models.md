# Supported models

The exact REC and fixed-batch CLS models are exposed through experimental
public C APIs. REC is verified through preprocessing, its full graph, and UTF-8
CTC decoding. CLS is verified through preprocessing, its full graph, and the
0/180-degree result. Input/output ownership is defined; encoded image-file
decoding stays outside the core API.

The following exact conversion inputs are analysis-verified. REC and CLS are
also converter-, loader-, workspace-planner-, full-graph-output-, and public
pipeline-verified. DET does not yet have an LWM converter.

| Model | Role | Runtime priority | SHA-256 |
|---|---|---|---|
| PP-OCRv6 tiny REC | Recognition | v0.1 primary | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| PP-OCRv6 tiny CLS | Direction classification | v0.1 fixed batch | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| PP-OCRv6 tiny DET | Text detection | after CLS | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |

“Analysis-verified” means ONNX validation, shape inference, operator inventory,
initializer inventory, dynamic-shape reporting, and representative FLOP
analysis pass. For REC and CLS, the private executor additionally produces the
complete output tensor and compares it with the original ONNX model. REC also
passes the pure-C preprocessing-to-text golden corpus with the production
dictionary; CLS preprocessing and its public result are compared with
independent NumPy and ONNX Runtime references.

The deterministic REC conversion currently produces a 4,455,632-byte LWM v0.1
file with SHA-256
`f5d8250797d0de82fc781efa988bf5bcfc1f7598c59a7668a7bd4b5ba84ff289`.
This hash is experimental and will change when the format or workspace plan
changes.

The deterministic fixed-batch CLS conversion produces a 1,017,568-byte LWM
v0.1 file with SHA-256
`cd453e08523d4c9677ea6b0d234277f7bba7c4371554f626d9dba0dd96042c84`.
This hash is likewise experimental before the format is frozen.
