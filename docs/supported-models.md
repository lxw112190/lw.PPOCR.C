# Supported models

No model is runtime-supported yet because the pure-C executor has not been
implemented.

The following exact conversion inputs are analysis-verified. REC is also
converter-, loader-, dynamic-shape-, and workspace-planner-verified; CLS and
DET do not yet have LWM converters.

| Model | Role | Runtime priority | SHA-256 |
|---|---|---|---|
| PP-OCRv6 tiny REC | Recognition | v0.1 primary | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| PP-OCRv6 tiny CLS | Direction classification | after REC | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| PP-OCRv6 tiny DET | Text detection | after CLS | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |

“Analysis-verified” means ONNX validation, shape inference, operator inventory,
initializer inventory, dynamic-shape reporting, and representative FLOP
analysis pass. It does not mean inference results are produced by lw.PPOCR.C.

The deterministic REC conversion currently produces a 4,455,632-byte LWM v0.1
file with SHA-256
`f5d8250797d0de82fc781efa988bf5bcfc1f7598c59a7668a7bd4b5ba84ff289`.
This hash is experimental and will change when the format or workspace plan
changes.
