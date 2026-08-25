# Private REC graph executor milestone

The runtime now executes every one of the 161 non-Identity nodes in the exact
converted PP-OCRv6 tiny REC graph. This is a correctness gate, not a public API
or performance claim.

## Execution contract

- one FP32 input matching the session's resolved `[1, 3, 48, W]` descriptor;
- exact input and output element counts are checked before execution;
- constants are read directly from the validated LWM weight section;
- intermediate and graph-output tensors use the session's 64-byte-aligned,
  lifetime-planned workspace;
- no heap allocation occurs during graph execution;
- one dispatcher covers all 15 converted operator types;
- a failed Kernel reports the node index, LWM operator id, and stable status;
- repeated calls reuse the workspace and must produce bit-identical output.

The private function is declared in `src/runtime/executor_internal.h`. It is
deliberately absent from `include/lw_infer.h`: public buffer ownership,
multi-input/output representation, concurrency, and ABI evolution must be
settled before applications can depend on it.

## Complete-output comparison

`rec_graph_reference` executes widths 7 and 17, covering one-step and two-step
dynamic REC outputs. It compares every FP32 value with the original opset-11
ONNX graph. ONNX Runtime 1.21.0 CPUExecutionProvider is a pinned development
dependency and is the required complete-graph oracle. ONNX's Python
ReferenceEvaluator is intentionally not used for this cumulative graph test.

On the Windows x64 development host, comparison against ONNX Runtime 1.21.0
measured:

| Input width | Output shape | Maximum absolute difference | Mean absolute difference |
|---:|---|---:|---:|
| 7 | `[1, 1, 6906]` | `6.377697e-6` | `1.0609754e-9` |
| 17 | `[1, 2, 6906]` | `3.8087368e-5` | `3.8027901e-9` |

Both Windows x64 and x86 builds pass the complete fifteen-test suite locally,
including the downstream real cropped-text recognition golden test described
in [`rec-pipeline.md`](rec-pipeline.md).
Physical Windows 7 and remote Linux CI remain separate platform claims.
