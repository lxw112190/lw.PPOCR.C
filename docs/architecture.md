# Architecture

## Current boundary

```text
Development machine                    Deployment target

PP-OCR ONNX                            rec.lwm + dictionary + BGR8 pixels
    |                                     |
Python + ONNX analyzer/converter          pure-C loader + session planner
    |                                     |
validated, simplified REC graph           public recognizer C API
    |                                     |
platform-independent LWM v0                preprocess + executor + CTC
                                          |
decoded BGR pixels -> REC preprocess -> probabilities -> UTF-8 CTC text
```

The converter and runtime are separate products with separate dependency
contracts:

- `converter/` may use Python, ONNX, NumPy, and protobuf;
- the deployment runtime may use only C11, libc, and narrowly wrapped OS APIs;
- ONNX names and graph metadata are not required at runtime unless retained for
  diagnostics in a non-executable debug section;
- the first executable scope is the exact bundled PP-OCRv6 tiny REC graph.

## Dependency direction

```text
models + converter
        |
       LWM
        |
runtime loader -> tensor/memory -> executor -> scalar kernels
                                           |
                              REC preprocess + CTC
```

`src/runtime` must not know about OCR text, dictionaries, boxes, or DB
post-processing. `src/ppocr` may use the runtime API but not its private model
layout. Platform code is isolated under `src/platform`; SIMD dispatch will be
isolated under `src/simd` after scalar correctness.

## Development gates

1. Exact model analysis and operator report — complete.
2. Non-frozen LWM v0 definition and deterministic REC converter — complete.
3. Bounds-checked loader for untrusted LWM input — complete.
4. Tensor and dynamic session memory planning — complete.
5. One scalar operator at a time with reference tests — complete; all 15
   operator types and 161 converted REC nodes have scalar Kernels.
6. Exact REC graph executor and complete-output comparison — complete as a
   private interface for widths 7 and 17; applications use its public
   recognize-only wrapper rather than the executor directly.
7. Pure-C preprocess, CTC decoding, and REC golden tests — complete behind
   private interfaces, including a ten-crop real-image corpus, ONNX reference
   comparison, and the production dictionary.
8. Public recognize-only C API with caller-owned UTF-8 output — complete and
   contract-tested on Windows x64/x86; ABI remains experimental before 1.0.
9. Static/shared development packages and dependency-free C Demo — complete
   locally for Windows x64/x86, including installed-package execution.
10. Scalar performance, deterministic-repeat, and RSS baseline — complete
    locally on Windows x64/x86 with a packaged JSON benchmark tool.
11. First profile-directed Scalar Conv address/bounds optimization — complete
    locally with end-to-end Windows x64/x86 A/B measurements and unchanged
    ONNX/Golden correctness gates.
12. Profile-directed pointwise Conv data-layout fast path — complete locally;
    node profiling identified 25 pointwise nodes as 96.24% of Conv time and
    Windows x64/x86 end-to-end latency fell by about 90% from the baseline.
13. Ordinary 3x3 stride-2 Conv spatially local path — complete locally with
    unchanged x64/x86 reference and Golden results; median end-to-end latency
    is now 12.842x and 11.174x faster than the original baselines respectively.
14. Cache-contiguous, four-row-blocked MatMul — complete locally; the x64
    MatMul median fell 57.23%, and x64/x86 end-to-end latency improved again
    with unchanged reference, Golden, determinism, and memory gates.
15. Next: add isolated CPU-feature dispatch and explicit SIMD for the now
    distributed Conv, MatMul, and elementwise hotspots before considering
    threads, CLS, DET, and full OCR.

## Compatibility claims

The design preserves Windows 7 x86 compatibility, but no such runtime claim is
made until a 32-bit binary is tested on a physical Windows 7 SP1 machine.
Windows x64 and Linux x64 are the first implementation targets. Linux ARM64 is
a planned primary target, not yet CI-verified.
