# Architecture

## Current boundary

```text
Development machine                    Deployment target

PP-OCR ONNX                            rec.lwm
    |                                     |
Python + ONNX analyzer/converter          pure-C loader + session planner
    |                                     |
validated, simplified REC graph           resolved shapes + one workspace
    |                                     |
platform-independent LWM v0                scalar executor/preprocess/CTC (future)
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
4. Tensor and dynamic session memory planning — complete; scalar executor next.
5. One scalar operator at a time with reference tests.
6. Pure-C preprocess, CTC decoding, and REC golden tests.
7. Only after correctness: memory, SIMD, threads, CLS, DET, and full OCR.

## Compatibility claims

The design preserves Windows 7 x86 compatibility, but no such runtime claim is
made until a 32-bit binary is tested on a physical Windows 7 SP1 machine.
Windows x64 and Linux x64 are the first implementation targets. Linux ARM64 is
a planned primary target, not yet CI-verified.
