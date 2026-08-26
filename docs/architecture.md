# Architecture

## Current boundary

```text
Development machine                    Deployment target

PP-OCR ONNX                            rec.lwm / cls.lwm / det.lwm + BGR8 pixels
    |                                     |
Python + ONNX analyzer/converter          pure-C loader + session planner
    |                                     |
validated REC / CLS / DET graphs          public recognizer / classifier / detector C APIs
    |                                     |
platform-independent LWM v0                preprocess + executor + CTC/class result
                                          |
decoded BGR pixels -> REC preprocess -> probabilities -> UTF-8 CTC text
decoded BGR pixels -> CLS preprocess -> probabilities -> 0/180 orientation
decoded BGR pixels -> DET -> perspective crop -> optional CLS -> REC -> UTF-8 lines
```

The converter and runtime are separate products with separate dependency
contracts:

- `converter/` may use Python, ONNX, NumPy, and protobuf;
- the deployment runtime may use only C11, libc, and narrowly wrapped OS APIs;
- ONNX names and graph metadata are not required at runtime unless retained for
  diagnostics in a non-executable debug section;
- the executable scope covers the exact bundled PP-OCRv6 tiny REC graph and
  fixed-batch CLS graph, and the DET probability graph.

## Dependency direction

```text
models + converter
        |
       LWM
        |
runtime loader -> tensor/memory -> executor -> scalar/SIMD kernels
                                           |
                              REC preprocess + CTC
                              CLS preprocess + orientation
                              DET + crop + optional CLS + REC composition
```

`src/runtime` must not know about OCR text, dictionaries, boxes, or DB
post-processing. `src/ppocr` may use the runtime API but not its private model
layout. Platform code is isolated under `src/platform`; CPU feature detection
and architecture-specific kernels are isolated under `src/simd`.

## Development gates

1. Exact model analysis and operator report — complete.
2. Non-frozen LWM v0 definition and deterministic REC converter — complete.
3. Bounds-checked loader for untrusted LWM input — complete.
4. Tensor and dynamic session memory planning — complete.
5. One scalar operator at a time with reference tests — complete for all 21
   LWM operator IDs used by REC, CLS, and DET.
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
15. x86/x64 SSE2 MatMul dispatch — complete locally; x64 uses its architectural
    SSE2 baseline, x86 checks CPUID bit 26, other architectures retain the
    scalar path, and the benchmark reports the selected backend.
16. x86/x64 AVX2 MatMul dispatch — complete locally; CPUID, OSXSAVE, and XGETBV
    checks prevent AVX state use on unsupported operating systems, the isolated
    eight-column kernel avoids FMA so it preserves the scalar accumulation
    order, and SSE2/scalar fallbacks remain available.
17. x86/x64 SSE2/AVX2 pointwise Conv dispatch — complete locally; isolated
    four- and eight-lane spatial loops retain the input-channel accumulation
    order, direct reference tests require byte-identical output, and grouped
    pointwise Conv retains the scalar fallback.
18. Flat binary SSE2/AVX2 dispatch — complete locally; profiling showed 55 of
    87 Add/Mul/Div nodes use either two same-shaped contiguous inputs or one
    contiguous input plus a right scalar. These modes now bypass per-element
    broadcast coordinate tracking, while all other broadcasts retain the
    general implementation.
19. Single-axis binary broadcast blocks — complete locally; the remaining 32
    REC binary nodes all broadcast one non-unit right dimension. Channel-style
    blocks reuse right-scalar SIMD and trailing-vector blocks reuse contiguous
    pair SIMD without per-element coordinate traversal.
20. Stride-1 3x3 Depthwise Conv SIMD — complete locally; seven REC nodes use
    this exact pad-1 shape. Four- and eight-output-wide kernels retain each
    output element's nine-weight accumulation order and scalar boundary tails.
21. Ordinary stride-2 3x3 Conv SIMD — complete locally; the two REC stem nodes
    use safe deinterleaving loads to advance eight AVX2 or four SSE2 output
    columns while preserving scalar accumulation order.
22. Fixed-batch CLS converter and graph execution — complete; offline metadata
    specialization reuses existing operators plus one checked static Reshape,
    and full graph output matches the original ONNX model on x64/x86.
23. Public CLS direction-classification API — complete; BGR preprocessing,
    0/180-degree labels, scores, UTF-8 paths, resource limits, shared exports,
    dependency-free PPM demo, and installed packages are contract-tested.
24. Exact DET probability graph — complete; its converter, five new data-path
    operators, dynamic shape planning, deterministic repeat, x64/x86 builds,
    and ONNX Runtime complete-output comparison are required gates.
25. Public DET pipeline — complete; decoded BGR preprocessing, dynamic session
    reuse, bounded DB-style postprocessing, original-coordinate quadrilaterals,
    capacity semantics, shared exports, real-image tests, and packaged Demo.
26. Pure-C crop extraction and composed DET/optional CLS/REC full-OCR API —
    complete; the API uses caller-owned dual-capacity output, one canonical
    quadrilateral, bounded crop pixels, optional 180-degree correction, a
    dependency-free PPM Demo, OpenCV crop-oracle tests, and a real-image text
    Golden gate on Windows x64/x86.
27. Next: broaden the full-OCR Golden corpus and harden invalid-input,
    sanitizer, repeated-run memory, and cross-platform CI coverage before any
    ABI freeze.

## Compatibility claims

The design preserves Windows 7 x86 compatibility, but no such runtime claim is
made until a 32-bit binary is tested on a physical Windows 7 SP1 machine.
Windows x64 and Linux x64 are the first implementation targets. Linux ARM64 is
a planned primary target, not yet CI-verified.
