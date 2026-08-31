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
28. Client integration Demos — complete locally; an ABI-size-checked .NET
    Framework 3.5 P/Invoke wrapper powers the Windows WinForms image/overlay
    client, while a separate C++ `cpp-httplib` executable provides the
    cross-platform HTTP API, browser-side image-to-PPM conversion, stable
    request IDs, quadrilateral drawing, and staged-package live tests.
29. Session-prepared pointwise weights — complete locally; eligible group-1
    1x1 Conv weights are packed once into four-output-channel blocks, then
    consumed by scalar/SSE2/AVX2 microkernels. Long OCR maps use this layout on
    SSE2-capable hosts; x64 AVX2 also prepares sufficiently large square maps.
    LWM and public ABI layouts do not change.
30. Integer nearest-neighbor NCHW Resize — complete locally; unchanged N/C
    dimensions and exact integer spatial scale factors use contiguous
    horizontal replication plus row copies. Fractional, downsampling, and
    general-rank cases retain the coordinate-based reference path.
31. Exact NCHW reduction and pooling fast paths — complete locally; spatial
    keep-dimension ReduceMean walks contiguous channel planes without coordinate
    decoding, while the detector's exact 2x2/stride-1/SAME_UPPER MaxPool handles
    its interior and borders separately. Other axes and pool configurations
    retain the general reference paths.
32. Contiguous-axis Softmax — complete locally; a final or otherwise contiguous
    classification axis uses direct row pointers while retaining the same
    maximum shift, `expf`, sum and per-element division order. Strided axes keep
    the general implementation, and in-place operation remains supported.
33. AVX2 stride-2 3x3 output-channel blocking — complete locally; four NCHW
    output planes share each gathered eight-value input vector while keeping
    separate accumulators and the original input-channel/kernel addition order.
    Non-multiple-of-four output counts retain the previous streaming path or a
    small-input-channel spatial block.
34. Shape-aware x64 AVX2 pointwise microkernel — complete locally; four output
    planes now keep two eight-value spatial vectors live, halving packed-weight
    broadcasts for the 16-value main loop. x86 retains the 4x8 kernel because
    it has fewer vector registers, and only x64 AVX2 enables packed square DET
    maps with at least 256 spatial positions.
35. AVX2 Erf approximation — complete locally; three degree-8 polynomial
    regions cover absolute inputs below 4, larger finite magnitudes saturate to
    one, and the original sign is restored without branches. Dense numerical,
    special-value, graph, pipeline, Golden-corpus, x64 and x86 gates protect the
    approximation. SIMD capability is detected once per graph execution;
    Emscripten SIMD128 uses the same piecewise algorithm through a four-lane
    WASM kernel, while non-AVX2 native targets retain the scalar `erff` path.
36. REC width-distribution profiling — complete locally; the private full-OCR
    profiler records actual resized width, target width, padding ratio and
    stable 64/96/128/160/192/256/320/overflow buckets without changing the
    public ABI or adding work to ordinary OCR calls. Width bucketing remains an
    evidence-gated experiment rather than a default behavior change.
37. DET output-channel parallelism — complete locally; large group-1 and
    depthwise convolutions split disjoint output-channel ranges across a fixed
    session-owned thread pool. Small convolutions stay serial, packed 1x1
    ranges remain four-channel aligned, and DET finishes before the existing
    CLS/REC line workers start, avoiding nested oversubscription.
38. Known-capacity CTC decoding — complete locally; full OCR supplies the
    recognizer's documented maximum text capacity, so greedy CTC collapse can
    write text and compute its exact used capacity in one pass. Size queries
    and smaller caller buffers retain the original two-pass public contract.
39. AVX2 GELU superkernel — complete locally; an exact
    `Div -> Erf -> Add -> Mul -> Mul` chain is fused only when its constants,
    tensor shapes, private intermediate lifetimes, and AVX2 backend all match.
    The fused kernel preserves the established piecewise Erf approximation and
    floating-point operation order while removing four intermediate traversals.
40. AVX2 unit-stride 3x3 output-channel blocking — complete locally; regular
    group-1 convolutions with an output count divisible by four reuse each
    contiguous eight-value input vector across four NCHW output planes. Border
    pixels retain scalar bounds checks, non-multiple-of-four shapes retain the
    previous output-stream kernel, and the public model and C ABI are unchanged.
41. AVX2 long-axis Softmax — complete locally; contiguous axes of at least 256
    values use a stable AVX2 range-reduced exponential and preserve the
    established left-to-right sum order for score compatibility. Short axes,
    strided axes, non-AVX2 hosts, and all public ABI contracts retain the
    original scalar implementation.
42. REC MatMul four-row tiling — complete locally; the x64 AVX2 wide-matrix
    path applies each loaded eight-column weight vector to four output rows,
    preserving the established inner-dimension accumulation order.
43. Session-packed wide MatMul weights — complete locally; eligible constant B
    matrices are packed once into 16-column panels and consumed by an x64 AVX2
    4x16 microkernel. x86, non-AVX2, small and irregular shapes retain the
    canonical layout. The cache is session-owned and changes neither LWM nor
    the public C ABI.

## Compatibility claims

The design preserves Windows 7 x86 compatibility, but no such runtime claim is
made until a 32-bit binary is tested on a physical Windows 7 SP1 machine.
The WinForms Demo targets .NET Framework 3.5 and follows the referenced
project's Win7-tested interop patterns. The HTTP Demo uses its vendored
Win7-compatible cpp-httplib header. These new binaries are not themselves
claimed as Win7-verified until the same physical-machine checks are repeated.
Windows x64 and Linux x64 are the first implementation targets. The manual
customer workflow provides a native Ubuntu 22.04 ARM64 gate and an experimental
Debian 13 LoongArch64 gate under pinned QEMU/container images. Until those jobs
run successfully, their matrix entries remain pending; even after CI succeeds,
LoongArch64 customer-distribution and performance claims require the exact
archive to pass on physical customer hardware. Both non-x86 architectures use
the portable scalar kernels until architecture-specific SIMD work is added and
profiled.
