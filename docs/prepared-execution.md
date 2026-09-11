# Prepared Execution (experimental)

`lw.PPOCR.C` keeps model constants and execution bindings separate:

- `lw_shared_prepared_constants` owns the reference-counted packed weights and
  per-node packing metadata shared by sessions.
- `lw_bound_node` is session-local metadata for a future prepared dispatch
  table. Its constant pointer refers to the shared metadata; it does not copy
  model weights.

The first phase is structural and conservative. Generic operator inputs and
parameters still come from the validated LWM node, while prepared Conv1x1,
Conv3x3, and MatMul paths consult the session-local binding table only when
the experiment is enabled. The feature is disabled by default:

```text
-DLW_EXPERIMENTAL_PREPARED_EXECUTION=OFF
```

To build and exercise the table in an isolated build directory:

```text
cmake -S . -B build-prepared-execution \
  -DLW_EXPERIMENTAL_PREPARED_EXECUTION=ON \
  -DLW_BUILD_HTTP_DEMO=OFF \
  -DLW_BUILD_CSHARP_DEMOS=OFF
cmake --build build-prepared-execution --config Release --target test-session-planner
ctest --test-dir build-prepared-execution -C Release --output-on-failure -R "^session_planner$"
```

The session test checks that the experimental table mirrors each validated LWM
node and that prepared-constant pointers remain tied to the shared constant
array. The default build checks the opposite contract: no execution table is
allocated and the existing runtime path is unchanged.

When the table is enabled, the internal full-OCR profile reports the counters
under `implementation_paths.<component>.prepared_binding`:

- `lookups`: nodes that consulted the session-local table;
- `hits`: nodes with a prepared constant binding;
- `fallbacks`: nodes without a prepared binding that continued through the
  generic path.

The counters are zero in the default build and are intentionally informational;
they are not part of the public C ABI.

The next phase can bind a small, measured operator subset (starting with
prepared Conv1x1) behind the same option. It should only be enabled after
scalar parity, native SIMD parity, checksum, and representative REC latency
measurements pass.
