# Prepared Execution (experimental)

`lw.PPOCR.C` keeps model constants and execution bindings separate:

- `lw_shared_prepared_constants` owns the reference-counted packed weights and
  per-node packing metadata shared by sessions.
- `lw_bound_node` is session-local metadata for the prepared dispatch
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

Phase 3.1 and Phase 3.2 now bind a small, measured Conv subset behind the
same option. The option remains experimental until scalar parity, native SIMD
parity, checksum, and representative REC latency measurements are repeated on
the release performance runners.

## Phase 3.1: direct packed Conv1x1 execution

When `LW_EXPERIMENTAL_PREPARED_EXECUTION=ON`, eligible packed Conv1x1
nodes bind their concrete scalar/SSE2/AVX2/NEON/LSX kernel, packed-weight
pointer, tensor indices, and output tile during session creation. The executor
uses this short path for single-worker and parallel execution; generic nodes
continue through the existing interpreter. AVX2 FMA policy is intentionally
not selected by this phase so the A/B comparison isolates dispatch overhead.

The internal profile now reports:

- `prepared_execution.prepared_nodes`: nodes executed through a prepared path;
- `prepared_execution.generic_nodes`: nodes executed by the generic interpreter;
- `prepared_execution.conv1x1`: prepared Conv1x1 executions.
- `prepared_execution.conv3x3`: prepared packed stride-2 Conv3x3 executions.

A prepared run must keep the same line count and output checksum as the default
run. The initial Conv1x1-only A/B result was positive at REC width 960, so the
option remains experimental while the broader operator subset is measured.

### Phase 3.1 A/B baseline (AVX2, FMA dispatch OFF)

The first complete local matrix used the default build and an isolated build
with only `LW_EXPERIMENTAL_PREPARED_EXECUTION=ON`. Both builds used the same
Tiny model, bundled sample, and AVX2 host. Each cell used five warm-up runs,
thirty measured iterations, and five repeats. Checksums and line counts matched
in every cell.

| REC width | Workers | Default mean (ms) | Prepared mean (ms) | Mean change | RSS delta |
|---:|---:|---:|---:|---:|---:|
| 320 | 1 | 228.105 | 226.069 | -0.89% | +0.078 MiB |
| 320 | 4 | 109.164 | 108.908 | -0.23% | +0.176 MiB |
| 960 | 1 | 346.451 | 301.203 | -13.06% | +0.086 MiB |
| 960 | 4 | 152.643 | 149.494 | -2.06% | +0.676 MiB |

The result is stable enough to keep the direct Conv1x1 path as an experimental
candidate. It is not enabled by default yet: the 320-width gain is within
noise, while the 960-width result is materially better. The next decision should be made after repeating this matrix on the release
performance runner and checking the same checksums across the supported SIMD
backends. Phase 3.2 applies the same guarded approach to the existing packed
Conv3x3 contract.

### Phase 3.2 direct packed Conv3x3 candidate

The existing packed Conv3x3 contract is limited to group-1, 3x3 kernel,
stride-2, pad-1 nodes with output channels divisible by eight. Phase 3.2 binds
the already-prepared weights and selects the existing no-FMA AVX2 kernel (or the
scalar packed kernel fallback). All other Conv3x3 shapes continue through the
existing dispatcher.

A follow-up A/B matrix used the same protocol as Phase 3.1 (five warm-ups,
thirty measured iterations, five repeats, AVX2 with FMA dispatch off):

| REC width | Workers | Default mean (ms) | Prepared Conv1x1+3x3 mean (ms) | Mean change | RSS delta |
|---:|---:|---:|---:|---:|---:|
| 320 | 1 | 301.789 | 293.276 | -2.82% | +0.074 MiB |
| 320 | 4 | 127.661 | 128.188 | +0.41% | +0.168 MiB |
| 960 | 1 | 431.236 | 392.655 | -8.95% | +0.090 MiB |
| 960 | 4 | 184.397 | 177.427 | -3.78% | +1.098 MiB |

Checksums and line counts matched in every cell. The prepared option remains
OFF by default because the 320/4 result is effectively neutral and the
memory increase is larger than the Conv1x1-only path. Before changing
the default, repeat this matrix on the release performance runner and verify
AVX2, SSE2, NEON, and LSX fallback behavior. The public API and LWM format are
unchanged.
