# ARM64 OCR performance profiling

The repository has a manual `ARM64 OCR Performance` workflow for collecting a
reproducible native baseline before adding new NEON kernels. It is deliberately
separate from the customer package workflow: a green package build proves
correctness and deployability, while this workflow records latency, operator
breakdown, thread use, and RSS.

## Run the workflow

Start `.github/workflows/arm64-performance.yml` manually. The default settings
are three warm-up calls, ten measured calls, and three instrumented profile
iterations. The workflow runs these cases:

| Line workers | DET intra-op threads | Purpose |
|---:|---:|---|
| 1 | 1 | single-thread baseline |
| 4 | 1 | line-worker scaling |
| 1 | 4 | DET intra-op scaling |
| 4 | 4 | combined candidate |

The workflow requires the native backend to report `neon`, checks the prepared
assets, runs the ARM64 correctness gates, and verifies that all cases retain
the same 16-line OCR checksum. It does not impose a latency threshold on the
shared GitHub runner because host scheduling and thermal state are noisy.

The `lw-ppocr-arm64-profile-results-<commit>` artifact contains the raw driver
JSON, a normalized `arm64-profile-summary.json`, and `SUMMARY.md`. The
`lw-ppocr-arm64-profile-kit-<commit>` artifact contains the exact ARM64
benchmark drivers and models needed for a physical-machine run. It is a
development artifact and is not part of a tagged release.

## Run the profile kit on an ARM64 device

After downloading and extracting the profile kit on the target device:

```bash
sha256sum --check SHA256SUMS
./run-profile.sh --output results/rk3576-all-cores
```

Before measuring, save the machine and scheduling state:

```bash
uname -a
lscpu
cat /proc/cpuinfo
for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  printf '%s=' "$file"
  cat "$file"
done
```

Do not assume which CPU numbers are the performance cores. Inspect `lscpu -e`
first, then optionally repeat the kit with `taskset` for all cores and the
performance-core set. Run at least three independent rounds; do not report the
fastest round as the baseline.

## Result contract

Every case must retain:

- backend `neon`;
- 16 OCR lines;
- one identical output checksum across all thread cases;
- finite positive latency and throughput values;
- positive RSS values on Linux;
- `det_intra_actual` no greater than the requested DET thread count.

The profile separates DET preprocessing, DET graph, DET postprocess, crop,
CLS, REC, line-worker critical path, operator classes, and DET convolution
nodes. Use the largest *per-request* operator totals to choose the next NEON
candidate; do not select a kernel from an x64 profile alone.

## Comparison discipline

Compare lw.PPOCR.C with another OCR implementation only after fixing the same
model variant, dictionary, image, CLS setting, REC target width, warm-up,
measurement count, CPU affinity, governor, and timing boundary. Report mean,
median, P95, throughput, warm RSS, peak RSS, CER, and exact-line rate together.

GitHub ARM64 numbers are CI trend data. A RK3576 result is an entity-specific
performance result and must identify the exact commit and downloaded artifact.
Neither result establishes performance on every ARM64 distribution or device.

## Kernel retention gates

An ARM64 kernel candidate is kept only when its direct reference test and the
full OCR gates pass, the checksum is unchanged, the targeted operator improves
in repeated alternating A/B runs, and complete OCR does not regress beyond
measurement noise. A node-level improvement must be documented as such when
the end-to-end wall time is inconclusive.
