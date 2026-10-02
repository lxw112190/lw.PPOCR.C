# Native x64 finer REC widths (experimental)

`LW_NATIVE_FINE_REC_WIDTHS=ON` reduces recognition padding in compiled native
x64 OCR. It is **OFF by default**, independent of `LW_WASM_FINE_REC_WIDTHS`.
WASM, ARM64 and canonical native builds are unchanged. Configuration rejects
enabling this option without the native x64 fast path.

The existing buckets are 192/320/480/640/960. The experiment uses
192/256/320/384/448/480/512/576/640/704/768/832/896/960. The minimum remains 192
and maximum remains 960; detection resolution, CLS, reading order and model
weights do not change. Widths above 960 still follow the existing cap. Session
and packed-weight reuse remain in effect; this does not enable resident widths.

The native experiment now preserves the production 480 bucket: every content
width from 1 through 960 is tested to ensure its selected bucket is no larger
than under the five-width policy. The independent WASM fine policy remains
unchanged at 13 buckets. This padding bound does not imply identical OCR text.

Less padding means less REC computation. However, padding also participates in
the model's attention computations: this is a shape-policy change, **not a
numerically equivalent kernel optimization**. Review text quality before
enabling it for a product. Production golden checksums are not relaxed.

## Local sample measurements

Historical first experiment (13 buckets, before restoring 480): baseline code
`21bc5d8`; candidate: the same source plus this option enabled.
Windows x64, AMD Ryzen 7 7735H (16 logical CPUs), MSVC 2019 Release, official
ONNX assets from `models/ppocrv6-models.json`, shared Tiny CLS enabled,
500x500 bundled PPM, REC max 960. Each case uses three alternating fresh-process
pairs, two warm-ups and five measured OCR calls per process. Values are medians
of per-process means; speedup is the median of paired before/after ratios.

| Model | Workers | Before ms | Fine ms | Paired speedup (range) | Before peak MiB | Fine peak MiB | Full text |
|---|---:|---:|---:|---|---:|---:|---|
| Tiny | 1 | 111.49 | 108.31 | 1.035x (1.027–1.057) | 84.2 | 84.9 | Same |
| Tiny | 4 | 54.41 | 50.27 | 1.082x (1.001–1.113) | 101.6 | 100.3 | Same |
| Small | 1 | 370.14 | 352.00 | 1.052x (1.013–1.125) | 198.4 | 199.6 | Different |
| Small | 4 | 284.05 | 210.81 | 1.347x (1.108–1.467) | 234.1 | 229.8 | Different |
| Medium | 1 | 1214.48 | 1101.61 | 1.116x (1.053–1.169) | 745.1 | 746.1 | Same |
| Medium | 4 | 976.96 | 875.01 | 1.105x (1.095–1.117) | 764.5 | 764.0 | Same |

Sample SHA-256:
`a694ab9def7845b53470aab9f3b819d6469a7b652f820fa30799d918cede4743`.
Peak working set includes initialization and the benchmark's extra standalone
detector, not just an OCR handle. It is not retained allocation size. The extra
CLI text collection runs after timing and is excluded from these measurements.

Small changes one line from `OEMODM` to `OEM ODM`; the user confirmed the latter
is correct. Its native fine-policy checksum is pinned independently in
`ci/native-fine-ppocrv6-small.json`, not substituted into the default Web/Node
goldens. It remains a quality experiment, not an output-equivalent speedup.
Small 4-worker timings show
substantial variance: do not treat 1.347x as a stable corpus-wide gain. Tiny and
Medium retain their existing sample text checksums. A single sample is not a
quality corpus; broader CER/Exact Line Rate and long/short line coverage are
required before promoting this width policy.

An additional one-pair confirmation measured Small 4 workers at 205.56 vs
182.65 ms (1.125x), and Medium 1/4 workers at 1.124x/1.111x. This supports the
direction of the gain, not a universal Small 4-worker improvement of 34.7%.

Several pointwise tiling, cache blocking, spin-wait and Erf experiments were
also measured; none produced a convincing all-model full-OCR benefit. Those
runtime/kernel changes were removed, rather than left enabled on microbenchmarks.

## Final 14-bucket sample A/B (2026-10-02)

Both builds use the same source with `LW_NATIVE_FINE_REC_WIDTHS` OFF/ON; all
other `LW_` CMake settings match. Same local machine and sample as above,
three alternating fresh-process pairs, two warm-ups, five measured OCR calls,
CLS enabled and REC max width 960. Values are per-process mean medians; the
speedup is the median of paired ratios, **not** the ratio of these medians.

| Model | Workers | Before ms | Fine ms | Paired speedup (range) | Before peak MiB | Fine peak MiB |
|---|---:|---:|---:|---|---:|---:|
| Tiny | 1 | 115.64 | 109.50 | 1.053x (1.033–1.115) | 84.6 | 85.1 |
| Tiny | 4 | 55.48 | 52.86 | 1.050x (1.041–1.091) | 101.6 | 100.6 |
| Small | 1 | 413.48 | 341.51 | 1.110x (1.107–1.348) | 198.0 | 199.8 |
| Small | 4 | 247.19 | 204.13 | 1.262x (1.085–1.303) | 234.0 | 229.8 |
| Medium | 1 | 1517.09 | 1446.75 | 0.989x (0.821–1.504) | 745.7 | 746.2 |
| Medium | 4 | 1085.65 | 912.34 | 1.139x (1.132–1.190) | 764.5 | 763.7 |

Tiny/Medium sample text matches; Small only changes the approved `OEM ODM`
spacing. Memory remains approximately flat. Medium 1-worker ratios have high
variance and cross 1.0, so **no gain or regression direction is established**
for that case. Small also has substantial timing spread. Use the same-runner
CI to reproduce; do not promote these local sample medians to corpus-wide
performance guarantees. JSON retains each raw measurement and the ratio range;
CI Markdown now prints the range and warns when the direction is unresolved.

## 100-image quality validation (2026-10-02)

The final 14-bucket policy was compared against the default five buckets on
the project's own generated corpus: 100 images, 614 ground-truth lines, seven
canvas sizes from 640x480 through 1792x1392, Chinese/English, identifiers, long
lines and 0/180-degree orientation. Seed: 20260907; Pillow 10.4.0; fonts:
Microsoft YaHei and SimSun. Manifest SHA-256:
`c51474cb3761515c8c9b07c0afb1846303aba1b2304ef9a87043c9d8c282159d`.
Both sides use official ONNX assets, shared CLS and REC max width 960.

| Model | Exact lines before / after | Exact rate before / after | Matched CER before / after | Edit errors before / after |
|---|---:|---|---|---:|
| Tiny | 366 / 366 | 59.609% / 59.609% | 3.6856% / 3.6757% | 740 / 738 |
| Small | 481 / 479 | 78.339% / 78.013% | 1.4247% / 1.4247% | 286 / 286 |
| Medium | 490 / 489 | 79.805% / 79.642% | 1.4012% / 1.3863% | 282 / 279 |

Detection F1 is unchanged: Tiny 99.352%, Small 99.919%, Medium 100%.
CER is computed only on IoU-matched lines, using NFC without deleting spaces
or normalizing punctuation. Missing lines are visible through detection and
exact-reference-line metrics; CER alone must not hide them.

**Do not promote the option to the production default yet.** Small loses two
exact lines and Medium loses one even though their aggregate CER does not
increase. Restoring 480 prevents increased padding but does not eliminate these
attention/decoding changes. Sample spacing approval is not blanket approval of
all changed corpus text. Generated-data results also do not establish quality
on real photographs or scanned documents.

This follow-up also measured line-only worker scheduling and a linear-epilogue
Pointwise specialization. The scheduling change slowed all three models with
four workers; Pointwise results were inconsistent and included regressions.
Both were reverted. No unproven kernel or scheduling change is shipped.

## Reproduce

Build both configurations from **one source revision** with one compiler:

```powershell
cmake -S . -B build/default -A x64 -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON -DLW_NATIVE_FINE_REC_WIDTHS=OFF -DLW_REC_RESIDENT_WIDTHS=OFF
cmake -S . -B build/fine -A x64 -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON -DLW_NATIVE_FINE_REC_WIDTHS=ON -DLW_REC_RESIDENT_WIDTHS=OFF
cmake --build build/default --config Release --target lw-ocr-benchmark lw-ocr-ppm
cmake --build build/fine --config Release --target lw-ocr-benchmark lw-ocr-ppm
python tools/compare_onnx_e2e.py --baseline build/default/Release/lw-ocr-benchmark.exe --candidate build/fine/Release/lw-ocr-benchmark.exe --sample build/default/models/sample.ppm --rounds 3 --iterations 5 --report-text-differences --collect-text --output build/onnx-width-comparison
```

The manual GitHub Actions workflow **Native x64 ONNX fine REC widths A/B** runs
the same comparison, then generates 100 project-owned images and measures
three-model CER, exact-line rate and detection F1. Default and fine builds must
both pass `native_onnx_import`; fine Small uses the reviewed native-only golden.
The experimental comparison records
full text differences without overwriting production goldens. Run/worker nondeterminism,
invalid results and incorrect model identities remain hard failures. Latency
and memory on hosted runners are informational. Experimental corpus accuracy
is also reported explicitly, including `quality_pass: false` when it regresses;
a green experiment workflow does **not** approve changing the product default.

For a strict local accuracy gate (exits nonzero on any aggregate exact-rate or
detection-F1 drop, or CER increase):

```powershell
python tools/generate_ocr_dataset.py --output build/onnx-width-corpus --count 100 --seed 20260907 --format jpg --font C:/Windows/Fonts/msyh.ttc --font C:/Windows/Fonts/simsun.ttc
python tools/compare_native_rec_quality.py --baseline build/default/Release/lw-ocr-ppm.exe --candidate build/fine/Release/lw-ocr-ppm.exe --dataset build/onnx-width-corpus/metadata.json --output build/onnx-width-quality
```

The manual experiment CI explicitly passes `--report-quality-differences` to
retain failed accuracy diagnostics without making a shape-policy experiment a
production quality claim. Malformed/incompatible reports still fail. Reports
include subgroup deltas (category/orientation/canvas) and font/manifest/model/
binary identities. Child Python processes use UTF-8 on Windows. Corpus images
remain ignored; CI uploads reports and the manifest, not the generated images.

Reports include binary/model/sample SHA identities, selected CMake options,
individual process measurements, paired ratio ranges, and untimed full text
whose buffer checksum must match the measured benchmark. All generated outputs
stay under ignored `build/` directories.
