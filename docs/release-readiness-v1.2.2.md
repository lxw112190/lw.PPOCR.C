# v1.2.2 release readiness / 发布准备清单

The patch includes the ARM64 NEON and thread scheduling changes through
`c097a5b`. The BM1684 benchmark results refer to source builds on that device.
[Benchmark and compatibility scope](release-notes-v1.2.2.md)

- Commit and push the release preparation changes to `main`. Record the exact
  commit and review the checks that ran for that SHA.
- Run the release metadata checker and version, asset, model-pack, and Web
  profile tests. Verify the annotated `v1.2.2` tag points at the reviewed
  release commit.
- Push the tag to start the Release workflow. It builds and tests eight
  reusable workflows and verifies 19 primary assets before publishing.
- To obtain the standalone Linux ARM64 Preview archive for this patch,
  run `customer-linux-architectures.yml` at `v1.2.2`. Verify the archive
  checksum and test the extracted archive on the deployment target.

```powershell
python -m unittest tests.test_versioning tests.test_release_assets tests.test_runtime_model_pack tests.test_web_release_profile tests.test_winforms_release -v
python tools/check_release_readiness.py --mode stable --version 1.2.2
git diff --check
git rev-parse HEAD
git tag --list v1.2.2
git ls-remote --tags origin refs/tags/v1.2.2
git tag -a v1.2.2 -m "lw.PPOCR.C v1.2.2: ARM64 native OCR performance"
git show --no-patch v1.2.2
git push origin v1.2.2
```

The workflow accepts an annotated tag without cryptographic signing.
The versioned Linux customer archive is a GitHub Actions artifact.
[Release workflow](../.github/workflows/release.yml) ·
[Linux package instructions](package.md#build-customer-linux-architecture-packages)
