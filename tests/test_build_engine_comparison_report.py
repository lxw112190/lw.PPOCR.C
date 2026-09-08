from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_engine_comparison_report import (
    ComparisonError,
    align_lines,
    build_report,
    load_contract,
)


LW_COMMIT = "1" * 40
HARNESS_COMMIT = "2" * 40
MODEL_HASHES = {
    "det.lwm": "a" * 64,
    "cls.lwm": "b" * 64,
    "rec.lwm": "c" * 64,
    "ppocr_keys.txt": "d" * 64,
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EngineComparisonReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract = self.root / "contract.json"
        self.dataset = self.root / "dataset"
        self.results = self.root / "results"
        self.model_sums = self.root / "model-SHA256SUMS"
        self.output = self.root / "report"
        self._write_contract()
        self._write_dataset()
        self._write_model_sums()
        self._write_results()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_contract(self) -> None:
        value = {
            "schema_version": 2,
            "name": "test",
            "simd_repository": "lxw112190/SimdPaddleOCR",
            "simd_ref": HARNESS_COMMIT,
            "dataset": {
                "generator_project": "test/Sdcb.SimdPaddleOCR.TestData",
                "expected_images": 3,
                "warmup_images": 1,
                "measured_images": 2,
            },
            "models": ["tiny"],
            "workers": [1, 4],
            "replicas": 3,
            "profiles": {
                "normalized-fixed320": {
                    "requested_isa": "avx2",
                    "sharp": {"rec_policy": "fixed", "rec_width": 320},
                    "c": {"rec_policy": "fixed", "rec_width": 320},
                },
                "normalized-adaptive960": {
                    "requested_isa": "avx2",
                    "sharp": {"rec_policy": "lw-adaptive960", "rec_width": 960},
                    "c": {"rec_policy": "adaptive960", "rec_width": 960},
                },
                "product": {
                    "requested_isa": "best",
                    "sharp": {"rec_policy": "native-adaptive", "rec_width": 320},
                    "c": {"rec_policy": "adaptive-max", "rec_width": 960},
                },
            },
            "normalized_options": {
                "reading_order": "horizontal",
                "detector": {
                    "limit_side": 960,
                    "bitmap_threshold": 0.3,
                    "box_threshold": 0.6,
                    "unclip_ratio": 1.6,
                    "use_dilation": False,
                    "max_candidates": 1000,
                },
                "classifier": {"enabled": True, "threshold": 0.9},
            },
            "comparison_policy": {
                "performance_gate": False,
                "accuracy_gate": False,
                "correctness_gate": True,
                "ratios": "paired-within-replica",
            },
        }
        self.contract.write_text(json.dumps(value), encoding="utf-8")

    def _write_dataset(self) -> None:
        self.dataset.mkdir()
        images = []
        for index in range(1, 4):
            name = f"img-{index:03d}.jpg"
            path = self.dataset / name
            path.write_bytes(f"image-{index}".encode())
            images.append(
                {
                    "file": name,
                    "sha256": file_hash(path),
                    "lines": [{"text": f"line-{index}"}],
                }
            )
        metadata = {"version": 1, "seed": 7, "images": images}
        metadata_path = self.dataset / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        entries = [
            f"{file_hash(path)}  {path.name}"
            for path in sorted(self.dataset.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        (self.dataset / "SHA256SUMS").write_text(
            chr(10).join(entries) + chr(10), encoding="utf-8"
        )

    def _write_model_sums(self) -> None:
        self.model_sums.write_text(
            chr(10).join(
                f"{digest}  {name}" for name, digest in MODEL_HASHES.items()
            )
            + chr(10),
            encoding="utf-8",
        )

    def _write_results(self) -> None:
        for replica in range(1, 4):
            replica_dir = self.results / f"artifact-{replica}" / f"replica-{replica}"
            result_dir = replica_dir / "results"
            result_dir.mkdir(parents=True)
            cases = []
            for profile in ("normalized-fixed320", "normalized-adaptive960", "product"):
                for workers in (1, 4):
                    for engine in ("sharp", "c"):
                        name = f"{profile}-{workers}w-{engine}-r{replica}.json"
                        relative = f"results/{name}"
                        cases.append(relative)
                        (result_dir / name).write_text(
                            json.dumps(
                                self._result(replica, engine, profile, workers)
                            ),
                            encoding="utf-8",
                        )
            manifest = {
                "schema_version": 1,
                "replica": replica,
                "lw_commit": LW_COMMIT,
                "harness_commit": HARNESS_COMMIT,
                "contract_sha256": file_hash(self.contract),
                "dataset_manifest_sha256": file_hash(
                    self.dataset / "metadata.json"
                ),
                "c_backend": "avx2",
                "cases": cases,
            }
            (replica_dir / "replica-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

    def _result(
        self, replica: int, engine: str, profile: str, workers: int
    ) -> dict[str, object]:
        product = profile == "product"
        adaptive = profile == "normalized-adaptive960"
        rec_policy = (
            "native-adaptive"
            if engine == "sharp" and product
            else "lw-adaptive960"
            if engine == "sharp" and adaptive
            else "adaptive-max"
            if engine == "c" and product
            else "adaptive960"
            if engine == "c" and adaptive
            else "fixed"
        )
        rec_width = (
            960
            if adaptive or (product and engine == "c")
            else 320
        )
        rows = []
        exact = 0
        for index in range(1, 4):
            expected = [f"line-{index}"]
            texts = expected
            if engine == "c" and index == 3:
                texts = ["different"]
            if index > 1 and texts == expected:
                exact += 1
            rows.append(
                {
                    "file": f"img-{index:03d}.jpg",
                    "warmup": index == 1,
                    "hash": hashlib.sha256(
                        (f"img-{index:03d}.jpg" + "|".join(texts)).encode()
                    ).hexdigest(),
                    "texts": texts,
                    "working_set_mb": 120.0,
                }
            )
        meta = {
            "schema_version": 1,
            "mode": engine,
            "benchmark": True,
            "benchmarkKind": "engine",
            "caseId": f"{profile}-{workers}w-{engine}-r{replica}",
            "replica": replica,
            "model": "tiny",
            "workers": workers,
            "comparison_profile": profile,
            "rec_policy": rec_policy,
            "rec_width": rec_width,
            "requested_isa": "best" if product else "avx2",
            "effective_isa": "avx512" if product and engine == "sharp" else "avx2",
            "sampleCount": 3,
            "working_set_mb_loaded": 80.0 if engine == "sharp" else 60.0,
            "working_set_mb_peak": 140.0 if engine == "sharp" else 100.0,
            "machine": f"runner-{replica}",
            "cpuName": "Synthetic CPU",
            "cpu": 4,
            "memoryMb": 16384.0,
            "accuracy": {
                "exact_lines": exact,
                "total_lines": 2,
                "exact_img": exact,
                "images": 2,
                "errors": 0 if engine == "sharp" else 1,
                "total_chars": 12,
                "cer": 0.0 if engine == "sharp" else 1.0 / 12.0,
                "char_acc": 1.0 if engine == "sharp" else 11.0 / 12.0,
            },
        }
        if engine == "c":
            meta.update(
                {
                    "c_assets_mode": "external",
                    "lw_commit": LW_COMMIT,
                    "lw_dll_sha256": hashlib.sha256(
                        f"dll-{replica}".encode()
                    ).hexdigest(),
                    "det_sha256": MODEL_HASHES["det.lwm"],
                    "cls_sha256": MODEL_HASHES["cls.lwm"],
                    "rec_sha256": MODEL_HASHES["rec.lwm"],
                    "dictionary_sha256": MODEL_HASHES["ppocr_keys.txt"],
                }
            )
        median = 10.0 if engine == "sharp" else 12.0
        return {
            "meta": meta,
            "summary": {
                "n": 2,
                "warmup": 1,
                "total_ms": {
                    "mean": median,
                    "median": median,
                    "p95": median + 1.0,
                },
            },
            "rows": rows,
        }

    def build(self) -> dict[str, object]:
        return build_report(
            self.contract,
            self.results,
            self.dataset,
            self.model_sums,
            self.output,
            LW_COMMIT,
            include_product=True,
            keep_detailed_results=True,
        )

    def test_builds_paired_report_and_disagreements(self) -> None:
        manifest = self.build()
        self.assertEqual(manifest["cases"], 36)
        self.assertEqual(set(manifest["c_runtime_dlls"]), {"1", "2", "3"})
        summary = (self.output / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("Ratios are calculated only within the same replica", summary)
        self.assertIn("1.200x", summary)
        disagreements = json.loads(
            (self.output / "disagreements" / "sharp-correct-c-wrong.json")
            .read_text(encoding="utf-8")
        )
        self.assertGreater(len(disagreements["cases"]), 0)
        line_cases = json.loads(
            (self.output / "disagreements" / "line-cases.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(line_cases["alignment"], "reference-line-dp")
        self.assertGreater(len(line_cases["cases"]), 0)
        contributors = json.loads(
            (self.output / "disagreements" / "cer-contributors.json")
            .read_text(encoding="utf-8")
        )
        self.assertGreater(len(contributors["cases"]), 0)
        self.assertIn("Executive summary", summary)
        disagreement_summary = (
            self.output / "disagreements" / "SUMMARY.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Line-level accuracy diagnostics", disagreement_summary)

    def test_line_alignment_keeps_following_lines_after_insertion(self) -> None:
        aligned, extras = align_lines(
            ["A", "B", "C"], ["A", "inserted", "B", "C"]
        )
        self.assertEqual(aligned, [(0, 0), (1, 2), (2, 3)])
        self.assertEqual(extras, [1])

    def test_missing_case_is_rejected(self) -> None:
        missing = next(self.results.rglob("*-sharp-r1.json"))
        missing.unlink()
        with self.assertRaisesRegex(ComparisonError, "case is missing"):
            self.build()

    def test_model_asset_mismatch_is_rejected(self) -> None:
        target = next(self.results.rglob("*-c-r1.json"))
        value = json.loads(target.read_text(encoding="utf-8"))
        value["meta"]["det_sha256"] = "f" * 64
        target.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "asset mismatch"):
            self.build()

    def test_normalized_isa_mismatch_is_rejected(self) -> None:
        target = next(self.results.rglob("normalized-fixed320-1w-sharp-r1.json"))
        value = json.loads(target.read_text(encoding="utf-8"))
        value["meta"]["effective_isa"] = "avx512"
        target.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "not AVX2"):
            self.build()

    def test_adaptive_normalized_isa_mismatch_is_rejected(self) -> None:
        target = next(
            self.results.rglob("normalized-adaptive960-1w-sharp-r1.json")
        )
        value = json.loads(target.read_text(encoding="utf-8"))
        value["meta"]["effective_isa"] = "avx512"
        target.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "not AVX2"):
            self.build()

    def test_product_profile_can_be_omitted(self) -> None:
        for manifest_path in self.results.rglob("replica-manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            product_cases = [
                item for item in manifest["cases"] if "product" in item
            ]
            for relative in product_cases:
                (manifest_path.parent / relative).unlink()
            manifest["cases"] = [
                item for item in manifest["cases"] if "product" not in item
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = build_report(
            self.contract,
            self.results,
            self.dataset,
            self.model_sums,
            self.output,
            LW_COMMIT,
            include_product=False,
            keep_detailed_results=False,
        )
        self.assertEqual(result["cases"], 24)
        self.assertEqual(
            result["profiles"], ["normalized-fixed320", "normalized-adaptive960"]
        )

    def test_repository_contract_pins_published_harness_commit(self) -> None:
        contract_path = (
            Path(__file__).resolve().parents[1]
            / "ci"
            / "csharp-engine-comparison.json"
        )
        contract = load_contract(contract_path)
        self.assertEqual(
            contract["simd_ref"],
            "3e4192f2ec03b84701d3c5c1658e277fdaa7aa12",
        )


if __name__ == "__main__":
    unittest.main()
