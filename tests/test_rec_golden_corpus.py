from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from test_rec_pipeline_reference import ctc_reference, preprocess_reference


def parse_result(stdout: str) -> tuple[float, int]:
    match = re.search(r"score=([^ ]+) chars=(\d+)", stdout)
    if match is None:
        raise AssertionError(f"missing result metadata: {stdout!r}")
    return float(match.group(1)), int(match.group(2))


class RecGoldenCorpusTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path
    dictionary: Path
    sample: Path
    corpus: Path

    def test_ten_real_crops_match_manifest_and_onnxruntime(self) -> None:
        manifest = json.loads(self.corpus.read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("schema_version"), 1)
        cases = manifest.get("cases")
        self.assertIsInstance(cases, list)
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(len({case["name"] for case in cases}), len(cases))
        self.assertEqual(
            hashlib.sha256(self.sample.read_bytes()).hexdigest(),
            manifest.get("source_sha256"),
        )

        rgb = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)
        session = ort.InferenceSession(
            str(self.onnx_model), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        dictionary_labels = self.dictionary.read_text(encoding="utf-8").splitlines()
        labels = ["", *dictionary_labels, " "]
        target_width = int(manifest["target_width"])

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for index, case in enumerate(cases):
                with self.subTest(case=case["name"]):
                    x1, y1, x2, y2 = (int(value) for value in case["box"])
                    self.assertTrue(0 <= x1 < x2 <= rgb.shape[1])
                    self.assertTrue(0 <= y1 < y2 <= rgb.shape[0])
                    bgr = np.ascontiguousarray(rgb[y1:y2, x1:x2, ::-1])
                    input_tensor, _ = preprocess_reference(bgr, target_width)
                    probabilities = session.run(
                        None, {input_name: input_tensor[np.newaxis, ...]}
                    )[0][0]
                    reference_text, reference_score, reference_count = ctc_reference(
                        probabilities, labels
                    )
                    self.assertEqual(reference_text, case["text"])

                    source_path = temporary / f"{index:02d}-{case['name']}.bgr"
                    output_path = temporary / f"{index:02d}-{case['name']}.txt"
                    source_path.write_bytes(bgr.tobytes(order="C"))
                    completed = subprocess.run(
                        [
                            str(self.driver),
                            "pipeline",
                            str(self.lwm_model),
                            str(self.dictionary),
                            str(source_path),
                            str(bgr.shape[1]),
                            str(bgr.shape[0]),
                            str(bgr.shape[1] * 3),
                            str(target_width),
                            str(output_path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=180,
                    )
                    self.assertEqual(
                        completed.returncode, 0, completed.stdout + completed.stderr
                    )
                    actual_text = output_path.read_text(encoding="utf-8")
                    actual_score, actual_count = parse_result(completed.stdout)
                    self.assertEqual(actual_text, reference_text)
                    self.assertAlmostEqual(actual_score, reference_score, places=5)
                    self.assertEqual(actual_count, reference_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    RecGoldenCorpusTest.driver = arguments.driver
    RecGoldenCorpusTest.lwm_model = arguments.lwm_model
    RecGoldenCorpusTest.onnx_model = arguments.onnx_model
    RecGoldenCorpusTest.dictionary = arguments.dictionary
    RecGoldenCorpusTest.sample = arguments.sample
    RecGoldenCorpusTest.corpus = arguments.corpus
    unittest.main(argv=[__file__], verbosity=2)
