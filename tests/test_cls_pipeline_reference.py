from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_bgr_source(width: int, height: int, stride: int) -> tuple[np.ndarray, bytes]:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    raw = bytearray([0xEE] * (stride * height))
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = (y * 53 + x * 29 + channel * 71 + 17) % 256
                pixels[y, x, channel] = value
                raw[y * stride + x * 3 + channel] = value
    return pixels, bytes(raw)


def preprocess_reference(source: np.ndarray) -> tuple[np.ndarray, int]:
    source_height, source_width, _ = source.shape
    resized_width = min(160, (80 * source_width + source_height - 1) // source_height)
    output = np.full((3, 80, 160), np.float32(-1.0))
    for output_y in range(80):
        source_y = (output_y + 0.5) * source_height / 80.0 - 0.5
        source_y0_raw = int(np.floor(source_y))
        source_y1_raw = source_y0_raw + 1
        source_y0 = min(max(source_y0_raw, 0), source_height - 1)
        source_y1 = min(max(source_y1_raw, 0), source_height - 1)
        weight_y = source_y - source_y0_raw
        for output_x in range(resized_width):
            source_x = (output_x + 0.5) * source_width / resized_width - 0.5
            source_x0_raw = int(np.floor(source_x))
            source_x1_raw = source_x0_raw + 1
            source_x0 = min(max(source_x0_raw, 0), source_width - 1)
            source_x1 = min(max(source_x1_raw, 0), source_width - 1)
            weight_x = source_x - source_x0_raw
            top = source[source_y0, source_x0].astype(np.float64) + (
                source[source_y0, source_x1].astype(np.float64)
                - source[source_y0, source_x0].astype(np.float64)
            ) * weight_x
            bottom = source[source_y1, source_x0].astype(np.float64) + (
                source[source_y1, source_x1].astype(np.float64)
                - source[source_y1, source_x0].astype(np.float64)
            ) * weight_x
            value = top + (bottom - top) * weight_y
            output[:, output_y, output_x] = (
                value * (2.0 / 255.0) - 1.0
            ).astype(np.float32)
    return output, resized_width


class ClsPipelineReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path

    def run_driver(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.driver), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )

    def test_preprocess_and_public_api_match_references(self) -> None:
        width, height, stride = 7, 5, 24
        pixels, raw = make_bgr_source(width, height, stride)
        expected_input, expected_width = preprocess_reference(pixels)
        reference = ort.InferenceSession(
            str(self.onnx_model), providers=["CPUExecutionProvider"]
        )
        probabilities = reference.run(
            None,
            {reference.get_inputs()[0].name: expected_input[np.newaxis, ...]},
        )[0][0]
        expected_label = int(np.argmax(probabilities))
        expected_score = float(probabilities[expected_label])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.bgr"
            output_path = root / "preprocessed.f32"
            source_path.write_bytes(raw)
            preprocessed = self.run_driver(
                [
                    "preprocess",
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                    str(output_path),
                ]
            )
            self.assertEqual(
                preprocessed.returncode, 0, preprocessed.stdout + preprocessed.stderr
            )
            actual_input = np.fromfile(output_path, dtype="<f4").reshape(
                expected_input.shape
            )
            unicode_dir = root / "分类模型"
            unicode_dir.mkdir()
            unicode_model = unicode_dir / "方向分类.lwm"
            shutil.copyfile(self.lwm_model, unicode_model)
            classified = self.run_driver(
                [
                    "pipeline",
                    str(unicode_model),
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                ]
            )
        np.testing.assert_allclose(actual_input, expected_input, rtol=0.0, atol=1.0e-6)
        self.assertTrue(np.all(actual_input[:, :, expected_width:] == np.float32(-1.0)))
        self.assertIn(f"resized_width={expected_width}", preprocessed.stdout)
        self.assertEqual(classified.returncode, 0, classified.stdout + classified.stderr)
        match = re.search(
            r"label=(\d+) score=([^ ]+) orientation=(\d+) resized_width=(\d+)",
            classified.stdout,
        )
        self.assertIsNotNone(match, classified.stdout)
        assert match is not None
        self.assertEqual(int(match.group(1)), expected_label)
        self.assertAlmostEqual(float(match.group(2)), expected_score, places=5)
        self.assertEqual(int(match.group(3)), expected_label * 180)
        self.assertEqual(int(match.group(4)), expected_width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    ClsPipelineReferenceTest.driver = arguments.driver
    ClsPipelineReferenceTest.lwm_model = arguments.lwm_model
    ClsPipelineReferenceTest.onnx_model = arguments.onnx_model
    unittest.main(argv=[__file__], verbosity=2)
