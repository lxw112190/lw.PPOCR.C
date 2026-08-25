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
from PIL import Image


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


def preprocess_reference(source: np.ndarray, target_width: int) -> tuple[np.ndarray, int]:
    source_height, source_width, _ = source.shape
    resized_width = min(
        target_width, (48 * source_width + source_height - 1) // source_height
    )
    output = np.full(
        (3, 48, target_width), np.float32(128.0 * (2.0 / 255.0) - 1.0)
    )
    for output_y in range(48):
        source_y = (output_y + 0.5) * source_height / 48.0 - 0.5
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
            top = (
                source[source_y0, source_x0].astype(np.float64)
                + (
                    source[source_y0, source_x1].astype(np.float64)
                    - source[source_y0, source_x0].astype(np.float64)
                )
                * weight_x
            )
            bottom = (
                source[source_y1, source_x0].astype(np.float64)
                + (
                    source[source_y1, source_x1].astype(np.float64)
                    - source[source_y1, source_x0].astype(np.float64)
                )
                * weight_x
            )
            value = top + (bottom - top) * weight_y
            output[:, output_y, output_x] = (value * (2.0 / 255.0) - 1.0).astype(
                np.float32
            )
    return output, resized_width


def ctc_reference(
    probabilities: np.ndarray, labels: list[str]
) -> tuple[str, float, int]:
    text: list[str] = []
    scores: list[float] = []
    previous = 0
    for step, row in enumerate(probabilities):
        index = int(np.argmax(row))
        if index > 0 and (step == 0 or index != previous):
            text.append(labels[index])
            scores.append(float(row[index]))
        previous = index
    return "".join(text), (sum(scores) / len(scores) if scores else 0.0), len(scores)


def parse_result(stdout: str) -> tuple[float, int, int]:
    match = re.search(r"score=([^ ]+) chars=(\d+) bytes=(\d+)", stdout)
    if match is None:
        raise AssertionError(f"missing result metadata: {stdout!r}")
    return float(match.group(1)), int(match.group(2)), int(match.group(3))


class RecPipelineReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path
    dictionary: Path
    sample: Path

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

    def test_preprocess_matches_independent_bilinear_reference(self) -> None:
        width, height, stride, target_width = 7, 5, 24, 80
        pixels, raw = make_bgr_source(width, height, stride)
        expected, expected_resized_width = preprocess_reference(pixels, target_width)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.bgr"
            output_path = Path(directory) / "output.f32"
            source_path.write_bytes(raw)
            completed = self.run_driver(
                [
                    "preprocess",
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                    str(target_width),
                    str(output_path),
                ]
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn(f"resized_width={expected_resized_width}", completed.stdout)
            actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)
        padding_value = np.float32(128.0 * (2.0 / 255.0) - 1.0)
        self.assertTrue(np.all(actual[:, :, expected_resized_width:] == padding_value))

    def test_ctc_utf8_blank_repeat_space_and_buffer_contract(self) -> None:
        labels = ["", "你", "A", "é", " "]
        selected = [0, 1, 1, 0, 1, 2, 2, 4, 3, 0]
        confidence = [0.90, 0.80, 0.70, 0.95, 0.85, 0.60, 0.65, 0.75, 0.88, 0.90]
        probabilities = np.full((len(selected), len(labels)), 0.01, dtype=np.float32)
        for step, (index, value) in enumerate(zip(selected, confidence)):
            probabilities[step, index] = np.float32(value)
        expected_text, expected_score, expected_count = ctc_reference(probabilities, labels)
        with tempfile.TemporaryDirectory() as directory:
            dictionary_path = Path(directory) / "dict.txt"
            probability_path = Path(directory) / "probabilities.f32"
            output_path = Path(directory) / "text.txt"
            dictionary_path.write_bytes("\ufeff你\r\nA\r\né\r\n".encode("utf-8"))
            probabilities.astype("<f4").tofile(probability_path)
            completed = self.run_driver(
                [
                    "decode",
                    str(dictionary_path),
                    str(probability_path),
                    str(probabilities.shape[0]),
                    str(probabilities.shape[1]),
                    str(output_path),
                ]
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            actual_text = output_path.read_text(encoding="utf-8")
            actual_score, actual_count, actual_bytes = parse_result(completed.stdout)
            invalid_dictionary = Path(directory) / "invalid.txt"
            invalid_dictionary.write_bytes(b"\xff\n")
            rejected = self.run_driver(
                [
                    "decode",
                    str(invalid_dictionary),
                    str(probability_path),
                    str(probabilities.shape[0]),
                    str(probabilities.shape[1]),
                    str(output_path),
                ]
            )
            empty_dictionary = Path(directory) / "empty.txt"
            empty_dictionary.write_bytes(b"\n")
            rejected_empty = self.run_driver(
                [
                    "decode",
                    str(empty_dictionary),
                    str(probability_path),
                    str(probabilities.shape[0]),
                    str(probabilities.shape[1]),
                    str(output_path),
                ]
            )
            invalid_probabilities = probabilities.copy()
            invalid_probabilities[0, 0] = np.nan
            invalid_probability_path = Path(directory) / "invalid.f32"
            invalid_probabilities.astype("<f4").tofile(invalid_probability_path)
            rejected_non_finite = self.run_driver(
                [
                    "decode",
                    str(dictionary_path),
                    str(invalid_probability_path),
                    str(probabilities.shape[0]),
                    str(probabilities.shape[1]),
                    str(output_path),
                ]
            )
        self.assertEqual(actual_text, expected_text)
        self.assertAlmostEqual(actual_score, expected_score, places=7)
        self.assertEqual(actual_count, expected_count)
        self.assertEqual(actual_bytes, len(expected_text.encode("utf-8")))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("invalid_format", rejected.stderr)
        self.assertNotEqual(rejected_empty.returncode, 0)
        self.assertIn("invalid_format", rejected_empty.stderr)
        self.assertNotEqual(rejected_non_finite.returncode, 0)
        self.assertIn("invalid_argument", rejected_non_finite.stderr)

    def test_preprocess_model_ctc_matches_onnxruntime(self) -> None:
        target_width = 320
        rgb = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)
        pixels = np.ascontiguousarray(rgb[30:76, 20:315, ::-1])
        height, width, _ = pixels.shape
        stride = width * 3
        raw = pixels.tobytes(order="C")
        input_tensor, _ = preprocess_reference(pixels, target_width)
        session = ort.InferenceSession(
            str(self.onnx_model), providers=["CPUExecutionProvider"]
        )
        probabilities = session.run(
            None, {session.get_inputs()[0].name: input_tensor[np.newaxis, ...]}
        )[0][0]
        dictionary_labels = self.dictionary.read_text(encoding="utf-8").splitlines()
        labels = ["", *dictionary_labels, " "]
        expected_text, expected_score, expected_count = ctc_reference(probabilities, labels)
        self.assertEqual(expected_text, "纯臻营养护发素")
        with tempfile.TemporaryDirectory() as directory:
            unicode_assets = Path(directory) / "模型与字典"
            unicode_assets.mkdir()
            lwm_model = unicode_assets / "识别模型.lwm"
            dictionary = unicode_assets / "字典.txt"
            shutil.copyfile(self.lwm_model, lwm_model)
            shutil.copyfile(self.dictionary, dictionary)
            source_path = Path(directory) / "source.bgr"
            output_path = Path(directory) / "text.txt"
            source_path.write_bytes(raw)
            completed = self.run_driver(
                [
                    "pipeline",
                    str(lwm_model),
                    str(dictionary),
                    str(source_path),
                    str(width),
                    str(height),
                    str(stride),
                    str(target_width),
                    str(output_path),
                ]
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            actual_text = output_path.read_text(encoding="utf-8")
            actual_score, actual_count, actual_bytes = parse_result(completed.stdout)
        self.assertEqual(actual_text, expected_text)
        self.assertAlmostEqual(actual_score, expected_score, places=5)
        self.assertEqual(actual_count, expected_count)
        self.assertEqual(actual_bytes, len(expected_text.encode("utf-8")))
        print(
            f"pipeline width={target_width} "
            f"text={actual_text.encode('unicode_escape').decode('ascii')!r} "
            f"score={actual_score:.8g} chars={actual_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    RecPipelineReferenceTest.driver = arguments.driver
    RecPipelineReferenceTest.lwm_model = arguments.lwm_model
    RecPipelineReferenceTest.onnx_model = arguments.onnx_model
    RecPipelineReferenceTest.dictionary = arguments.dictionary
    RecPipelineReferenceTest.sample = arguments.sample
    unittest.main(argv=[__file__], verbosity=2)
