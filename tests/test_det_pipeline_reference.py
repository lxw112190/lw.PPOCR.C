from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def make_source(width: int, height: int, stride: int) -> tuple[np.ndarray, bytes]:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    raw = bytearray([0xA5] * (stride * height))
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                value = (y * 41 + x * 19 + channel * 67 + 13) % 256
                pixels[y, x, channel] = value
                raw[y * stride + x * 3 + channel] = value
    return pixels, bytes(raw)


def resize_reference(source: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width, _ = source.shape
    output = np.empty((3, height, width), dtype=np.float32)
    means = (0.485, 0.456, 0.406)
    inverse_stds = (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)
    for output_y in range(height):
        source_y = (output_y + 0.5) * source_height / height - 0.5
        y0_raw = int(np.floor(source_y))
        y1_raw = y0_raw + 1
        y0 = min(max(y0_raw, 0), source_height - 1)
        y1 = min(max(y1_raw, 0), source_height - 1)
        wy = source_y - y0_raw
        for output_x in range(width):
            source_x = (output_x + 0.5) * source_width / width - 0.5
            x0_raw = int(np.floor(source_x))
            x1_raw = x0_raw + 1
            x0 = min(max(x0_raw, 0), source_width - 1)
            x1 = min(max(x1_raw, 0), source_width - 1)
            wx = source_x - x0_raw
            top = source[y0, x0].astype(np.float64) + (
                source[y0, x1].astype(np.float64) - source[y0, x0]
            ) * wx
            bottom = source[y1, x0].astype(np.float64) + (
                source[y1, x1].astype(np.float64) - source[y1, x0]
            ) * wx
            value = (top + (bottom - top) * wy) / 255.0
            for channel in range(3):
                output[channel, output_y, output_x] = np.float32(
                    (value[channel] - means[channel]) * inverse_stds[channel]
                )
    return output


class DetPipelineReferenceTest(unittest.TestCase):
    driver: Path
    model: Path
    sample: Path

    def run_driver(self, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.driver), *args], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout
        )

    def test_preprocess_postprocess_and_public_pipeline(self) -> None:
        width, height, stride = 65, 37, 200
        pixels, raw = make_source(width, height, stride)
        expected = resize_reference(pixels, 64, 32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bgr"
            output = root / "det-input.f32"
            source.write_bytes(raw)
            preprocessed = self.run_driver([
                "preprocess", str(source), str(width), str(height),
                str(stride), "64", str(output)
            ])
            self.assertEqual(preprocessed.returncode, 0, preprocessed.stdout + preprocessed.stderr)
            actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)

            sample_rgb = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)
            sample_bgr = sample_rgb[:, :, ::-1].copy()
            sample_raw = root / "sample.bgr"
            sample_raw.write_bytes(sample_bgr.tobytes())
            pipeline = self.run_driver([
                "pipeline", str(self.model), str(sample_raw),
                str(sample_bgr.shape[1]), str(sample_bgr.shape[0]),
                str(sample_bgr.shape[1] * 3)
            ], timeout=600)
        postprocessed = self.run_driver(["postprocess"])
        self.assertEqual(postprocessed.returncode, 0, postprocessed.stdout + postprocessed.stderr)
        synthetic = re.search(r"count=1 score=([^ ]+)", postprocessed.stdout)
        self.assertIsNotNone(synthetic, postprocessed.stdout)
        assert synthetic is not None
        self.assertAlmostEqual(float(synthetic.group(1)), 0.9, places=6)
        self.assertEqual(pipeline.returncode, 0, pipeline.stdout + pipeline.stderr)
        header = re.search(r"count=(\d+) width=(\d+) height=(\d+)", pipeline.stdout)
        self.assertIsNotNone(header, pipeline.stdout)
        assert header is not None
        count = int(header.group(1))
        self.assertEqual(count, 16)
        self.assertEqual((int(header.group(2)), int(header.group(3))), (320, 320))
        lines = re.findall(r"score=([^ ]+) points=([^\r\n]+)", pipeline.stdout)
        self.assertEqual(len(lines), count)
        expected_y1 = [
            33.52, 77.84, 108.75, 141.21, 174.01, 205.34, 236.57,
            231.98, 269.07, 299.06, 330.22, 361.52, 363.48, 392.71,
            423.95, 455.64,
        ]
        previous_y = -1.0
        for index, (score_text, points_text) in enumerate(lines):
            score = float(score_text)
            points = [float(value) for value in points_text.split(",")]
            self.assertGreaterEqual(score, 0.6)
            self.assertLessEqual(score, 1.0)
            self.assertEqual(len(points), 8)
            self.assertTrue(all(0.0 <= value <= 499.0 for value in points))
            y = points[1]
            self.assertAlmostEqual(y, expected_y1[index], delta=1.0)
            self.assertGreaterEqual(y + 10.0, previous_y)
            previous_y = y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    DetPipelineReferenceTest.driver = arguments.driver
    DetPipelineReferenceTest.model = arguments.model
    DetPipelineReferenceTest.sample = arguments.sample
    unittest.main(argv=[__file__], verbosity=2)
