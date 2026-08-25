from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_input(height: int, width: int) -> np.ndarray:
    count = 3 * height * width
    integers = ((np.arange(count, dtype=np.int64) * 23) % 269) - 134
    return (integers.astype(np.float32) / np.float32(134.0)).reshape(1, 3, height, width)


class DetGraphReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path

    def test_graph_matches_onnxruntime_at_dynamic_shapes(self) -> None:
        reference = ort.InferenceSession(str(self.onnx_model), providers=["CPUExecutionProvider"])
        for height, width in ((32, 32), (32, 64)):
            with self.subTest(shape=(height, width)):
                input_tensor = make_input(height, width)
                expected = np.asarray(
                    reference.run(None, {reference.get_inputs()[0].name: input_tensor})[0],
                    dtype=np.float32,
                )
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "actual.f32"
                    completed = subprocess.run(
                        [str(self.driver), str(self.lwm_model), str(height), str(width), str(output_path)],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
                difference = np.abs(actual - expected)
                np.testing.assert_allclose(actual, expected, rtol=3.0e-3, atol=3.0e-5)
                self.assertTrue(np.all((actual >= 0.0) & (actual <= 1.0)))
                print(
                    f"backend=onnxruntime shape={actual.shape} "
                    f"max_abs={float(difference.max()):.8g} "
                    f"mean_abs={float(difference.mean()):.8g}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    DetGraphReferenceTest.driver = arguments.driver
    DetGraphReferenceTest.lwm_model = arguments.lwm_model
    DetGraphReferenceTest.onnx_model = arguments.onnx_model
    unittest.main(argv=[__file__], verbosity=2)
