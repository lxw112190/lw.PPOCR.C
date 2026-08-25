from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_input(width: int) -> np.ndarray:
    count = 3 * 48 * width
    integers = ((np.arange(count, dtype=np.int64) * 17) % 257) - 128
    return (integers.astype(np.float32) / np.float32(127.0)).reshape(1, 3, 48, width)


class RecGraphReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path
    reference: ort.InferenceSession
    input_name: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = ort.InferenceSession(
            str(cls.onnx_model), providers=["CPUExecutionProvider"]
        )
        cls.input_name = cls.reference.get_inputs()[0].name

    def run_reference(self, input_tensor: np.ndarray) -> np.ndarray:
        return self.reference.run(None, {self.input_name: input_tensor})[0]

    def check_width(self, width: int) -> None:
        input_tensor = make_input(width)
        expected = np.asarray(self.run_reference(input_tensor), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "actual.f32"
            completed = subprocess.run(
                [str(self.driver), str(self.lwm_model), str(width), str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
        difference = np.abs(actual - expected)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-5)
        print(
            f"width={width} backend=onnxruntime shape={actual.shape} "
            f"max_abs={float(difference.max()):.8g} mean_abs={float(difference.mean()):.8g}"
        )

    def test_minimum_width(self) -> None:
        self.check_width(7)

    def test_second_dynamic_width(self) -> None:
        self.check_width(17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--lwm-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    RecGraphReferenceTest.driver = arguments.driver
    RecGraphReferenceTest.lwm_model = arguments.lwm_model
    RecGraphReferenceTest.onnx_model = arguments.onnx_model
    unittest.main(argv=[__file__], verbosity=2)
