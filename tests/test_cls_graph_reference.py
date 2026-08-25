from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_input() -> np.ndarray:
    count = 3 * 80 * 160
    integers = ((np.arange(count, dtype=np.int64) * 19) % 263) - 131
    return (integers.astype(np.float32) / np.float32(131.0)).reshape(1, 3, 80, 160)


class ClsGraphReferenceTest(unittest.TestCase):
    driver: Path
    lwm_model: Path
    onnx_model: Path

    def test_graph_matches_onnxruntime(self) -> None:
        reference = ort.InferenceSession(
            str(self.onnx_model), providers=["CPUExecutionProvider"]
        )
        input_tensor = make_input()
        expected = np.asarray(
            reference.run(None, {reference.get_inputs()[0].name: input_tensor})[0],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "actual.f32"
            completed = subprocess.run(
                [str(self.driver), str(self.lwm_model), str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
        difference = np.abs(actual - expected)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-5)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=5)
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
    ClsGraphReferenceTest.driver = arguments.driver
    ClsGraphReferenceTest.lwm_model = arguments.lwm_model
    ClsGraphReferenceTest.onnx_model = arguments.onnx_model
    unittest.main(argv=[__file__], verbosity=2)
