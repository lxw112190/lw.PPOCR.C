from __future__ import annotations

import argparse
import json
import subprocess
import unittest


class RecProfileTest(unittest.TestCase):
    def test_profile_covers_every_rec_node(self) -> None:
        completed = subprocess.run(
            [ARGUMENTS.driver, ARGUMENTS.model, "320", "2"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["width"], 320)
        self.assertEqual(report["iterations"], 2)
        operators = report["operators"]
        self.assertEqual(len(operators), 15)
        self.assertEqual(sum(item["invocations"] for item in operators), 161 * 2)
        self.assertGreater(sum(item["nanoseconds"] for item in operators), 0)
        self.assertEqual(
            {item["name"] for item in operators if item["invocations"] > 0},
            {
                "Conv",
                "Add",
                "Mul",
                "Div",
                "Erf",
                "HardSigmoid",
                "BatchNormalization",
                "ReduceMean",
                "Relu",
                "AveragePool",
                "Squeeze",
                "Transpose",
                "Unsqueeze",
                "MatMul",
                "Softmax",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
